from vllm import LLM, SamplingParams
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from tree_reward import TreeRewardManager
from util.reward_func import reward
from util.debug import debug
from util.opsd import change_prompts
from util.entropy import calculate_entropy_from_logits
import torch

class OPSD:
    @staticmethod
    def calculate_entropy_of_teacher(
        hf_model: AutoModelForCausalLM,
        sequences: list[torch.Tensor],
        reward_masks: list[torch.Tensor],
        pad_token_id: int,
        teacher_prefixes: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        For each student sequence:

            student:
                [student prefix] [completion]
                0 0 0 0 0        1 1 1 1 1

            teacher input:
                [longer teacher prefix] [same completion]
                0 0 0 0 0 0 0           1 1 1 1 1

        The teacher entropy over the completion is then mapped back to the
        student's shifted-token positions:

            student-aligned entropy:
                0 0 0 0 [teacher completion entropies]

        Returns:
            list[Tensor], where each tensor has length:

                len(student_sequence) - 1

            because logits at position t predict token t + 1.
        """
        if not (
            len(sequences)
            == len(reward_masks)
            == len(teacher_prefixes)
        ):
            raise ValueError(
                "sequences, reward_masks, and teacher_prefixes "
                "must have the same batch size."
            )

        device = next(hf_model.parameters()).device
        
        # Move everything to device
        sequences = [seq.to(device) for seq in sequences]
        reward_masks = [mask.to(device) for mask in reward_masks]
        teacher_prefixes = [prefix.to(device) for prefix in teacher_prefixes]

        teacher_sequences: list[torch.Tensor] = []
        teacher_reward_masks: list[torch.Tensor] = []

        # Information needed to map teacher entropy back to student positions.
        student_shift_masks: list[torch.Tensor] = []

        for i, (student_sequence, student_reward_mask, teacher_prefix) in enumerate(
            zip(sequences, reward_masks, teacher_prefixes)
        ):
            student_sequence = student_sequence.long().flatten()
            student_reward_mask = student_reward_mask.bool().flatten()
            teacher_prefix = teacher_prefix.long().flatten()

            if student_sequence.numel() != student_reward_mask.numel():
                raise ValueError(
                    f"Sample {i}: student sequence length "
                    f"{student_sequence.numel()} does not match reward-mask "
                    f"length {student_reward_mask.numel()}."
                )

            reward_positions = torch.nonzero(
                student_reward_mask,
                as_tuple=False,
            ).flatten()

            if reward_positions.numel() == 0:
                raise ValueError(
                    f"Sample {i}: reward mask contains no completion tokens."
                )

            completion_start = int(reward_positions[0].item())

            # Assumes the reward region is the completion suffix.
            if not student_reward_mask[completion_start:].all():
                raise ValueError(
                    f"Sample {i}: reward mask must be a contiguous suffix."
                )

            completion_ids = student_sequence[completion_start:]

            # Teacher sees privileged prefix + the exact same completion.
            #debug("teacher_prefix: ", teacher_prefix.device)
            #debug("completion_ids: ", completion_ids.device)
            
            teacher_sequence = torch.cat(
                [teacher_prefix, completion_ids],
                dim=0,
            )

            teacher_reward_mask = torch.cat(
                [
                    torch.zeros(
                        teacher_prefix.numel(),
                        dtype=torch.bool,
                    ),
                    torch.ones(
                        completion_ids.numel(),
                        dtype=torch.bool,
                    ),
                ],
                dim=0,
            )

            teacher_sequences.append(teacher_sequence)
            teacher_reward_masks.append(teacher_reward_mask)

            # Shifted because logits[:, t] predict token at position t + 1.
            student_shift_masks.append(student_reward_mask[1:])

        teacher_input_ids = pad_sequence(
            teacher_sequences,
            batch_first=True,
            padding_value=pad_token_id,
        )

        teacher_reward_mask = pad_sequence(
            teacher_reward_masks,
            batch_first=True,
            padding_value=False,
        )

        teacher_lengths = torch.tensor(
            [sequence.numel() for sequence in teacher_sequences],
            dtype=torch.long,
        )

        positions = torch.arange(
            teacher_input_ids.size(1),
            dtype=torch.long,
        ).unsqueeze(0)

        teacher_attention_mask = (
            positions < teacher_lengths.unsqueeze(1)
        )

        teacher_input_ids = teacher_input_ids.to(device)
        teacher_attention_mask = teacher_attention_mask.to(device)
        teacher_reward_mask = teacher_reward_mask.to(device)

        with torch.no_grad():
            outputs = hf_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask.long(),
                use_cache=False,
                return_dict=True,
            )

            # logits[:, t] predicts input_ids[:, t + 1]
            shift_teacher_logits = outputs.logits[:, :-1, :]

            teacher_entropies = calculate_entropy_from_logits(
                shift_teacher_logits
            )

            # Select entropies according to the token being predicted.
            shift_teacher_reward_mask = teacher_reward_mask[:, 1:]
            shift_teacher_attention_mask = teacher_attention_mask[:, 1:]

            valid_teacher_completion_mask = (
                shift_teacher_reward_mask
                & shift_teacher_attention_mask
            )

        student_aligned_entropies: list[torch.Tensor] = []

        for i, student_shift_mask in enumerate(student_shift_masks):
            # These correspond exactly to the shared completion tokens.
            completion_entropies = teacher_entropies[i][
                valid_teacher_completion_mask[i]
            ]

            expected_completion_tokens = int(
                student_shift_mask.sum().item()
            )

            if completion_entropies.numel() != expected_completion_tokens:
                raise RuntimeError(
                    f"Sample {i}: teacher produced "
                    f"{completion_entropies.numel()} completion entropies, "
                    f"but student alignment expects "
                    f"{expected_completion_tokens}."
                )

            # Output is aligned with the student's shifted logits.
            #
            # Example:
            # student mask:         [0, 0, 0, 0, 1, 1, 1]
            # shifted mask:            [0, 0, 0, 1, 1, 1]
            # returned entropy:         [0, 0, 0, H, H, H]
            aligned_entropy = torch.zeros(
                student_shift_mask.numel(),
                dtype=teacher_entropies.dtype,
                device=device,
            )

            aligned_entropy[
                student_shift_mask.to(device)
            ] = completion_entropies

            student_aligned_entropies.append(aligned_entropy)
            
        student_aligned_entropies = pad_sequence(
            student_aligned_entropies,
            batch_first=True,
            padding_value=0.0,
        )

        return student_aligned_entropies
    
    def handle_batch(vllm: LLM,
                     teacher_messages: List[str],
                     teacher_reference_answers: List[str],
                     teacher_n_branches: int,
                     teacher_gts: List[str],
                     manager: TreeRewardManager,
                     input_ids):
        # Generate teacher's answers
        debug("*"*40,"Start Teacher message", "*"*40)
        debug(teacher_messages)
        debug(teacher_reference_answers)
        teacher_new_messages = change_prompts(
            teacher_messages,
            teacher_reference_answers
        )
        
        teacher_accuracies, prompts_token_ids = OPSD.generate_teacher_answers(
            vllm = vllm,
            prompts = teacher_new_messages,
            ground_truths = teacher_gts,
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                #TODO: Calculate max_tokens
                max_tokens=512,
                n = teacher_n_branches,
                detokenize=True,
                prompt_logprobs=0,
            )
        )
        
        #TODO: Use zip over here to make the code look better
        for input_ids_, prompt_token_ids, teacher_acc in zip(input_ids, prompts_token_ids, teacher_accuracies):
            manager.add_teacher_ids(
                input_ids=input_ids_,
                teacher_ids=prompt_token_ids,
                teacher_acc=teacher_acc
            )
            
        debug("*"*40,"End Teacher message", "*"*40)
    
    def generate_teacher_answers(vllm: LLM, 
                                 prompts: List[str],
                                 ground_truths: List[str],
                                 sampling_params: SamplingParams):
        
        outputs = vllm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
        )

        accuracies = []
        prompts_token_ids = []

        for i, request_output in enumerate(outputs):
            prompts_token_ids.append(request_output.prompt_token_ids)
            correct_count = 0

            for completion in request_output.outputs:
                full_answer = prompts[i] + completion.text

                reward_score = reward(
                    full_answer,
                    ground_truths[i],
                )

                correct_count += int(reward_score >= 1)

            accuracy = (
                correct_count / len(request_output.outputs)
                if request_output.outputs
                else 0.0
            )

            accuracies.append(accuracy)
            
        return (
            accuracies,
            prompts_token_ids
        )
        
        
        
        
        
        
    
        
        
    
        
