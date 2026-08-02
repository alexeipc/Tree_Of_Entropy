from transformers import AutoModelForCausalLM
from util.get_logits import get_shift_logits_and_labels
from util.entropy import calculate_entropy_from_logits
from util.ratio import get_ratio
import torch
from util.debug import debug
from typing import List


def ids_key(ids):
    if isinstance(ids, torch.Tensor):
        return tuple(ids.detach().cpu().tolist())
    return tuple(int(x) for x in ids)


class TreeRewardManager:
    class Node:
        def __init__(self, input_ids, manager):
            self.correct_answer = 0
            self.wrong_answer = 0
            self.children = []
            self.input_ids = input_ids
            self.manager = manager
            self.is_leaf = False
            self.advantage = None
            self.teacher_ids = None
            self.teacher_acc = None
            self.small_delta = 0.2
            
            self.n_branches = 1

        def add_child(self, childNode):
            self.children.append(childNode)
            
        def count_branches(self):
            # Is leaf node
            if len(self.children) == 0:
                self.n_branches = 1
            else:
                self.n_branches = sum(child.count_branches() for child in self.children)
            return self.n_branches

        def __call__(self):
            if self.correct_answer + self.wrong_answer != 1:
                self.correct_answer = 0
                self.wrong_answer = 0
                
                self.advantage = 0
                
                for child in self.children:
                    self.correct_answer += child.correct_answer
                    self.wrong_answer += child.wrong_answer
                    self.advantage += child.advantage
                    
                self.advantage /= len(self.children)

            success_rate = self.correct_answer / max(
                self.correct_answer + self.wrong_answer,
                1,
            )

            H_target = (
                self.manager.H_min
                + (1 - success_rate)
                * (self.manager.H_max - self.manager.H_min)
            )
            
            # Calculate the H_target ratio
            def clip(x: int, low: int, high: int) -> int:
                return max(low, min(x, high))
            
            def safe_ratio(
                numerator: float,
                denominator: float,
                delta: float,
                eps: float = 1e-8,
            ) -> float:
                # Both are essentially zero
                if abs(denominator) < eps:
                    if abs(numerator) < eps:
                        ratio = 1.0
                    else:
                        ratio = float("inf")
                else:
                    ratio = numerator / denominator

                return clip(ratio, 1 - delta, 1 + delta)
            
            
            h_target_ratio = safe_ratio(
                success_rate,
                self.teacher_acc,
                self.small_delta,
            )
            debug(success_rate)
            debug(self.teacher_acc)
            debug(h_target_ratio)
            
            return self.advantage, H_target, h_target_ratio

    def __init__(self):
        self.nodes = {}
        self.batch = []

        self.H_min = 0.5
        self.H_max = 3.0
        self.alpha = 0.01
        self.eps_clip = 0.2

    def add_node(self, input_ids):
        key = ids_key(input_ids)

        if key not in self.nodes:
            self.nodes[key] = TreeRewardManager.Node(input_ids, self)

        return self.nodes[key]
    
    def add_teacher_ids(self, input_ids, teacher_ids, teacher_acc):
        node = self.add_node(input_ids=input_ids)
        
        for child in node.children:
            child.teacher_ids = teacher_ids
            child.teacher_acc = teacher_acc

    def add_child(self, parent_ids, child_ids):
        parent = self.add_node(parent_ids)
        child = self.add_node(child_ids)
        parent.add_child(child)

    def traverse(
        self,
        input_ids: torch.Tensor | None = None,
        node: Node | None = None,
        parent_node: Node | None = None,
        batch:List | None = None,
        depth: int = 0
    ):
        if input_ids is not None:
            node = self.nodes[ids_key(input_ids)]
        else:
            input_ids = node.input_ids

        prev_len = 0 if parent_node is None else len(parent_node.input_ids)


        for next_node in node.children:
            if next_node is node:
                continue
            self.traverse(node=next_node, parent_node=node, batch=batch, depth=depth + 1)
        
        debug("IHI"*30)
        debug(depth)
        debug(input_ids)
        
        # If it is the prompt then do not push it to the batch
        if parent_node is not None:
            correctness_advantage, H_target, ratio = node()

            reward_mask = torch.zeros_like(input_ids, dtype=torch.long)
            reward_mask[prev_len:] = 1
            
            batch.append({
                "correctness_advantage": correctness_advantage,
                "H_target": H_target,
                "reward_mask": reward_mask,
                "input_ids": input_ids,
                "h_target_ratio": ratio,
                "teacher_prefix": node.teacher_ids
            })

    def __freeze_old_policy_logits__(self, mini_batch):
        input_ids = [x["input_ids"] for x in mini_batch]
        reward_masks = [x["reward_mask"] for x in mini_batch]

        with torch.no_grad():
            old_logits, _, _ = get_shift_logits_and_labels(
                sequences=input_ids,
                reward_masks=reward_masks,
                hf_model=self.hf_model,
                pad_token_id=128001,
            )

        return old_logits.detach()

    def __process_mini_batch__(self, mini_batch, old_logits):
        input_ids = [x["input_ids"] for x in mini_batch]
        reward_masks = [x["reward_mask"] for x in mini_batch]

        new_logits, labels, masks = get_shift_logits_and_labels(
            sequences=input_ids,
            reward_masks=reward_masks,
            hf_model=self.hf_model,
            pad_token_id=128001,
        )

        device = new_logits.device
        dtype = new_logits.dtype

        old_logits = old_logits.to(device=device, dtype=dtype)
        labels = labels.to(device=device)
        masks = masks.to(device=device, dtype=dtype)

        with torch.no_grad():
            entropies = calculate_entropy_from_logits(
                logits=new_logits.detach()
            ).to(device=device, dtype=dtype)

            correctness_advantages = torch.tensor(
                [x["correctness_advantage"] for x in mini_batch],
                device=device,
                dtype=dtype,
            )

            H_targets = torch.tensor(
                [x["H_target"] for x in mini_batch],
                device=device,
                dtype=dtype,
            )

            advantages = (
                correctness_advantages.unsqueeze(1)
                - self.alpha * (entropies - H_targets.unsqueeze(1)) ** 2
            ) * masks

        ratio, new_log_probs, old_log_probs = get_ratio(
            new_logits=new_logits,
            old_logits=old_logits,
            labels=labels,
            masks=masks,
        )

        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.eps_clip,
            1.0 + self.eps_clip,
        )

        loss_per_token = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages,
        )

        return (loss_per_token * masks).sum()

    def process_batch(batch, mini_batch_size, epoch=3):
        batch = sorted(batch, key=lambda x: len(x["input_ids"]))

        n_total_tokens = sum(
            int(item["reward_mask"].sum().item())
            for item in batch
        )
        
        mini_batches = []
        for start in range(0, len(batch), mini_batch_size):
            end_idx = min(start + mini_batch_size, len(batch))
            
            mini_batches.append(batch[start:end_idx])
            
        return n_total_tokens, mini_batches
            
        
        """
        if n_total_tokens == 0:
            self.batch.clear()
            return

        was_training = self.hf_model.training

        self.hf_model.eval()

        old_logits_array = []
        for start in range(0, len(batch), mini_batch_size):
            end_idx = min(start + mini_batch_size, len(batch))
            items = batch[start:end_idx]

            old_logits_array.append(
                self.__freeze_old_policy_logits__(items)
            )

        if was_training:
            self.hf_model.train()

        for epoch_idx in range(epoch):
            self.optimizer.zero_grad()
            total_loss = 0.0

            for i, start in enumerate(range(0, len(batch), mini_batch_size)):
                end_idx = min(start + mini_batch_size, len(batch))
                items = batch[start:end_idx]

                loss = self.__process_mini_batch__(
                    items,
                    old_logits=old_logits_array[i],
                )

                loss = loss / n_total_tokens

                total_loss += loss.detach().item()

                loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.hf_model.parameters(),
                1.0,
            )

            self.optimizer.step()

            print(
                f"EPOCH {epoch_idx + 1}/{epoch}: "
                f"loss = {total_loss:.6f}"
            )

        self.optimizer.zero_grad()
        self.batch.clear()
    """