from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from util.jsd_sparse import future_disagreement
from util.debug import debug
from util.reward_func import reward
from tree_reward import TreeRewardManager
from util.entropy_processor import EntropyStopper
from util.opsd import change_prompts
from opsd.opsd import OPSD
from multiprocessing import shared_memory
import numpy as np

class Tree:
    def __init__(self, llm: LLM, eos_id, tree_reward_manager, branching_threshold:float = 0.3, batch_size = 32):
        self.llm = llm
        self.eos_id = eos_id
        self.n_branch = 2
        self.max_depth = 4
        self.branching_threshold = branching_threshold
            
        
        self.manager:TreeRewardManager = tree_reward_manager
        
        self.batch_size = batch_size
        
        self.max_length = 678
        
    def normalize_group(groups):
        debug("*#"*80)
        
        for group in groups.values():
            rewards = torch.tensor([x["reward"] for x in group], dtype=torch.float32)
            advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
            
            debug(advantages)

            for item, adv in zip(group, advantages):
                item["node"].advantage = adv.item()
                
        debug("#"*80)
        

    def handle_single_completion(self, completion, prompt, input_ids, do_branching = True):
        """
            return: [<the token ids of the prompt + completions>, (<next prompts after branching>, <next token ids after branching>)]
        """
        if do_branching:
            generated_token_ids = completion.token_ids[:-1]    
        else:
            generated_token_ids = completion.token_ids
            
        generated_token_ids = torch.tensor(generated_token_ids)
        generated_text = completion.text
        
        next_prompt = prompt + generated_text
        # Conver them to ids to long
        input_ids = input_ids.to(torch.long)
        generated_token_ids = generated_token_ids.to(torch.long)
        next_ids = torch.cat([input_ids, generated_token_ids])
        
        if not do_branching:
            return [next_ids]
                
        sampling_params = SamplingParams(
            n=self.n_branch,
            max_tokens=10,
            temperature=0.6,
            top_p=0.95,
            logprobs=10,   # K
        )
        
        outputs = self.llm.generate(
            prompts=[next_prompt],
            sampling_params=sampling_params,
        )
        
        completions = outputs[0].outputs
        '''
        for completion in completions:
            debug(completion.text)
        '''
        
        branching_score = future_disagreement(completions)
        
        if branching_score >= self.branching_threshold:
            next_prompts = [next_ids]

            for completion in completions:
                completion_ids = torch.tensor(
                    completion.token_ids,
                    device=generated_token_ids.device,
                    dtype=generated_token_ids.dtype,
                )

                new_token_ids = torch.cat(
                    [next_ids, completion_ids],
                    dim=0,
                )

                next_prompts.append(
                    (
                        next_prompt + completion.text,
                        new_token_ids,
                    )
                )

            return next_prompts
        else:
            # Very similar so should not branch
            return [next_ids, (next_prompt, next_ids)]
    
    def forward(self, batch_messages, depths, reference_answers, gts, groups = {}, is_init = False):
        debug(depths)
        
        prompts = batch_messages["text"]
        input_ids = batch_messages["input_ids"]
        group_ids = batch_messages["group_ids"]
        thresholds = batch_messages["thresholds"]
        
        num_requests = len(prompts)

        # One float64 per request, stored in CPU shared memory.
        entropy_shm = shared_memory.SharedMemory(
            create=True,
            size=num_requests * np.dtype(np.float64).itemsize,
        )

        stop_entropies_shared = np.ndarray(
            (num_requests,),
            dtype=np.float64,
            buffer=entropy_shm.buf,
        )

        # NaN means the request did not cross its threshold.
        stop_entropies_shared[:] = np.nan

        sampling_params = []

        for i, threshold in enumerate(thresholds):
            effective_threshold = (
                threshold
                if depths[i] < self.max_depth
                else float("inf")
            )

            sampling_params.append(
                SamplingParams(
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=max(
                        2,
                        self.max_length - len(input_ids[i]),
                    ),
                    logprobs=19,
                    extra_args={
                        "entropy_threshold": float(effective_threshold),
                        "entropy_eos_token_id": int(self.eos_id),

                        # Shared CPU-memory location for this request.
                        "entropy_shm_name": entropy_shm.name,
                        "entropy_slot": i,
                        "entropy_num_slots": num_requests,
                    },
                )
            )

        try:
            outputs = self.llm.generate(
                prompts,
                sampling_params,
            )

            # Copy values before closing shared memory.
            stop_entropies = [
                None if np.isnan(value) else float(value)
                for value in stop_entropies_shared.copy()
            ]
        finally:
            entropy_shm.close()
            entropy_shm.unlink()
        
        # outputs = self.llm.generate( prompts, sampling_params )
        
        # Collect for next iteration
        next_prompts = []
        next_group_ids = []
        next_input_ids = []
        next_thresholds = []
        next_depths = []
        
        curr_node_ids = []
        relations = []
        
        # Collect teacher prompts
        teacher_reference_answers = []
        teacher_messages = []
        teacher_gts = []
        teacher_childs = []
        teacher_n_branches = 0
                
        for i, output in enumerate(outputs):
            # debug("+"*30,f"Prompt {i}","+"*30)
            completion = output.outputs[0]
            # debug(prompts[i] + completion.text)
            
            # Stop before reaching the threshold, meaning that it reaches the eos or tokens limit
            #completion = output.outputs[0]

            if stop_entropies[i] is None:
                debug("HE"*40)
                debug(completion.token_ids)
                reward_score = reward(prompts[i] + completion.text, gts[group_ids[i]])
                
                debug(reward_score)
                tmp = self.handle_single_completion(completion, prompts[i], input_ids[i], do_branching=False)
                
                # Save the nodes witht the manager
                node_ids = tmp[0]
                curr_node_ids.append(node_ids)
                node = self.manager.add_node(node_ids)
                
                # Save the reward score
                node.correct_answer = (reward_score >= 1)
                node.wrong_answer = 1 - node.correct_answer
                
                node.is_leaf = True
                if group_ids[i] not in groups:
                    groups[group_ids[i]] = []
                    
                groups[group_ids[i]].append(
                    {
                        "reward": reward_score,
                        "node": node
                    }
                )
                
                # If is_init then add the completion to the tree
                if is_init:
                    self.manager.add_child(
                        parent_ids = input_ids[i],
                        child_ids = node_ids
                    )
                
                continue                
            
            # Else start branching
            tmp = self.handle_single_completion(completion, prompts[i], input_ids[i])
            
            node_ids = tmp[0]
            
            # If is_init then add the completion to the tree
            if is_init:
                self.manager.add_child(
                    parent_ids = input_ids[i],
                    child_ids = node_ids
                )
            
            curr_node_ids.append(node_ids)
            node = self.manager.add_node(node_ids)
            
            # Save this for the teacher
            teacher_reference_answers.append(reference_answers[group_ids[i]])
            teacher_messages.append(prompts[i] + completion.text)
            teacher_gts.append(gts[group_ids[i]])
            # Len of curr_node_ids - 1 is the index of this node ids
            teacher_childs.append(len(curr_node_ids) - 1)

            # Only consider to be a branch if the number of branch is larger than 2
            next_depth = depths[i] + (len(tmp) > 2)
            next_threshold = stop_entropies[i]
            
            # Start branching from here
            for prompt, ids in tmp[1:]:
                # If ID contain eos then add it to curr_node_ids
                # It does need to branch anymore cuz it's already reached its end
                #debug("-"*80)
                #debug(ids)
                #debug(prompt)
                if ids[-1] == self.eos_id or len(ids) >= self.max_length:                    
                    # Get reward for this
                    reward_score = reward(prompt, gts[group_ids[i]])
                    
                    node = self.manager.add_node(ids)
                    
                    # Save the reward score
                    node.correct_answer = (reward_score >= 1)
                    node.wrong_answer = 1 - node.correct_answer
                    
                    node.is_leaf = True
                    if group_ids[i] not in groups:
                        groups[group_ids[i]] = []
                        
                    groups[group_ids[i]].append(
                        {
                            "reward": reward_score,
                            "node": node
                        }
                    )
                    
                    # Add this as a child of node_ids
                    self.manager.add_child(
                        parent_ids=node_ids,
                        child_ids=ids
                    )
                    
                    debug("NODE ADDED")
                    
                    continue
                
                relations.append(
                    (len(curr_node_ids) - 1, len(next_prompts))
                )
                
                next_prompts.append(prompt)
                next_group_ids.append(group_ids[i])
                next_input_ids.append(ids)
                next_depths.append(next_depth)
                next_thresholds.append(next_threshold)
        '''
        debug(next_prompts)
        debug(next_group_ids)
        debug(next_input_ids)
        debug(next_thresholds)
        debug(next_depths)
        '''
        
        next_node_ids = []
        
        # Divide branches into mini batches
        for start_idx in range(0, len(next_prompts), self.batch_size):
            end_idx = min(len(next_prompts), start_idx + self.batch_size)
            batch = {
                "text": next_prompts[start_idx:end_idx],
                "group_ids": next_group_ids[start_idx:end_idx],
                "input_ids": next_input_ids[start_idx:end_idx],
                "thresholds": next_thresholds[start_idx:end_idx]
            }
            
            next_node_ids.extend(
                self.forward(batch_messages=batch,
                             depths=next_depths[start_idx:end_idx],
                             gts=gts,
                             groups=groups,
                             reference_answers=reference_answers)
            )
            
        for parent, child in relations:
            self.manager.add_child(
                parent_ids=curr_node_ids[parent],
                child_ids=next_node_ids[child]
            )
            
            # Calculate the number of branches from the children's nodes
            child_node = self.manager.add_node(next_node_ids[child])
            child_node.count_branches()
            
            debug("CHILD NODE BRANCHES")
            debug(child_node.n_branches)
            teacher_n_branches = max(teacher_n_branches, child_node.n_branches)
            '''
            debug("o"*80)
            debug(curr_node_ids[parent])
            debug(next_node_ids[child])
            '''
            
        # Generate teacher's answers
        #TODO: Calculate teacher_n_branches
        #debug("TEACHER N BRANCHES")
        #debug(teacher_n_branches)
        
        teacher_n_branches = 4
        
        OPSD.handle_batch(
            vllm=self.llm,
            teacher_messages = teacher_messages,
            teacher_reference_answers=teacher_reference_answers,
            teacher_n_branches=teacher_n_branches,
            teacher_gts=teacher_gts,
            manager=self.manager, 
            input_ids=[curr_node_ids[i] for i in teacher_childs]
        )
        
        # If it is init then add the teachers's prompts
        if is_init:
            #TODO: Calculate # of branches
            OPSD.handle_batch(
                vllm=self.llm,
                # For init, teacher messages = prompts
                teacher_messages=prompts,
                teacher_reference_answers=teacher_reference_answers,
                teacher_n_branches=teacher_n_branches,
                # For init, gts are the same
                teacher_gts=gts,
                manager=self.manager,
                input_ids=input_ids
            )
            
        if is_init:
            Tree.normalize_group(groups=groups)
            
        return curr_node_ids