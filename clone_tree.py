from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
from util.entropy import calculate_entropy
from util.debug import debug

class Tree:
    def __init__(self, llm: LLM, tokenizer: AutoTokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
        self.n_branch = 3
        self.max_depth = 3
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
            logprobs=19,
        )
    
    def handle_single_completion(self, completion):
        entropies = []
        for lp_dict in completion.logprobs:
            logprobs = torch.tensor(
                [x.logprob for x in lp_dict.values()],
                dtype=torch.float32,
            )
            entropies.append(calculate_entropy(logprobs))
        
        entropies = torch.stack(entropies)
        token_ids = completion.token_ids
        
        debug(entropies)
        
        # Find the max_entropy
        arg_max_pos = torch.argmax(entropies).item()
        max_entropy = entropies.max()
        sigma = entropies.std(unbiased=False)
        threshold = max_entropy - sigma * 0.5
        candidates = torch.where(entropies >= threshold)[0]
        pos = candidates[0].item()
        
        debug(entropies[pos: arg_max_pos + 1])
        
        logs = list(completion.logprobs[pos].values())
        debug(logs)
        text_before = self.tokenizer.decode(token_ids[:pos])
        debug(text_before)
        
        # Choose the next few possible branch (ignore the rank 1)
        next_prompts = []
        for i in range(1, self.n_branch):
            new_text = text_before + logs[i].decoded_token
            next_prompts.append(new_text)
            
        return next_prompts
    
    def forward(self, batch_messages, depth = 1):
        if depth > self.max_depth:
            return
        if depth == 1:
            prompts = self.tokenizer.apply_chat_template(
                batch_messages,
                tokenize = False,
                add_generation_prompt = True
            )
            group_ids = [i for i in range(len(prompts))]
            debug(prompts)
        else:
            prompts = batch_messages["text"]
            group_ids = batch_messages["group_ids"]
        
        outputs = self.llm.generate(
            prompts,
            self.sampling_params
        )
        
        next_prompts = []
        next_group_ids = []
        main_branches = []
        for i, output in enumerate(outputs):
            print(f"Prompt {i}")
            original_prompt = prompts[i]

            for completion in output.outputs:
                main_branches.append(original_prompt + completion.text)
                # If it's the max depth then no more branching
                if depth == self.max_depth:
                    continue
                next_prompt = self.handle_single_completion(completion)
                next_prompt = [
                    original_prompt + p
                    for p in next_prompt
                ]
                next_prompts.extend(next_prompt)
                next_group_ids.extend([group_ids[i]]*len(next_prompt))
        
        if depth == self.max_depth:
            return {
                "text": main_branches,
                "group_ids": group_ids
            }
        
        debug(next_prompts)
        debug(next_group_ids)
        
        batch_messages = {
            "text": next_prompts,
            "group_ids": next_group_ids
        }
        branches = self.forward(batch_messages=batch_messages, depth=depth + 1)
        
        branches["text"].extend(main_branches)
        branches["group_ids"].extend(group_ids)
        return branches
    
'''
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
from util.entropy import calculate_entropy
from util.debug import debug

class EntropyStopper:
    def __init__(self, eos_token_id: int, threshold: float):
        self.eos_token_id = eos_token_id
        self.threshold = threshold
        self.entropies = {}
        self.used_ids = set()

    def __call__(self, input_ids, logits):
        # Get the array of entropies        
        if len(input_ids) == 0:
            self.entropies[input_ids] = []
        else:
            if input_ids not in self.entropies:
                prev_input_ids = input_ids[:-1]
                if  prev_input_ids not in self.used_ids:
                    self.used_ids.add(prev_input_ids)
                    self.entropies[input_ids] = self.entropies[prev_input_ids]
                else:
                    self.entropies[input_ids] = self.entropies[prev_input_ids][:len(input_ids) - 1].copy()
            else:
                self.entropies[input_ids] = self.entropies[input_ids][:len(input_ids) - 1].copy()
            
        entropies_hist = self.entropies[input_ids]
        
        # logits: [vocab]
        logprobs = torch.log_softmax(logits, dim=-1)
        debug(input_ids)
        probs = torch.exp(logprobs)

        entropy = -(probs * logprobs).sum()
        entropies_hist.append(entropy)
    
        if entropy >= self.threshold:
            logits[:] = -float("inf")
            logits[self.eos_token_id] = 0.0  # 100% EOS after softmax

        return logits

class Tree:
    def __init__(self, llm: LLM, tokenizer: AutoTokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
        self.n_branch = 3
        self.max_depth = 3
        eos_id = tokenizer.eos_token_id
        
        self.entropy_stopper = EntropyStopper(
                    eos_token_id=eos_id,
                    threshold=2.0,
                )

        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
            logprobs=19,
            logits_processors=[
                self.entropy_stopper
            ],
        )

    def handle_single_completion(self, completion):
        token_ids = completion.token_ids
    
        entropies = self.entropy_stopper.entropies[tuple(token_ids)][:len(token_ids)]
        
        debug(token_ids[:-1])
        debug([x.item() for x in entropies])
    
    def forward(self, batch_messages, depth = 1):
        if depth > self.max_depth:
            return
        if depth == 1:
            prompts = self.tokenizer.apply_chat_template(
                batch_messages,
                tokenize = False,
                add_generation_prompt = True
            )
            group_ids = [i for i in range(len(prompts))]
            debug(prompts)
        else:
            prompts = batch_messages["text"]
            group_ids = batch_messages["group_ids"]
        
        outputs = self.llm.generate(
            prompts,
            self.sampling_params
        )
        
        next_prompts = []
        next_group_ids = []
        for i, output in enumerate(outputs):
            print(f"Prompt {i}")
            original_prompt = prompts[i]

            for completion in output.outputs:
                debug(completion.text)
                self.handle_single_completion(completion=completion)
                
'''

"""
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from util.jsd_sparse import future_disagreement
from util.debug import debug
from util.reward_func import reward
from tree_reward import TreeRewardManager


class EntropyStopper:
    def __init__(self, eos_token_id: int, threshold: float):
        self.eos_token_id = eos_token_id
        self.threshold = threshold

    def __call__(self, input_ids, logits):
        # logits: [vocab]
        logprobs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(logprobs)

        entropy = -(probs * logprobs).sum()
    
        if entropy >= self.threshold:
            debug("xo"*40)
            debug(input_ids)
            debug(entropy)
            logits[:] = -float("inf")
            logits[self.eos_token_id] = 0.0  # 100% EOS after softmax

        return logits

class Tree:
    def __init__(self, llm: LLM, hf_model: AutoModelForCausalLM, tokenizer: AutoTokenizer, branching_threshold:float = 0.3):
        self.llm = llm
        self.tokenizer = tokenizer
        self.hf_model = hf_model
        self.n_branch = 3
        self.max_depth = 3
        self.branching_threshold = branching_threshold
        
        self.prompt_ids = []
        
        self.optimizer = torch.optim.AdamW(
            self.hf_model.parameters(),
            lr=1e-6,
            weight_decay=0.0,
        )
        
        self.manager = TreeRewardManager(
            hf_model=hf_model,
            optimizer = self.optimizer
        )
        

    def handle_single_completion(self, completion, prompt, input_ids, do_branching = True):
        generated_token_ids = completion.token_ids[:-1]        
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
            temperature=0.8,
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
    
    def forward(self, batch_messages, depth = 1, gts = None):
        if depth == 1:
            prompts = self.tokenizer.apply_chat_template(
                batch_messages,
                tokenize = False,
                add_generation_prompt = True
            )
            encoded = self.tokenizer(
                prompts,
                padding=False,
                truncation=False,
                return_tensors=None,   # important
            )

            input_ids = [
                torch.tensor(ids, dtype=torch.long)
                for ids in encoded["input_ids"]
            ]
            
            for ids in input_ids:
                self.manager.add_node(ids)
                self.prompt_ids.append(ids)
                        
            group_ids = [i for i in range(len(prompts))]
            
            self.gts = gts
        else:
            prompts = batch_messages["text"]
            input_ids = batch_messages["input_ids"]
            group_ids = batch_messages["group_ids"]
        
        logits_processors = []

        if depth < self.max_depth:
            logits_processors.append(
                EntropyStopper(
                    eos_token_id=self.tokenizer.eos_token_id,
                    threshold=2.0,
                )
            )

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
            logprobs=19,
            logits_processors=logits_processors,
        )
        
        outputs = self.llm.generate( prompts, sampling_params )
        
        next_prompts = []
        next_group_ids = []
        next_input_ids = []
        
        curr_node_ids = []
        relations = []
                
        for i, output in enumerate(outputs):
            print(f"Prompt {i}")

            for completion in output.outputs:
                if depth < self.max_depth:
                    tmp = self.handle_single_completion(completion, prompts[i], input_ids[i])
                else:
                    '''debug("*"*80)
                    debug(completion.text)'''
                    # If it's the last one then get the reward
                    reward_score = reward(prompts[i] + completion.text, self.gts[group_ids[i]])
                    tmp = self.handle_single_completion(completion, prompts[i], input_ids[i], do_branching=False)
                
                # TODO: Stop if reahces <eos>
                
                node_ids = tmp[0]
                curr_node_ids.append(node_ids)
                node = self.manager.add_node(node_ids)
                
                if depth == 1:
                    debug("=*"*40)
                    debug(self.prompt_ids[i])
                    debug(node_ids)
                    
                    self.manager.add_child(
                        parent_ids=self.prompt_ids[i],
                        child_ids=node_ids
                    )
                
                
                if depth == self.max_depth:
                    node.correct_answer = reward_score
                    node.wrong_answer = 1 - reward_score
                    
                    debug(node.__dict__)
    
                for prompt, ids in tmp[1:]:
                    relations.append(
                        (len(curr_node_ids) - 1, len(next_prompts))
                    )
                    next_prompts.append(prompt)
                    next_group_ids.append(group_ids[i])
                    next_input_ids.append(ids)
        
        '''          
        debug(next_prompts)
        debug(next_group_ids)
        debug(next_input_ids)
        '''
        
        if depth < self.max_depth:
            next_node_ids = self.forward(
                {
                    "text": next_prompts,
                    "group_ids": next_group_ids,
                    "input_ids": next_input_ids
                },
                depth=depth + 1
            )
        
            for parent, child in relations:
                self.manager.add_child(
                    parent_ids=curr_node_ids[parent],
                    child_ids=next_node_ids[child]
                )
                
                debug("o"*80)
                debug(curr_node_ids[parent])
                debug(next_node_ids[child])
    
        
        return curr_node_ids    
    
    def backward(self):
        for ids in self.prompt_ids:
            self.manager.traverse(ids)
            
        self.manager.process_batch(3)
"""