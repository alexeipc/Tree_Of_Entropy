from opsd.prompts import create_student_prompt, create_teacher_prompt
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from util.reward_func import reward


class OSPD:
    def generate_teacher_answers(vllm: LLM, 
                                 reference_answers: List[str], 
                                 problems: List[str], 
                                 reasonings: List[str],
                                 ground_truths: List[str],
                                 sampling_params: SamplingParams):
        prompts = [
            f"{create_teacher_prompt(problem, reference_answer)}\n{reasonings}"
            for problem, reference_answer, reasoning in zip(reference_answers, problems, reasonings)
        ]
        
        outputs = vllm.generate(
            prompts=prompts,
            sampling_params = sampling_params
        )
        
        accuracies = []
        
        for i, completions in enumerate(outputs):
            accuracy = 0
            for completion in completions.outputs:
                full_answer = prompts[i] + completion 
                
                # Calculate rewards
                reward_score = reward(full_answer, ground_truths[i])
                
                accuracy += (reward_score >= 1)
                
            accuracies.append(accuracy / len(completions.outputs))
            
        return accuracies
        
        
        
        
        
        
    
        
        
    
        
