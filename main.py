from tree import Tree
from transformers import AutoTokenizer, AutoModelForCausalLM
from util.debug import debug
from util.jsd_sparse import future_disagreement
import torch
from vllm import LLM


if __name__ == "__main__":
    MODEL_PATH = "./sft/meta-llama/Llama-3.2-1B_mot_sft_final"
    llm = LLM(MODEL_PATH, dtype="bfloat16")
    
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,   # or float16
    ).cuda("cuda:1")

    hf_model.eval()
    
    hf_model = None
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    tree = Tree(llm, None, tokenizer.eos_token_id)
    
    prompts = [
        [
            {
                "role": "user",
                "content": (
                    "Solve me this problem: "
                    "Among the natural numbers from 1 to 2020, there are 404 numbers "
                    "that are multiples of 5. If these 404 numbers are multiplied "
                    "together, how many consecutive zeros are there at the end of "
                    "the product?"
                ),
            }
        ],
        [
            {
                "role": "user",
                "content": (
                    "Solve me this problem: "
                    "Evaluate the limit: \[ \lim_{x \to \infty} \sqrt{x} \left( \sqrt[3]{x+1} - \sqrt[3]{x-1} \right) \]"
                ),
            }
        ]
    ]
    
    gts = [
        "503","0"
    ]
    
    applied_template_prompts = tokenizer.apply_chat_template(
        prompts,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Get input_ids
    encoded = tokenizer(
        applied_template_prompts,
        padding=False,
        truncation=False,
        return_tensors=None,   # return an array of 1d tensor instead of a 2d tensor
    )
    input_ids = [
        torch.tensor(ids, dtype=torch.long)
        for ids in encoded["input_ids"]
    ]
    
    group_ids = [i for i in range(len(prompts))]
    
    # At first every depth is 1
    depths = [1] * len(prompts)
     
    tree.forward({
        "text": applied_template_prompts,
        "input_ids": input_ids,
        "group_ids": group_ids,
        "thresholds": [2] * len(prompts) # at first entropy threshold is 2
    }, depths, gts, is_init=True)