import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

MAX_LEN = 2048

class ResponseOnlyCollator:
    def __init__(self, tokenizer, response_template="<|start_header_id|>assistant<|end_header_id|>\n\n"):
        self.tokenizer = tokenizer
        self.response_ids = tokenizer(
            response_template,
            add_special_tokens=False,
        ).input_ids

        self.debug_printed = 0

    def __call__(self, examples):
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        for i in range(labels.size(0)):
            ids = batch["input_ids"][i].tolist()

            start = -1
            for j in range(len(ids) - len(self.response_ids) + 1):
                if ids[j:j + len(self.response_ids)] == self.response_ids:
                    start = j + len(self.response_ids)
                    break

            if start == -1:
                print("\n===== FAILED ASSISTANT MARKER =====")
                print(self.tokenizer.decode(ids, skip_special_tokens=False))
                print("===================================\n")
                labels[i, :] = -100
                continue

            if self.debug_printed < 3:
                print("\n========== COLLATOR DEBUG ==========")
                print("loss starts after token index:", start)
                print(self.tokenizer.decode(ids[start:start + 500], skip_special_tokens=False))
                print("====================================\n")
                self.debug_printed += 1

            labels[i, :start] = -100
            labels[i][batch["attention_mask"][i] == 0] = -100

        batch["labels"] = labels
        return batch


class SFT:
    def __init__(self, model_name):
        self.model_name = model_name
        self.output_dir = model_name.split("/models/")[-1] + "_mot_sft_output"
        self.final_dir = model_name.split("/models/")[-1] + "_mot_sft_final"

        self.dataset = load_dataset(
            "open-r1/Mixture-of-Thoughts",
            "all",              # or "math", "code", "science"
            split="train[:10000]",
        )

        self.load_model_and_tokenizer()
        self.load_data()

    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ message['content'] }}<|eot_id|>
{% elif message['role'] == 'user' %}
<|start_header_id|>user<|end_header_id|>

{{ message['content'] }}<|eot_id|>
{% elif message['role'] == 'assistant' %}
<|start_header_id|>assistant<|end_header_id|>

{{ message['content'] }}<|eot_id|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}"""

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.model.config.use_cache = False

    def format_example(self, example):
        # Dataset is already messages format.
        return {
            "text": self.tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    

    def load_data(self):
        print("Columns:", self.dataset.column_names)
        print("Raw example:", self.dataset[0])

        self.dataset = self.dataset.map(
            self.format_example,
            remove_columns=self.dataset.column_names,
            num_proc=8,
        )

        def length_filter(example):
            ids = self.tokenizer(
                example["text"],
                add_special_tokens=False,
            ).input_ids
            return len(ids) <= MAX_LEN

        self.dataset = self.dataset.filter(
            length_filter,
            num_proc=8,
        )

        print("Dataset after length filter:", len(self.dataset))
        print("\nFormatted example:\n")
        print(self.dataset[0]["text"][:3000])

    def train(self):
        args = SFTConfig(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=2e-5,
            num_train_epochs=1,
            max_length=MAX_LEN,
            bf16=True,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            weight_decay=0.01,
            warmup_ratio=0.03,
            dataset_text_field="text",
            packing=False,
        )

        collator = ResponseOnlyCollator(
            tokenizer=self.tokenizer,
            response_template="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

        trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,
            data_collator=collator,
        )

        trainer.train()

        trainer.save_model(self.final_dir)
        self.tokenizer.save_pretrained(self.final_dir)


if __name__ == "__main__":
    sft = SFT(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
    )
    sft.train()