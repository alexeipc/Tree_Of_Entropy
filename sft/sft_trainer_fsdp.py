import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer


MAX_LEN = 10000


class ResponseOnlyCollator:
    def __init__(self, tokenizer, response_template=None):
        self.tokenizer = tokenizer
        if response_template is None:
            response_template = self._get_response_template(tokenizer)
        self.response_ids = tokenizer(
            response_template,
            add_special_tokens=False,
        ).input_ids

    @staticmethod
    def _get_response_template(tokenizer):
        messages = [{"role": "user", "content": "template probe"}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        conversation = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if not prompt.startswith(conversation) or prompt == conversation:
            raise ValueError("Could not derive the assistant marker from the chat template.")
        return prompt[len(conversation):]

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
                labels[i, :] = -100
                continue

            labels[i, :start] = -100
            labels[i][batch["attention_mask"][i] == 0] = -100

        batch["labels"] = labels
        return batch


class SFT:
    def __init__(self, model_name):
        self.model_name = model_name
        self.output_dir = model_name.split("/")[-1] + "_mot_sft_output"
        self.final_dir = model_name.split("/")[-1] + "_mot_sft_final"

        self.dataset = load_dataset(
            "open-r1/Mixture-of-Thoughts",
            "all",
            split="train[:10000]",
        )

        self.load_model_and_tokenizer()
        self.load_arg()
        self.load_data()

    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.model.config.use_cache = False

    def format_example(self, example):
        return {
            "text": self.tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    def length_filter(self, example):
        ids = self.tokenizer(
            example["text"],
            add_special_tokens=False,
        ).input_ids

        return len(ids) <= MAX_LEN

    def load_data(self):
        print("Columns:", self.dataset.column_names)
        print("Raw example:", self.dataset[0])

        self.dataset = self.dataset.map(
            self.format_example,
            remove_columns=self.dataset.column_names,
            num_proc=8,
        )

        before = len(self.dataset)

        self.dataset = self.dataset.filter(
            self.length_filter,
            num_proc=8,
        )

        after = len(self.dataset)

        print(f"Kept {after}/{before} examples with length <= {MAX_LEN}")
        print("\nFormatted example:\n")
        print(self.dataset[0]["text"][:3000])

    def load_arg(self):
        self.args = SFTConfig(
            output_dir=self.output_dir,

            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,

            learning_rate=2e-5,
            num_train_epochs=1,
            max_length=MAX_LEN,

            bf16=True,
            
            gradient_checkpointing=False,

            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            weight_decay=0.01,
            warmup_steps=100,

            dataset_text_field="text",
            packing=False,
            dataloader_num_workers=0,

            fsdp="full_shard auto_wrap",
            fsdp_config={
                "transformer_layer_cls_to_wrap": self.model._no_split_modules[0],
                "use_orig_params": True,
                "activation_checkpointing": True,
            },
        )
    def train(self):
        collator = ResponseOnlyCollator(
            tokenizer=self.tokenizer,
        )

        trainer = SFTTrainer(
            model=self.model,
            args=self.args,
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
