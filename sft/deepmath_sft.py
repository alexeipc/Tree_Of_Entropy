import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer


MAX_LEN = 2048

DATASET_PATH = "./data/deepmath_sft_4_6_qwen2_5_7b"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

OUTPUT_DIR = "./sft/Qwen2.5-7B-Instruct_deepmath_sft_output"
FINAL_DIR = "./sft/Qwen2.5-7B-Instruct_deepmath_sft_final"

class ResponseOnlyCollator:
    def __init__(
        self,
        tokenizer,
        response_template=None,
    ):
        self.tokenizer = tokenizer

        if response_template is None:
            response_template = self._get_response_template(tokenizer)

        self.response_ids = tokenizer(
            response_template,
            add_special_tokens=False,
        ).input_ids

        self.debug_printed = 0

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

            for j in range(
                len(ids) - len(self.response_ids) + 1
            ):
                if (
                    ids[j:j + len(self.response_ids)]
                    == self.response_ids
                ):
                    start = j + len(self.response_ids)
                    break

            if start == -1:
                print(
                    "\n===== FAILED ASSISTANT MARKER ====="
                )
                print(
                    self.tokenizer.decode(
                        ids,
                        skip_special_tokens=False,
                    )
                )
                print(
                    "===================================\n"
                )

                labels[i, :] = -100
                continue

            if self.debug_printed < 3:
                print(
                    "\n========== COLLATOR DEBUG =========="
                )
                print(
                    "Loss starts after token index:",
                    start,
                )
                print(
                    self.tokenizer.decode(
                        ids[start:start + 500],
                        skip_special_tokens=False,
                    )
                )
                print(
                    "====================================\n"
                )

                self.debug_printed += 1

            labels[i, :start] = -100

            labels[i][
                batch["attention_mask"][i] == 0
            ] = -100

        batch["labels"] = labels

        return batch


class SFT:
    def __init__(self, model_name):
        self.model_name = model_name

        self.output_dir = OUTPUT_DIR
        self.final_dir = FINAL_DIR

        self.dataset = load_from_disk(
            DATASET_PATH
        )

        self.load_model_and_tokenizer()
        self.load_data()

    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )

        self.model.config.use_cache = False
        self.model.config.pad_token_id = (
            self.tokenizer.pad_token_id
        )

    def load_data(self):
        print(
            "Columns:",
            self.dataset.column_names,
        )

        print(
            "Dataset size:",
            len(self.dataset),
        )

        print(
            "\nFormatted example:\n"
        )

        print(
            self.dataset[0]["text"][:3000]
        )

        # Keep only the text column. This is the important fix.
        columns_to_remove = [
            column
            for column in self.dataset.column_names
            if column != "text"
        ]

        if columns_to_remove:
            self.dataset = self.dataset.remove_columns(
                columns_to_remove
            )

        print(
            "Final columns:",
            self.dataset.column_names,
        )

    def train(self):
        args = SFTConfig(
            output_dir=self.output_dir,

            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,

            learning_rate=2e-5,
            num_train_epochs=1,

            max_length=MAX_LEN,

            bf16=True,
            tf32=True,

            logging_steps=10,

            save_steps=100,
            save_total_limit=2,

            weight_decay=0.01,
            warmup_ratio=0.03,

            dataset_text_field="text",
            packing=False,

            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            },

            fsdp="full_shard auto_wrap",
            fsdp_config={
                "transformer_layer_cls_to_wrap": (
                    self.model._no_split_modules[0]
                ),
                "use_orig_params": True,
                "sync_module_states": True,
            },

            dataloader_num_workers=0,

            report_to="wandb",
            run_name=(
                "llama-3.1-8b-deepmath-4-6-sft"
            ),
        )

        collator = ResponseOnlyCollator(
            tokenizer=self.tokenizer,
        )

        trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,
            data_collator=collator,
        )

        trainer.train()

        trainer.save_model(
            self.final_dir
        )

        if trainer.is_world_process_zero():
            self.tokenizer.save_pretrained(
                self.final_dir
            )


if __name__ == "__main__":
    sft = SFT(
        model_name=MODEL_NAME,
    )

    sft.train()
