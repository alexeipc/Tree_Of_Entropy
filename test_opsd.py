import torch
from transformers import AutoModelForCausalLM

# Replace with your import
from opsd.opsd import OPSD

MODEL_PATH = "./sft/Llama-3.2-3B-Instruct_mot_sft_final"

device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model.eval()

# ==========================================================
# Student sequence
#
# [student prefix] [completion]
# ==========================================================

student_sequence = torch.tensor([
    128000,
    9125,
    374,
    264,
    1294,
    315,
    220,
    1234,
    5678,
    9012,
    3456,
], dtype=torch.long)

student_reward_mask = torch.tensor([
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
], dtype=torch.bool)

# ==========================================================
# Teacher prefix (longer)
# ==========================================================

teacher_prefix = torch.tensor([
    128000,
    128006,
    9125,
    374,
    264,
    1294,
    315,
    220,
    9999,
    8888,
    7777,
], dtype=torch.long)

entropies = OPSD.calculate_entropy_of_teacher(
    hf_model=model,
    sequences=[student_sequence],
    reward_masks=[student_reward_mask],
    pad_token_id=128009,          # or tokenizer.pad_token_id
    teacher_prefixes=[teacher_prefix],
)

print("=" * 80)
print("Student sequence")
print(student_sequence.tolist())

print("\nStudent reward mask")
print(student_reward_mask.int().tolist())

print("\nTeacher prefix")
print(teacher_prefix.tolist())

print("\nReturned entropy")
print(entropies[0])

print("\nLength:", len(entropies[0]))
print("=" * 80)