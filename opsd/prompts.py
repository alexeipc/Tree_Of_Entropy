def create_student_prompt(question: str) -> str:
    return (
        f"{question}"
        "Provide your reasoning and final answer"
    )

def create_teacher_prompt(question: str, reference_solution: str) -> str:
    return (
        f"{question}"
        "Reference Solution:"
        f"{reference_solution}"
        "Use the verified solution to assess and improve the reasoning process."
        "Continue by producing a correct solution to the original problem."
    )