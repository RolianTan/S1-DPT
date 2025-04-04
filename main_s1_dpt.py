from datasets import load_dataset
import random
import json
from s1_stage1 import S1Stage1
from s1_stage2 import S1Stage2
from dpt_cti import cti_generation

# PART: Settings
N = 5
NUM_CTI_PER_SAMPLE = 10

# PART: Load MATH500
ds = load_dataset("HuggingFaceH4/MATH-500")
math_data = ds["test"]
selected_samples = random.sample(list(math_data), N)

# PART: S1 Module Init
s1_stage1 = S1Stage1()
s1_stage2 = S1Stage2()

# Part: Prompt Template for CTI generation
prompt_template = """A student is trying to solve a math question, he provide his thinking process, please provide an instruction that could help him better recheck and continue his thoughts, the instruction should be general not specific to this problem.

The math question is: [INPUT]
The student thinking process: [HINT]
The correct answer is: [OUTPUT]

Your instruction is: [APE]
"""

# Part: CTI Evaluation
final_cti_results = []

for idx, sample in enumerate(selected_samples):
    print(f"Processing Sample {idx+1}/{N}")
    question, solution, answer = sample["problem"], sample["solution"], sample["answer"]

    # S1 Stage 1: Extract reasoning trace
    thought = s1_stage1.extract_thought(question)

    # Generate CTIs
    dataset = ([question], [answer])
    ctis = cti_generation(
        dataset=dataset,
        prompt_gen_template=prompt_template,
        prompt_gen_model="deepseek-chat",
        soft_prompt=thought,
        num_prompts=NUM_CTI_PER_SAMPLE,
    )

    # S2 Stage 2: Evaluate each CTI
    for cti in ctis:
        correct = 0
        for test_sample in math_data:
            test_problem, test_answer = test_sample["problem"], test_sample["answer"]
            # For each test sample, run CTI with full s1 process
            output = s1_stage2.evaluate_cti(test_problem, cti)
            if test_answer.strip() in output:
                correct += 1
        acc = correct / len(math_data)
        final_cti_results.append({
            "cti": cti,
            "accuracy": acc
        })

# Save final results
with open("cti_results_dpt.json", "w") as f:
    json.dump(final_cti_results, f, indent=2)

print("Done. Final CTI results saved to cti_results_dpt.json.")
