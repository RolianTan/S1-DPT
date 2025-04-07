from datasets import load_dataset
import re
import random
import json
from s1_stage1 import S1Stage1
from s1_stage2 import S1Stage2
from dpt_cti import cti_generation
from prm800k.prm800k.grading.grader import grade_answer

# PART: Settings
N = 1
NUM_CTI_PER_SAMPLE = 10

# PART: Load MATH500
ds = load_dataset("HuggingFaceH4/MATH-500")
math_data = ds["test"]
selected_samples = random.sample(list(math_data), N)

# # PART: S1 Module Init
# s1_stage1 = S1Stage1()
# s1_stage2 = S1Stage2()

# Part: Prompt Template for CTI generation
prompt_template = """A student is trying to solve a math question, he provide his thinking process, please provide an instruction that could help him better recheck and continue his thoughts. The instruction should be short and precise, less than 500 tokens. The instructin should describe how to check any type of math problems, do not mention any specific formula or principles related to this question.

The math question is: [INPUT]
The student thinking process: [HINT]
The correct answer is: [OUTPUT]

The instruction should be short and precise, less than 1000 tokens. The instruction should describe how to check any type of math problems generally, do not mention any specific formula or principles related to only this question.
Your instruction is: [APE]
"""

def extract_math_answer(text):
    marker = "<|im_start|>answer"
    if marker in text:
        return text.split(marker)[-1].strip()
    return text.strip()


# Part: CTI Evaluation
final_cti_results = []

for idx, sample in enumerate(selected_samples):
    print(f"Processing Sample {idx+1}/{N}")
    question, solution, answer = sample["problem"], sample["solution"], sample["answer"]

    # S1 Stage 1: Extract reasoning trace
    print("------------Stage 1--------------")
    s1_stage1 = S1Stage1()
    thought = s1_stage1.extract_thought(question)
    del s1_stage1
    # print("-------------------------------------------------")
    # print(thought)

    # Generate CTIs
    dataset = ([question], [answer])
    ctis = cti_generation(
        dataset=dataset,
        prompt_gen_template=prompt_template,
        prompt_gen_model="deepseek-chat",
        num_prompts=NUM_CTI_PER_SAMPLE,
    )
    # print("--------CTI----------")
    # print(ctis)

    # S2 Stage 2: Evaluate each CTI
    print("------------Stage 2--------------")
    s1_stage2 = S1Stage2()
    for cti in ctis:
        correct = 0
        for test_sample in math_data:
        # for test_sample in random.sample(list(math_data), 5): # debug
            test_problem, test_answer = test_sample["problem"], test_sample["answer"]
            # For each test sample, run CTI with full s1 process
            output = s1_stage2.evaluate_cti(test_problem, cti)
            # get answer
            answer = extract_math_answer(output)
            # is_correct = grade_answer(given_answer, ground_truth)
            # print(f"output:{output}")
            print("---------answer---------")
            print(f"answer:{answer}")
            print(f"true:{test_answer}")
            if grade_answer(answer, test_answer):
                print("correct")
                correct += 1
            # exit()
        acc = correct / len(math_data)
        final_cti_results.append({
            "cti": cti,
            "accuracy": acc
        })
    del s1_stage2

# Save final results
with open("cti_results_dpt.json", "w") as f:
    json.dump(final_cti_results, f, indent=2)

print("Done. Final CTI results saved to cti_results_dpt.json.")
