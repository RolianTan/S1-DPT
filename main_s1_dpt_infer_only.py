from datasets import load_dataset
import re
import random
import json
from s1_stage1 import S1Stage1
from s1_stage2 import S1Stage2
from dpt_cti import cti_generation
from prm800k.prm800k.grading.grader import grade_answer
import csv
import tqdm
from utils import extract_boxed_content, extract_text_content, extract_math_answer

def main():
    # PART: Load MATH500
    ds = load_dataset("HuggingFaceH4/MATH-500")
    math_data = list(ds["test"])
    # test_samples = math_data[200:202]

    # Part: CTI Evaluation
    final_cti_results = []
    all_predictions = []

    # S2 Stage 2: Evaluate each CTI
    print("------------Stage 2--------------")
    s1_stage2 = S1Stage2()
    cti_step = 0
    cti = "When solving math problems, always verify your solution by retracing each step systematically. First, ensure you've applied all formulas and transformations correctly by cross-referencing with reliable sources. Next, check for cancellation or simplification errors by writing out intermediate steps clearly. Finally, test your answer by plugging it back into the original problem or using alternative methods to confirm consistency. This multi-layered verification process helps catch mistakes and reinforces understanding."
    # print('here')
    correct_s1 = 0
    correct_s1dpt = 0
    step = 0
    # for test_sample in tqdm.tqdm(test_samples, desc=f"cti:{cti_step} | eval_step:{step}"):
    for test_sample in tqdm.tqdm(math_data, desc=f"cti:{cti_step} | eval_step:{step}"):
    # for test_sample in random.sample(list(math_data), 5): # debug
        step += 1
        good_example = 0
        test_problem, test_answer = test_sample["problem"], test_sample["answer"]
        # For each test sample, run CTI with full s1 process
        # evaluate and save
        s1_thoughts, s1_answer, s1dpt_hint, s1dpt_thoughts, s1dpt_answers = s1_stage2.evaluate_cti(test_problem, cti)
        if grade_answer(s1_answer, test_answer):
            # print("correct")
            correct_s1 += 1
        if grade_answer(s1dpt_answers, test_answer):
            # print("correct")
            correct_s1dpt += 1
        # save good examples
        if (not grade_answer(s1_answer, test_answer)) and grade_answer(s1dpt_answers, test_answer):
            good_example = 1
        # exit()
        all_predictions.append({
            "problem": test_problem,
            "cti": cti,
            "s1_thoughts": s1_thoughts,
            "s1_answer": s1_answer,
            "s1dpt_thoughts" : s1dpt_thoughts,
            "s1dpt_answers" : s1dpt_answers,
            "true_answer": test_answer,
            "good_example": good_example,
        })
        # if step == 2:
        #     break
    s1_acc = correct_s1 / len(math_data)
    s1_dpt_acc = correct_s1dpt / len(math_data)
    print(f"CTI: {cti} | S1-DPT Acc: {s1_dpt_acc}; S1 Acc: {s1_acc}")
    final_cti_results.append({
        "cti": cti,
        "s1_dpt_acc": s1_dpt_acc,
        "s1_acc": s1_acc
    })
    del s1_stage2

    # Save final results
    with open("cti_results_dpt.json", "w") as f:
        json.dump(final_cti_results, f, indent=2)

    print("Done. Final CTI results saved to cti_results_dpt.json.")


    csv_filename = "cti_prediction_results.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["problem", "cti", "s1_thoughts", "s1_answer", "s1dpt_thoughts", "s1dpt_answers", "true_answer", "good_example"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in all_predictions:
            writer.writerow(row)

    print(f"Detailed predictions saved to {csv_filename}")

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     import multiprocessing as mp
#     mp.set_start_method("spawn", force=True)  # Optional but explicit
#     main()

