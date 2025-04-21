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

def main():
    # PART: Settings
    N = 4
    NUM_CTI_PER_SAMPLE = 5

    # PART: Load MATH500
    ds = load_dataset("HuggingFaceH4/MATH-500")
    math_data = ds["test"]
    selected_samples = random.sample(list(math_data), N)
    test_samples = random.sample(list(math_data), 50)

    # # PART: S1 Module Init
    # s1_stage1 = S1Stage1()
    # s1_stage2 = S1Stage2()

    # Part: Prompt Template for CTI generation
    prompt_template = """A student is trying to solve a math question, he provide his thinking process, please provide an instruction that could help him better recheck and continue his thoughts. The instruction should be short and precise, less than 500 tokens. The instructin should describe how to check any type of math problems, do not mention any specific formula or principles related to this question.
    
    The math question is: [INPUT]
    His thought and correct answer is: [OUTPUT]
    
    The instruction should be short and precise, less than 1000 tokens. The instruction should describe how to check any type of math problems generally, do not mention any specific formula or principles related to only this question.
    Your instruction is: [APE]
    """

    def extract_boxed_content(result):
        start = result.find(r'\boxed{')
        if start == -1:
            return result  # No \boxed found

        start += len(r'\boxed{')
        brace_count = 1
        i = start
        while i < len(result):
            if result[i] == '{':
                brace_count += 1
            elif result[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return result[start:i]
            i += 1
        return result

    def extract_text_content(result):
        start = result.find(r'\text{')
        if start == -1:
            return result  # No \boxed found

        start += len(r'\text{')
        brace_count = 1
        i = start
        while i < len(result):
            if result[i] == '{':
                brace_count += 1
            elif result[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return result[start:i]
            i += 1
        return result


    def extract_math_answer(text):
        marker = "<|im_start|>answer"
        result = ""
        if marker in text:
            result = text.split(marker)[-1].strip()
        else:
            result = text.strip()
        # Remove everything after "<|im_end|>"(including"<|im_end|>")
        if "<|im_end|>" in result:
            result = result.split("<|im_end|>")[0].strip()
        # Remove everything after ""<|endoftext|>""
        if "<|endoftext|>" in result:
            result = result.split("<|endoftext|>")[0].strip()

        # if \boxed{...} in result, extract the content inside \boxed{...}
        # make sure the {} inside \boxed{...} is paired
        result = extract_text_content(result)
        result = extract_boxed_content(result)

        return result


    # Part: CTI Evaluation
    final_cti_results = []
    all_predictions = []
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
        th_ans = thought + "The right answer is:" + answer
        dataset = ([question], [th_ans])
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
        cti_step = 0
        for cti in ctis:
            # print('here')
            cti_step += 1
            correct = 0
            step = 0
            for test_sample in tqdm.tqdm(test_samples, desc=f"cti:{cti_step} | eval_step:{step}"):
            # for test_sample in random.sample(list(math_data), 5): # debug
                step += 1
                test_problem, test_answer = test_sample["problem"], test_sample["answer"]
                # For each test sample, run CTI with full s1 process
                output = s1_stage2.evaluate_cti(test_problem, cti)
                # get answer
                answer = extract_math_answer(output)
                # is_correct = grade_answer(given_answer, ground_truth)
                # print(f"output:{output}")
                # print("---------answer---------")
                # print(f"answer:{answer}")
                # print(f"true:{test_answer}")
                if grade_answer(answer, test_answer):
                    # print("correct")
                    correct += 1
                # exit()
                all_predictions.append({
                    "problem": test_problem,
                    "cti": cti,
                    "predicted_answer": answer,
                    "true_answer": test_answer
                })
                # if step == 2:
                #     break
            acc = correct / 50
            print(f"CTI: {cti} | Acc: {acc}")
            final_cti_results.append({
                "cti": cti,
                "accuracy": acc
            })
        del s1_stage2

    # Save final results
    with open("cti_results_dpt.json", "w") as f:
        json.dump(final_cti_results, f, indent=2)

    print("Done. Final CTI results saved to cti_results_dpt.json.")


    csv_filename = "cti_prediction_results.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["problem", "cti", "predicted_answer", "true_answer"]
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

