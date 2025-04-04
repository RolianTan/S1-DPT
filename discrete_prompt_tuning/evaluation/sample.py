from discrete_prompt_tuning import llm, data, evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm

special_output_token = '[[[[OUTPUT]]]]'

def split_into_batches(data, bs):
    return [data[i:i + bs] for i in range(0, len(data), bs)]

def get_query(prompt, eval_template, input_, demo_data, demos_template):
    demos = demos_template.fill(demo_data)

    if isinstance(input_, list):
        count = len(input_)
        formatted_input = "\n".join([f"{i + 1}. {inp}" for i, inp in enumerate(input_)])
    else:
        count = 1  # Single input
        formatted_input = input_

    query = eval_template.fill(
        count=count,
        prompt=prompt,
        full_demo=demos,
        input=formatted_input,
        output=special_output_token
    )
    return query


# Completely rewritten evaluator
def sample_evaluator(prompts, eval_template, eval_data, demos_template, demo_data, config, save_path='evaluation_result.csv'):
    model = llm.model_from_config(config['model'])
    batch_size = config['model']['batch_size']
    batched_sentences = split_into_batches(eval_data[0], batch_size)
    batched_labels = split_into_batches(eval_data[1], batch_size)
    batched_eval_data = list(zip(batched_sentences, batched_labels))
    all_preds = []
    total_prompts = len(prompts)
    for i, prompt in enumerate(prompts):
        total_correct = 0
        total_samples = 0
        for batch in tqdm(batched_eval_data, desc=f"Prompt: {i}/{total_prompts}"):
            input_, output_ = batch
            # Random sample
            # query = get_query(prompt, eval_template, input_,
            #                      data.subsample_data(demo_data, config['num_few_shot']),
            #                      demos_template)

            query = get_query(prompt, eval_template, input_,
                                 demo_data,
                                 demos_template)

            response = model.generate_text([query], n=1)[0]
            preds = parse_batched_response(response, len(input_))
            # Compare predictions with true labels
            for pred, true_label in zip(preds, output_):
                if extract_label(pred) == true_label.strip().lower():
                    total_correct += 1
                total_samples += 1
            # print(query)

        # Compute accuracy for the current prompt
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        all_preds.append(accuracy)

    return SampleEvaluationResult(prompts, all_preds, save_path)

# New result class
class SampleEvaluationResult(evaluate.EvaluationResult):
    def __init__(self, prompts, accuracies, save_path):
        self.prompts = prompts
        self.accuracies = accuracies
        self.save_path = save_path

    def sorted(self, method='default'):
        sorted_pairs = sorted(zip(self.accuracies, self.prompts), reverse=True)
        return [p for _, p in sorted_pairs], [a for a, _ in sorted_pairs]

    def in_place(self, method='default'):
        """Required by parent abstract class"""
        return self.prompts, self.predictions

    def __str__(self):
        s = 'Accuracy: Prompt\n----------------\n'
        sorted_p, sorted_a = self.sorted()

        # save to csv
        df = pd.DataFrame({
            "Prompt": sorted_p,
            "Accuracy": sorted_a
        })
        df.to_csv(self.save_path, index=False)

        for prompt, acc in zip(sorted_p, sorted_a):
            s += f"{acc:.2f}: {prompt}\n"

        return s


def parse_batched_response(response, expected_count):
    """Parse the model's response into a list of outputs."""
    outputs = []
    lines = response.strip().split('\n')

    # Extract the first N numbered responses
    for line in lines:
        if line.strip() and line[0].isdigit():
            parts = line.split('.', 1)
            if len(parts) > 1:
                outputs.append(parts[1].strip())
        if len(outputs) >= expected_count:
            break

    return outputs[:expected_count]


# Helper function
def extract_label(text):
    text = text.strip().lower()
    if any(kw in text for kw in ['positive', 'pos']):
        return 'positive'
    if any(kw in text for kw in ['negative', 'neg']):
        return 'negative'
    return None