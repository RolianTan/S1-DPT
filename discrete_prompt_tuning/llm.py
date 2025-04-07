"""Contains classes for querying large language models."""
from math import ceil
import os
import time
from tqdm import tqdm
from abc import ABC, abstractmethod
import tiktoken
from openai import OpenAI

# deepseek V3
client = OpenAI(api_key="", base_url="https://api.deepseek.com/beta")
# gpt
# client = OpenAI(api_key="")

def format_response_to_old_api(responses):
    formatted_response = {
        'choices': [],
        'usage': {
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
        },
    }

    for response in responses:
        for choice in response.choices:
            # Extract text from message.content if available
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                text = choice.message.content
            else:
                text = choice.text  # Fallback

            formatted_choice = {
                'text': text,  # <--- Fixed
                'index': choice.index,
                'logprobs': None,
                'finish_reason': choice.finish_reason,
            }
            formatted_response['choices'].append(formatted_choice)

        # Aggregate usage (unchanged)
        formatted_response['usage']['total_tokens'] += response.usage.total_tokens
        formatted_response['usage']['prompt_tokens'] += response.usage.prompt_tokens
        formatted_response['usage']['completion_tokens'] += response.usage.completion_tokens

    return formatted_response


def model_from_config(config, disable_tqdm=True):
    """Returns a model based on the config."""
    model_type = config["name"]
    if model_type == "GPT_forward":
        return GPT_Forward(config, disable_tqdm=disable_tqdm)
    elif model_type == "GPT_insert":
        # return GPT_Insert(config, disable_tqdm=disable_tqdm)
        pass
    raise ValueError(f"Unknown model type: {model_type}")


class LLM(ABC):
    """Abstract base class for large language models."""

    @abstractmethod
    def generate_text(self, prompt):
        """Generates text from the model.
        Parameters:
            prompt: The prompt to use. This can be a string or a list of strings.
        Returns:
            A list of strings.
        """
        pass

class GPT_Forward(LLM):
    """Wrapper for GPT-3."""

    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model."""
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        try:
            return fn(prompt, n)
        except BatchSizeException as e:
            if n == 1:
                raise e
            return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(fn, prompt, n // 2)

    def generate_text(self, prompt, n):
        if not isinstance(prompt, list):
            prompt = [prompt]
        batch_size = self.config['batch_size']
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        text = []

        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            text += self.auto_reduce_n(self.__generate_text, prompt_batch, n)
        return text

    def __generate_text(self, prompt, n):
        """Generates text from the model."""
        config = self.config['gpt_config'].copy()
        config['n'] = 1
        # If there are any [APE] tokens in the prompts, remove them
        for i in range(len(prompt)):
            prompt[i] = prompt[i].replace('[APE]', '').strip()
        responses = []
        for p in prompt:
            response = None
            while response is None:
                try:
                    response = client.completions.create(
                        **config, prompt=p)
                except Exception as e:
                    if 'is greater than the maximum' in str(e):
                        raise BatchSizeException()
                    print(e)
                    print('Retrying_fw_gen...')
                    time.sleep(5)
            responses.append(response)

        response = format_response_to_old_api(responses)
        # print(response)

        return [response['choices'][i]['text'] for i in range(len(response['choices']))]
class BatchSizeException(Exception):
    pass
