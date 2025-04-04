from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MAX_TOKENS_THINKING = 32000
NUM_IGNORE = 1

# Same as s1_test.py but replace the ignore_str with CTI
class S1Stage2:
    def __init__(self, model_name="simplescaling/s1.1-32B", tensor_parallel_size=4):
        self.model = LLM(model_name, tensor_parallel_size=tensor_parallel_size)
        self.tok = AutoTokenizer.from_pretrained(model_name)

    def evaluate_cti(self, problem, cti):
        # Initial problem prompt
        p = self._build_prompt(problem)
        prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n"
        stop_token_ids = self.tok("<|im_start|><|im_end|>")["input_ids"]
        sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS_THINKING,
            min_tokens=0,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature=0.0,
        )
        prompt += "<|im_start|>think"
        o = self.model.generate(
            prompt,
            sampling_params=sampling_params
        )
        ignore_str = cti # Replace Wait with CTI
        max_tokens_thinking_tmp = MAX_TOKENS_THINKING
        # Num of times to skip stop token
        for i in range(NUM_IGNORE):
            max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)
            prompt += o[0].outputs[0].text + ignore_str
            print(prompt)
            print("------------------------------------------------------------")
            sampling_params = SamplingParams(
                max_tokens=max_tokens_thinking_tmp,
                min_tokens=1,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                temperature=0.0,
            )
            o = self.model.generate(prompt, sampling_params=sampling_params)

        ### Final answer ###
        prompt += o[0].outputs[0].text # You can also append "Final Answer:" here like we do for some evaluations to prevent the model from just continuing to reason in its answer when early exiting
        stop_token_ids = self.tok("<|im_end|>")["input_ids"]
        sampling_params = SamplingParams(
            max_tokens=32768,
            min_tokens=0,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature=0.0,
        )
        o = self.model.generate(prompt, sampling_params=sampling_params)

        return prompt + o[0].outputs[0].text

    def _build_prompt(self, problem):
        return (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + problem + "<|im_end|>\n<|im_start|>assistant\n"
        )
