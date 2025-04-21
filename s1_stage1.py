from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MAX_TOKENS_THINKING = 15000

# Just retrieve the thinking process (before the first time stop)
class S1Stage1:
    def __init__(self, model_name="simplescaling/s1.1-1.5B", tensor_parallel_size=1):
        self.model = LLM(model_name, tensor_parallel_size=tensor_parallel_size)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.stop_token_ids = self.tok("<|im_end|>")["input_ids"]
        self.sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS_THINKING,
            stop_token_ids=self.stop_token_ids,
            skip_special_tokens=False,
            temperature=0.0,
        )

    def extract_thought(self, problem):
        prompt = (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. The thinking process should be shorter than 15000 tokens.<|im_end|>\n"
            "<|im_start|>user\n" + problem + "<|im_end|>\n<|im_start|>assistant\n<|im_start|>think"
        )
        output = self.model.generate(prompt, sampling_params=self.sampling_params)[0].outputs[0].text
        return output.strip()
