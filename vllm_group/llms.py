import vllm
import ray
import torch


@ray.remote
class LLM:
    def __init__(self, *args, **kwargs):
        kwargs["distributed_executor_backend"] = "ray"
        self.llm = vllm.LLM(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)


class LLMs:
    def __init__(
        self, *args, tensor_parallel_size=1, pipeline_parallel_size=1, **kwargs
    ):
        num_gpu = torch.cuda.device_count()
        num_gpu_per_llm = tensor_parallel_size * pipeline_parallel_size
        self.llms = []

        funcs = {}
        for key, value in kwargs.items():
            if key.startswith("func_of_"):
                assert callable(
                    value
                ), "value of arguments starting with 'func_of_' should be callable"
                funcs[key[len("func_of_") :]] = value

        for key in funcs:
            kwargs.pop(f"func_of_{key}")

        for idx in range(num_gpu // num_gpu_per_llm):
            other_kwargs = {key: value(idx) for key, value in funcs.items()}
            llm = LLM.remote(
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                *args,
                **kwargs,
                **other_kwargs,
            )
            self.llms.append(llm)

    def __len__(self):
        return len(self.llms)

    def __getitem__(self, index):
        return self.llms[index]


if __name__ == "__main__":
    llms = LLMs(
        "Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=2,
        trust_remote_code=True,
        func_of_seed=lambda idx: idx,
    )
    outputs = []
    for i in range(5):
        outputs.append(
            llms[i % len(llms)].generate.remote(
                "Why more is different? Please explain in one sentence.",
                vllm.SamplingParams(
                    max_tokens=128,
                    top_p=0.7,
                    temperature=0.8,
                    stop=["</s>", "<|im_end|>", "<|endoftext|>"],
                ),
            )
        )

    outputs = [ray.get(output) for output in outputs]
    for output in outputs:
        print(output[0].outputs[0].text)
        print("=" * 50)
