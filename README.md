## vLLM Group

In many usecases of [vllm-project/vllm](https://github.com/vllm-project/vllm), we hope to use all the GPUs we have on the local node, with data parallel instead of a over-extended tensor parallel. This project is trying to help with that. You could use the following code to initialize `torch.cuda.device_count() / tensor_parallel_size` models with vLLM directly through ray.

```python
from vllm_group import LLMs

# Note that currently we need to download the model to local dir first.
llms = LLMs(
    "/root/Qwen2.5-7B-Instruct",
    tensor_parallel_size=2,
    trust_remote_code=True,
    func_of_seed=lambda idx: idx,
)
```

- Note that the value of arguments that start with `func_of_` need to be callable, so that different llm could receive different argument value. For example, for `func_of_seed=lambda idx: idx`, each llm will have its index as the `seed` argument.

And then you could generate with each llm asynchronizely like:

```python
import ray

outputs = []
for i in range(5):
    outputs.append(llms[i % len(llms)].generate.remote(
        "Hey, how are you doing?",
    ))

outputs = [ray.get(output) for output in outputs]
```

This project is highly inspired by [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
