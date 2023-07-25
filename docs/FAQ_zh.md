# 常见问题解答（FAQ）

以下是经常被问到的问题，我们在使用过程中认为这些问题是有用的。

1. FSDP是否支持一个FSDP单元内的混合精度？意思是在一个FSDP单元内，一些参数为Fp16/Bf16，而其他参数为FP32。

    FSDP要求每个FSDP单元具有一致的精度，因此目前不支持这种情况。可能会在未来添加该功能，但目前没有具体时间表。

2. FSDP如何处理混合的梯度要求？

    FSDP不支持在一个FSDP单元内使用混合的`require_grad`。这意味着如果您计划冻结某些层，您需要在FSDP单元级别而不是模型层级别上进行。例如，假设我们的模型有30个解码器层，我们想要冻结底部的28个层，只训练顶部的2个Transformer层。在这种情况下，我们需要确保顶部两个Transformer层的`require_grad`被设置为`True`。

3. PEFT方法在梯度要求/层冻结方面如何与FSDP一起使用？

    我们在自动包装策略中将PEFT模块与transformer层分开包装，这将导致PEFT模型的`require_grad=True`，而模型的其余部分为`require_grad=False`。

4. 我可以添加自定义数据集吗？

    是的，您可以在[这里](Dataset.md)找到更多信息。

5. 部署这些模型的硬件SKU要求是什么？

    硬件要求取决于延迟、吞吐量和成本限制。为了获得良好的延迟，模型被分割到具有张量并行性的多个GPU中，使用NVIDIA A100s或H100s的机器。但也可以使用TPU、其他类型的GPU，比如A10G、T4、L4，甚至是普通硬件来部署这些模型（例如：https://github.com/ggerganov/llama.cpp）。如果在CPU上工作，可以参考英特尔的这篇[博文](https://www.intel.com/content/www/us/en/developer/articles/news/llama2.html)来了解Llama 2在CPU上的性能。

6. 对于微调Llama预训练模型，需要什么硬件SKU？

    微调要求根据数据量、完成微调所需的时间和成本限制而异。为了微调这些模型，通常使用多个NVIDIA A100机器进行数据并行，节点内使用数据和张量并行的混合方式。但也可以使用单个机器，或者其他GPU类型，比如NVIDIA A10G或H100（例如：alpaca模型在单个RTX4090上进行了训练：https://github.com/tloen/alpaca-lora）。