# Llama 2 模型微调/推理配方和示例

'llama-recipes' 仓库是 [Llama 2 模型](https://github.com/facebookresearch/llama) 的伴侣。该仓库的目标是提供示例，快速入门进行领域自适应的模型微调以及如何运行微调模型的推理。为了方便使用，示例使用了 Hugging Face 转换后的模型版本。有关模型转换的步骤，请参见[此处](#model-conversion-to-hugging-face)。

Llama 2 是一项新技术，其使用可能带来潜在风险。迄今为止进行的测试无法涵盖所有场景。为了帮助开发人员应对这些风险，我们创建了[负责任使用指南](https://github.com/facebookresearch/llama/blob/main/Responsible-Use-Guide.pdf)。更多详细信息也可以在我们的研究论文中找到。要下载模型，请遵循[Llama 2 仓库](https://github.com/facebookresearch/llama)上的说明。

# 目录
1. [快速入门](#quick-start)
2. [模型微调](#fine-tuning)
   - [单 GPU](#single-gpu)
   - [多 GPU 单节点](#multiple-gpus-one-node)
   - [多 GPU 多节点](#multi-gpu-multi-node)
3. [推理](./docs/inference.md)
4. [模型转换](#model-conversion-to-hugging-face)
5. [仓库组织结构](#repository-organization)
6. [许可证和可接受使用政策](#license)

# 快速入门

[Llama 2 Jupyter Notebook](quickstart.ipynb)：这个 Jupyter Notebook 将引导您如何在文本摘要任务上对 Llama 2 模型进行微调，使用的数据集是 [samsum](https://huggingface.co/datasets/samsum)。该 Notebook 使用参数高效微调（PEFT）和 int8 量化，在单个 GPU（如 A10，24GB 显存）上对 7B 模型进行微调。

**注意** 所有在[配置文件](./configs/)中定义的设置可以通过命令行参数传递给脚本，无需直接从配置文件中更改。

**注意** 如果需要在 PEFT 模型上使用 FSDP，请确保使用 PyTorch Nightlies。

**要获取更深入的信息，请查看以下内容：**

* [单 GPU 微调](./docs/single_gpu.md)
* [多 GPU 微调](./docs/mutli_gpu.md)
* [LLM 微调](./docs/LLM_finetuning.md)
* [添加自定义数据集](./docs/Dataset.md)
* [推理](./docs/inference.md)
* [常见问题](./docs/FAQ.md)

## 要求
要运行这些示例，请确保安装了所需的依赖项，使用以下命令安装：

```bash
pip install -r requirements.txt
```

**请注意，上述 requirements.txt 将安装 PyTorch 2.0.1 版本。如果要运行 FSDP + PEFT，请确保安装 PyTorch Nightlies。**

# 模型微调

为了对 Llama 2 模型进行领域特定的微调，配方中包含了 PEFT、FSDP、PEFT+FSDP 等多个示例数据集。详情请参见 [LLM 微调](./docs/LLM_finetuning.md)。

## 单 GPU 和多 GPU 微调

如果您想立即进行单 GPU 或多 GPU 微调，请在单个 GPU 上运行以下示例，例如 A10、T4、V100、A100 等。以下示例和配方中的所有参数需要进一步调整，以获得符合模型、方法、数据和任务要求的期望结果。

**注意：**
* 要更改以下命令中的数据集，请传递 `dataset` 参数。当前数据集的选项包括 `grammar_dataset`、`alpaca_dataset` 和 `samsum_dataset`。数据集的描述以及如何添加自定义数据集的信息，请参阅 [Dataset.md](./docs/Dataset.md)。对于 `grammar_dataset` 和 `alpaca_dataset`，请确保按照 [此处](./docs/single_gpu.md#how-to-run-with-different-datasets) 的建议说明进行设置。

* 默认数据集和其他 LORA 配置已设置为 `samsum_dataset`。

* 确保在[训练配置](./configs/training.py)中设置正确的模型路径。

### 单 GPU：

```bash
# 如果在多 GPU 机器上运行
export CUDA_VISIBLE_DEVICES=0

python llama_finetuning.py  --use_peft --peft_method lora --quantization --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model
```

在上面的命令中，我们使用参数高效方法（PEFT），如下一节所述。要运行上面的命令，请确保传递 `peft_method` 参数，可以设置为 `lora`、`llama_adapter` 或 `prefix`。

**注意** 如果您在具有多个 GPU 的计算机上运行，请确保只有一个 GPU 可见，可以使用 `export CUDA_VISIBLE_DEVICES=GPU:id`。

**确保在[训练.py](configs/training.py)中设置 [save_model](configs/training.py) 来保存模型。请务必检查[训练配置](configs/training.py)中的其他训练设置，以及需要的配置文件夹中的其他设置，或者它们也可以作为参数传递给训练脚本。**

### 多 GPU 单节点：

**注意** 请确保在使用 PEFT+FSDP 时使用 PyTorch Nightlies。此外，请注意 FSDP 不支持来自 Bit&Bytes 的 int8 量化。

```bash
torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py --enable_fsdp --use

_peft --peft_method lora --model_name /patht_of_model_folder/7B --pure_bf16 --output_dir Path/to/save/PEFT/model
```

在上述示例中，我们使用 FSDP，如下一节中所述，它可以与 PEFT 方法一起使用。要使用 FSDP 进行 PEFT 方法，请确保传递 `use_peft` 和 `peft_method` 参数以及 `enable_fsdp` 参数。在上面的示例中，我们使用 `BF16` 进行训练。

### 仅使用 FSDP 进行微调

如果您希望在不使用 PEFT 方法的情况下运行全参数微调，请使用以下命令。确保将 `nproc_per_node` 更改为您可用的 GPU 数量。该命令在 8 个 A100，40GB 显存的 GPU 上进行了 BF16 测试。

```bash
torchrun --nnodes 1 --nproc_per_node 8  llama_finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned
```

### 多 GPU 多节点：

```bash
sbatch multi_node.slurm
# 在运行脚本之前，请在脚本中更改节点数和每个节点的 GPU 数量。
```

您可以在[此处](./docs/LLM_finetuning.md)阅读更多有关我们的微调策略的信息。

# 将模型转换为 Hugging Face 格式

该仓库中的配方和笔记本使用 Hugging Face 的 transformers 库提供的 Llama 2 模型定义。

给定原始检查点位于 models/7B，您可以安装所有要求并转换检查点，方法如下：

```bash
## 从源代码安装 HuggingFace Transformers
pip install git+https://github.com/huggingface/transformers
cd transformers

python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir models_hf/7B
```

# 仓库组织结构

这个仓库的组织如下：

[configs](configs/)：包含 PEFT 方法、FSDP、数据集的配置文件。

[docs](docs/)：包含单 GPU 和多 GPU 微调的示例配方。

[ft_datasets](ft_datasets/)：包含每个数据集的单独脚本，用于下载和处理数据。请注意，使用任何数据集必须符合数据集的相关许可证（包括但不限于非商业用途）。

[inference](inference/)：包含对微调模型进行推理的示例以及如何安全使用它们。

[model_checkpointing](model_checkpointing/)：包含 FSDP 检查点处理程序。

[policies](policies/)：包含提供不同策略的 FSDP 脚本，例如混合精度、transformer 封装策略和激活检查点以及任何精度优化器（用于以纯 BF16 模式运行 FSDP）。

[utils](utils/)：包含实用工具文件：

- `train_utils.py` 提供训练/评估循环和更多训练实用工具。

- `dataset_utils.py` 用于获取预处理数据集。

- `config_utils.py` 用于重写从 CLI 接收的配置。

- `fsdp_utils.py` 提供 PEFT 方法的 FSDP 封装策略。

- `memory_utils.py` 上下文管理器，用于跟踪训练循环中的不同内存统计信息。

# 许可证

请参阅许可证文件 [here](LICENSE) 和可接受使用政策 [here](USE_POLICY.md)。