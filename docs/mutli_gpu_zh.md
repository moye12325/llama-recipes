# 使用多个GPU进行微调

要在多个GPU上进行微调，我们将使用两个软件包：

1. [PEFT](https://huggingface.co/blog/peft) 方法，特别是使用Hugging Face [PEFT](https://github.com/huggingface/peft) 库。

2. [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html) 可以帮助我们在多个GPU上并行训练。[更多细节](LLM_finetuning.md/#2-full-partial-parameter-finetuning)。

通过结合PEFT和FSDP，我们可以在一个节点或多个节点上同时对Llama 2模型进行微调。

## 要求
要运行示例，请确保使用以下命令安装要求：

```bash

pip install -r requirements.txt

```

**请注意，上面的requirements.txt将安装PyTorch 2.0.1版本，如果要运行FSDP + PEFT，请确保安装PyTorch Nightlies。**

## 如何运行

获得一台配备多个GPU的计算机（在本例中，我们测试了4个A100和A10）。
默认情况下，这将使用'samsum_dataset'进行摘要应用。

**一个节点上的多个GPU**：

**注意**：请确保使用PyTorch Nightlies来使用PEFT + FSDP。另外，请注意，FSDP目前不支持从bit&bytes进行int8量化。

```bash

torchrun --nnodes 1 --nproc_per_node 4  ../llama_finetuning.py --enable_fsdp --model_name /模型文件夹路径/7B --use_peft --peft_method lora --output_dir 保存PEFT模型的路径

```

上面的命令中使用的参数如下：

* `--enable_fsdp`：启用脚本中的FSDP的布尔标志
* `--use_peft`：启用脚本中的PEFT方法的布尔标志
* `--peft_method`：指定PEFT方法，这里我们使用'lora'，其他选项有'llama_adapter'和'prefix'。

我们在这里使用`torchrun`来为FSDP生成多个进程。

### 仅使用FSDP进行微调

如果想在不使用PEFT方法的情况下进行全参数微调，请使用以下命令。请将`nproc_per_node`更改为您可用的GPU数量。此命令已在8个A100，40GB的GPU上测试了`BF16`。

```bash

torchrun --nnodes 1 --nproc_per_node 8  llama_finetuning.py --enable_fsdp --model_name /模型文件夹路径/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 

```

**多个节点上的多个GPU**：

这里我们使用一个slurm脚本在多个节点上安排一个作业。

```bash

sbatch multi_node.slurm
# 在运行之前请在脚本中更改节点数量和每个节点的GPU数量。

```

## 如何使用不同的数据集运行？

目前支持4个数据集，可以在[Datasets config file](../configs/datasets.py)中找到。

* `grammar_dataset`：使用这个[notebook](../ft_datasets/grammar_dataset/grammar_dataset_process.ipynb)来获取和处理Jfleg和C4 200M数据集以进行语法检查。

* `alpaca_dataset`：要获取这个开源数据，请将`aplaca.json`下载到`ft_dataset`文件夹中。

```bash
wget -P ft_dataset https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
```

* `samsum_dataset`

要使用每个数据集运行，请在命令中设置`dataset`标志，如下所示：

```bash
# grammer_dataset
torchrun --nnodes 1 --nproc_per_node 4  ../llama_finetuning.py --enable_fsdp  --model_name /模型文件夹路径/7B --use_peft --peft_method lora --dataset grammar_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned  --pure_bf16 --output_dir 保存PEFT模型的路径

#

 alpaca_dataset

torchrun --nnodes 1 --nproc_per_node 4  ../llama_finetuning.py --enable_fsdp  --model_name /模型文件夹路径/7B --use_peft --peft_method lora --dataset alpaca_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 --output_dir 保存PEFT模型的路径


# samsum_dataset

torchrun --nnodes 1 --nproc_per_node 4  ../llama_finetuning.py --enable_fsdp --model_name /模型文件夹路径/7B --use_peft --peft_method lora --dataset samsum_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 --output_dir 保存PEFT模型的路径

```

## 如何配置设置？

* [Training config file](../configs/training.py) 是主要的配置文件，可以帮助指定运行的设置，并可以在[configs文件夹](../configs/)中找到。

它允许我们为训练设置指定一切，从`model_name`到`dataset_name`，`batch_size`等等。以下是支持的设置列表：

```python

model_name: str="PATH/to/LLAMA 2/7B"
enable_fsdp: bool= False
run_validation: bool=True
batch_size_training: int=4
num_epochs: int=3
num_workers_dataloader: int=2
lr: float=2e-4
weight_decay: float=0.0
gamma: float= 0.85
use_fp16: bool=False
mixed_precision: bool=True
val_batch_size: int=4
dataset = "samsum_dataset" # alpaca_dataset, grammar_dataset
micro_batch_size: int=1
peft_method: str = "lora" # None , llama_adapter, prefix
use_peft: bool=False
output_dir: str = "./ft-output"
freeze_layers: bool = False
num_freeze_layers: int = 1
quantization: bool = False
save_model: bool = False
dist_checkpoint_root_folder: str="model_checkpoints"
dist_checkpoint_folder: str="fine-tuned"
save_optimizer: bool=False

```

* [Datasets config file](../configs/datasets.py) 提供了数据集的可用选项。

* [peft config file](../configs/peft.py) 提供了支持的PEFT方法和相应的设置，可以进行修改。

* [FSDP config file](../configs/fsdp.py) 提供了FSDP设置，例如：

    * `mixed_precision`：指定是否使用混合精度，默认为true。

    * `use_fp16`：指定是否在混合精度中使用FP16，默认为false。我们建议不设置此标志，而只设置`mixed_precision`，它将使用`BF16`，这将有助于提高速度和节省内存，同时避免FP16标量精度问题。

    * `sharding_strategy`：这指定了FSDP的分片策略，可以是：
        * `FULL_SHARD`：分片模型参数、梯度和优化器状态，可节省最多内存。

        * `SHARD_GRAD_OP`：分片梯度和优化器状态，并在第一个`all_gather`后保留参数。这减少了通信开销，特别是在使用较慢网络时，尤其有益于多节点情况。但这会导致更高的内存消耗。

        * `NO_SHARD`：等效于DDP，不分片模型参数、梯度或优化器状态。在第一个`all_gather`后保留全部参数。

        * `HYBRID_SHARD`：仅适用于PyTorch Nightlies。它在节点内使用FSDP，在节点间使用DDP。适用于多节点情况，并且适用于较慢的网络，因为您的模型将适合一个节点内。

* `checkpoint_type`：指定用于保存模型的状态字典检查点类型。`FULL_STATE_DICT`从一个秩流到CPU流式传输每个模型分片的state_dict，并在CPU上组装完整的state_dict。`SHARDED_STATE_DICT`为每个秩保存一个检查点，并允许在不同的世界大小中重新加载模型。

* `fsdp_activation_checkpointing`：启用FSDP的激活检查点，这可以节省大量内存，但会导致在反向传递期间重新计算中间激活。所节省的内存可以重新投资于更高的批量大小，以增加吞吐量。我们建议您使用此选项。

* `pure_bf16`：它将模型转换为`BFloat16`，如果`optimizer`设置为`anyprecision`，则优化器状态也将保留在`BFloat16`中。如果需要，可以使用此选项。