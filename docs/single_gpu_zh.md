# 使用单个GPU进行微调

要在单个GPU上运行微调，我们将使用两个软件包：

1- [PEFT](https://huggingface.co/blog/peft) 方法，具体来说是使用HuggingFace [PEFT](https://github.com/huggingface/peft)库。

2- [BitandBytes](https://github.com/TimDettmers/bitsandbytes) int8 量化。

通过结合PEFT和Int8量化，我们可以在一个消费级GPU上（例如A10）上对Llama 2 7B模型进行微调。

## 要求
要运行示例，请确保使用以下命令安装要求：

```bash

pip install -r requirements.txt

```

**请注意，上面的requirements.txt将安装PyTorch 2.0.1版本，如果要运行FSDP + PEFT，请确保安装PyTorch Nightlies。**

## 如何运行？

获取一台配备一个GPU的计算机，如果使用多个GPU的计算机，请确保只有一个GPU可见，使用`export CUDA_VISIBLE_DEVICES=GPU:id`，然后运行以下命令。默认情况下，这将使用'samsum_dataset'进行摘要应用。

```bash

python ../llama_finetuning.py  --use_peft --peft_method lora --quantization --use_fp16 --model_name /模型文件夹路径/7B --output_dir 保存PEFT模型的路径

```

上面命令中使用的参数如下：

* `--use_peft`：启用脚本中的PEFT方法的布尔标志
* `--peft_method`：指定PEFT方法，这里我们使用'lora'，其他选项有'llama_adapter'和'prefix'。
* `--quantization`：启用int8量化的布尔标志


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

python ../llama_finetuning.py  --use_peft --peft_method lora --quantization  --dataset grammar_dataset --model_name /模型文件夹路径/7B --output_dir 保存PEFT模型的路径

# alpaca_dataset

python ../llama_finetuning.py  --use_peft --peft_method lora --quantization  --dataset alpaca_dataset --model_name /模型文件夹路径/7B --output_dir 保存PEFT模型的路径


# samsum_dataset

python ../llama_finetuning.py  --use_peft --peft_method lora --quantization  --dataset samsum_dataset --model_name /模型文件夹路径/7B --output_dir 保存PEFT模型的路径

```

## 如何配置设置？

* [Training config file](../configs/training.py) 是主要的配置文件，用于指定运行的设置，可以在[configs文件夹](../configs/)中找到。

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
one_gpu: bool = False
save_model: bool = False
dist_checkpoint_root_folder: str="model_checkpoints"
dist_checkpoint_folder: str="fine-tuned"
save_optimizer: bool=False

```

* [Datasets config file](../configs/datasets.py) 提供了数据集的可用选项。

* [peft config file](../configs/peft.py) 提供了支持的PEFT方法和相应的设置，可以进行修改。