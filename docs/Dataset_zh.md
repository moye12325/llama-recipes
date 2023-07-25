# 数据集和评估指标

提供的微调脚本允许您通过将 `dataset` 参数传递给 `llama_finetuning.py` 脚本来选择三个数据集中的一个。当前可用的选项包括 `grammar_dataset`、`alpaca_dataset` 和 `samsum_dataset`。请注意：使用任何数据集都应符合数据集的相关许可证（包括但不限于非商业用途）。

* [grammar_dataset](https://huggingface.co/datasets/jfleg) 包含 15 万对英语句子和可能的更正版本。
* [alpaca_dataset](https://github.com/tatsu-lab/stanford_alpaca) 提供了 5.2 万个由 `text-davinci-003` 生成的指令-回应对。
* [samsum_dataset](https://huggingface.co/datasets/samsum) 包含约 1.6 万个类似于聊天对话的对话和摘要。

## 添加自定义数据集

可以通过以下步骤轻松地添加自定义数据集以扩展可用的数据集列表：

每个数据集都在 [configs/dataset.py](../configs/dataset.py) 中有相应的配置（数据类），其中包含数据集名称、训练/验证集名称，以及可选参数，比如数据文件等。

此外，在 [ft_datasets](../ft_datasets) 文件夹中，每个数据集都有一个预处理函数。
数据集的返回数据需要能够由微调模型的前向方法以 ```model(**data)``` 的形式进行使用。
对于 CausalLM 模型，通常需要将数据组织成一个带有 "input_ids"、"attention_mask" 和 "labels" 字段的字典形式。

要添加自定义数据集，需要执行以下步骤。

1. 根据上述描述创建一个数据集配置。示例可以在 [configs/dataset.py](../configs/dataset.py) 中找到。
2. 创建一个预处理函数，该函数加载数据并返回一个符合 PyTorch 风格的数据集。预处理函数的签名应为 (dataset_config, tokenizer, split_name)，其中 split_name 将是在数据类中定义的用于训练/验证拆分的字符串。
3. 在 [utils/dataset_utils.py](../utils/dataset_utils.py) 中的 DATASET_PREPROC 字典中将数据集名称和预处理函数注册为键值对。
4. 将训练配置中的数据集字段设置为数据集名称，或者使用 llama_finetuning.py 训练脚本的 --dataset 选项。

## 应用
以下列出了其他可用于微调的数据集及其主要用途。

### 问答（Q&A），这些数据集也可用于评估
- [MMLU](https://huggingface.co/datasets/lukaemon/mmlu/viewer/astronomy/validation)
- [BoolQ](https://huggingface.co/datasets/boolq)
- [NarrativeQA](https://huggingface.co/datasets/narrativeqa)
- [NaturalQuestions](https://huggingface.co/datasets/natural_questions)（封闭书本）
- [NaturalQuestions](https://huggingface.co/datasets/openbookqa)（开放书本）
- [QuAC](https://huggingface.co/datasets/quac)
- [HellaSwag](https://huggingface.co/datasets/hellaswag)
- [OpenbookQA](https://huggingface.co/datasets/openbookqa)
- [TruthfulQA](https://huggingface.co/datasets/truthful_qa)（对于事实检查/误导性模型可能有帮助）

### 指令微调
- [Alpaca](https://huggingface.co/datasets/yahma/alpaca-cleaned)	52k	指令微调
- [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k) 15k	15k	指令微调

### 简单文本生成用于快速测试
[英文名言](https://huggingface.co/datasets/Abirate/english_quotes)	2508	多标签文本分类，文本生成

### 推理，主要用于对 LLM 进行评估
- [bAbI](https://research.facebook.com/downloads/babi/)
- [Dyck](https://huggingface.co/datasets/dyk)
- [GSM8K](https://huggingface.co/datasets/gsm8k)
- [MATH](https://github.com/hendrycks/math)
- [APPS](https://huggingface.co/datasets/codeparrot/apps)
- [HumanEval](https://huggingface.co/datasets/openai_humaneval)
- [LSAT](https://huggingface.co/datasets/dmayhem93/agieval-lsat-ar)
- [Entity matching](https://huggingface.co/datasets/lighteval/EntityMatching)

### 毒性评估
- [Real_toxic_prompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts)

### 偏见评估
- [Crows_pair](https://huggingface.co/datasets/crows_pairs) 性别偏见
- WinoGender 性别偏见

### 有用链接
有关评估数据集的更多信息，请参阅 [HELM](https://crfm.stanford.edu/helm/latest/)。