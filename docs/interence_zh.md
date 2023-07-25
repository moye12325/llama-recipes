# 推理

我们提供了一个推理脚本（[inference.py](../inference/inference.py)），根据训练过程中进行的微调类型，[推理脚本](../inference/inference.py)接受不同的参数。
要对所有模型参数进行微调，需要将训练的输出目录作为--model_name参数提供。
对于像lora这样的参数高效方法，需要将基本模型作为--model_name参数，并将训练的输出目录作为--peft_model参数提供。
此外，还需要提供一个模型的提示（prompt）文本文件。提示文件可以通过标准输入传递，也可以作为--prompt_file参数提供。

**内容安全性**
推理脚本还支持对用户提示和模型输出进行安全检查。特别地，我们使用了两个包，[AuditNLG](https://github.com/salesforce/AuditNLG/tree/main)和[Azure content safety](https://pypi.org/project/azure-ai-contentsafety/1.0.0b1/)。

**注意**
如果使用Azure内容安全性，请确保按照[这里](https://pypi.org/project/azure-ai-contentsafety/1.0.0b1/)的描述获取终端点和API密钥，并将它们添加为以下环境变量：`CONTENT_SAFETY_ENDPOINT`和`CONTENT_SAFETY_KEY`。

示例：

 ```bash
# 对所有参数进行完整微调
cat <test_prompt_file> | python inference/inference.py --model_name <training_config.output_dir> --use_auditnlg
# 使用PEFT方法
cat <test_prompt_file> | python inference/inference.py --model_name <training_config.model_name> --peft_model <training_config.output_dir> --use_auditnlg
# 提示作为参数
python inference/inference.py --model_name <training_config.output_dir> --prompt_file <test_prompt_file> --use_auditnlg
 ```
推理文件夹包含用于摘要用例的测试提示：
```
inference/samsum_prompt.txt
...
```

**聊天完成**
推理文件夹还包含一个聊天完成的示例，该示例在微调的模型中添加了内置的安全功能以处理提示标记。要运行此示例：

```bash
python chat_completion.py --model_name "PATH/TO/MODEL/7B/" --prompt_file chats.json  --quantization --use_auditnlg

```

## 其他推理选项

其他推理选项包括：

[**vLLM**](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html)：要使用vLLM，您需要按照[这里](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#installation)的说明进行安装。
安装完成后，您可以使用[inference](../inference/vLLM_inference.py)文件夹中提供的vLLM_ineference.py脚本。

以下是在inference文件夹中找到的vLLM_inference.py脚本的运行示例。

``` bash
python vLLM_inference.py --model_name <PATH/TO/MODEL/7B>
```

[**TGI**](https://github.com/huggingface/text-generation-inference)：文本生成推理（TGI）是另一种可用于推理的选项。有关如何设置和使用TGI的更多信息，请参见[这里](../inference/hf-text-generation-inference/README.md)。