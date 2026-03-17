# 解决 Python 项目环境配置

**Date**: March 17, 2026 at 01:49 PM
**Turns**: 2
**Source**: [Gemini Chat](https://gemini.google.com/app/98431610d951200a?hl=ja)

---

## Turn 1

### 🤖 Assistant

这个问题确实是整个“向量魔改”计划中最考验工程能力的环节。
现在的矛盾点在于：**模型推理时需要一个完整的信号**。在作者的代码中，这个信号是“指令字符串 + 压缩文本字符串”；而在你的方法中，由于压缩内容变成了 Tensor，我们不能直接用 `+` 号把字符串和向量连起来。
我们需要在 `qa_manager.py` 中对三个核心点进行修改，将推理链路从“纯文本”升级为“向量拼接”。以下是详细的操作指南：

---

### 修改点 1：底层推理函数 `_lm_generate`
**位置：**`TaskManager` 类（约第 181 行）
**目的：** 让模型具备接收 `inputs_embeds` 的能力。
你需要将原本只接收字符串的 `_lm_generate` 改造成一个“双模”函数。

```python
def _lm_generate(self, prompt, num_retry=5):
    # --- 新增：处理向量输入的分支 ---
    if isinstance(prompt, torch.Tensor):
        with torch.no_grad():
            # 这里的 prompt 已经是拼接好的 [1, seq_len, 768] 的 Tensor
            outputs = self.model.generate(
                inputs_embeds=prompt,  # 注入点
                generation_config=self.generation_config,
                max_new_tokens=450,
                pad_token_id=self.tokenizer.eos_token_id
            )
        # 生成的 outputs 是 token id，需要解码回文字进行后续评估
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- 原有的字符串处理逻辑 ---
    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=520)
    input_ids = inputs.input_ids.to(self.model.device)
    # ... 其余代码保持不变 ...

```

---

### 修改点 2：任务提示词模板 `prompt_for_the_task`
**位置：**`Summarisation` 类（或你正在测试的其他任务类，约第 262 行）
**目的：** 分离“指令”与“内容”。
当内容是 Tensor 时，我们不能直接格式化字符串。我们需要一个方式只获取指令部分（如 "Summarize: "）。

```python
def prompt_for_the_task(self, context: ArxivContext):
    # 增加一个逻辑判断：如果 context.context 是 Tensor，只返回指令文本
    if isinstance(context.context, torch.Tensor):
        if self.model_type == "flan-t5-xxl":
            return "Summarize: "
        elif not self.model_instruct_tuned:
            return "The following is a document, please provide a summary: "
        else:
            return "Summarize the above content: "
    
    # 原有的字符串格式化逻辑
    if self.model_type == "flan-t5-xxl":
        prompt = f"Summarize: {context.context}"
    # ... 其余原代码 ...

```

---

### 修改点 3：核心执行逻辑 `get_answer`
**位置：**`Summarisation` 类（约第 281 行）
**目的：** 执行真正的“向量拼接”。
这是最关键的逻辑：我们要把“指令文字”转成向量，然后和你的“魔改向量”在 $dim=1$（序列长度维度）上拼接。

```python
def get_answer(self):
    ans = self.ans
    answer_of_contexts = ans.answer_of_contexts if ans.answer_of_contexts is not None else {}
    
    for context_type, contexts in ans.contexts_dict.items():
        answer_of_contexts[context_type] = []
        
        for context in contexts:
            # --- 核心修改：处理向量拼接 ---
            if isinstance(context.context, torch.Tensor):
                # 1. 获取纯指令字符串
                instruction_text = self.prompt_for_the_task(context)
                
                # 2. 将指令转为向量
                # instr_ids 形状: [1, instr_len]
                instr_ids = self.tokenizer.encode(instruction_text, return_tensors="pt").to(self.model.device)
                # 通过模型的词嵌入层转为向量: [1, instr_len, 768]
                instr_embeds = self.model.get_input_embeddings()(instr_ids)
                
                # 3. 拼接！
                # instr_embeds: [1, instr_len, 768]
                # context.context: [1, magic_len, 768]
                # 拼接后: [1, instr_len + magic_len, 768]
                full_prompt_embeds = torch.cat([instr_embeds, context.context], dim=1)
                
                # 4. 调用魔改后的推理函数
                summary = self._lm_generate(full_prompt_embeds)
                
                # 记录结果（为了适配作者的保存逻辑）
                summary_save_file = os.path.join(self.summary_saved_path, f"{ans.dataset_type}_{self.model_type}_{context.id}_{context_type}_{self.mask_ratio}.tsv")
                with open(summary_save_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                answer_of_contexts[context_type].append(summary)
            
            else:
                # --- 原有的字符串批量/单条处理逻辑 ---
                # ... (此处保留作者原有的代码) ...

```

---

### 总结：为什么要这么改？

1. **统一数学空间**：GPT-2 内部运算全是向量。通过 `model.get_input_embeddings()`，我们将“指令文字”拉到了和你“魔改向量”相同的 768 维空间。
2. **顺序逻辑**：拼接时 `instr_embeds` 在前，`magic_vector` 在后。这样模型就会先看到“请摘要：”，然后看到你卷积后的“魔改信号”，从而触发摘要行为。
3. **结果对齐**：由于 `_lm_generate` 最终解码（Decode）出了 `str`，它会被存入 `.tsv` 文件，这让作者原有的 `evaluate` 函数完全不需要修改就能运行。

### 💡 一个性能提示
由于你的魔改方法使用了 `inputs_embeds`，**不能直接使用作者原有的 _lm_answer_batch（批量推理）**。因为作者的批量推理使用的是 Hugging Face 的 `pipeline` 接口，该接口对自定义向量的支持非常差。建议你先采用我上面写的“逐条循环”模式，确认分数没问题后，再考虑如何优化速度。
**你想让我帮你写出 my_compression.py 里那个用于生成初始向量并进行卷积处理的具体类代码吗？**

## Turn 2

### 🤖 Assistant

这份自述文件（README.md）是专门为你基于 **Selective Context** 项目修改后的版本定制的。它记录了你从“离散文本删除”进化到“连续向量卷积”的创新探索。

---

# Selective Context (Modified: Convolutional Vector Compression)
本项目基于原作者的 Selective Context for LLMs 进行深度定制开发。原项目通过计算自信息量（Self-Information）来删除冗余 Token，而本修改版引入了**“软压缩（Soft Compression）”**概念，通过卷积神经网络（CNN）在向量空间对上下文进行语义融合。

## 🌟 核心修改介绍
与原版项目相比，本版本在保持原有评估框架的同时，实现了以下突破：

- **连续空间压缩 (Soft Compression)**：不再简单地删除单词，而是将文本转化为 Embedding 向量，通过 `1D Convolution` 进行下采样，缩短序列长度的同时保留关键语义。
- **直接向量注入 (Inputs Embeddings Injection)**：绕过了传统的 Tokenizer 限制，直接将魔改后的 Tensor 注入模型的隐藏层（Hidden States），实现了更底层的模型交互。
- **平行对比框架**：在 `main.py` 中新增了 `my-method` 路径，可与原版的 `self-info` 和 `Random` 方法在完全相同的数据集下进行 BLEU、ROUGE、METEOR 评分对比。

---

## 🛠️ 环境搭建 (Windows 11 + uv)
本项目针对 Windows 环境进行了优化，建议使用 `uv` 包管理器以获得最快的安装体验。

### 1. 创建并激活虚拟环境

```bash
uv venv --python 3.10
source .venv/Scripts/activate

```

### 2. 安装核心依赖

```bash
uv pip install torch transformers spacy==3.2.0 beautifulsoup4 pandas nltk rouge-score evaluate

```

### 3. 下载 NLP 静态模型

```bash
# 安装适配 spacy 3.2.0 的模型
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.2.0/en_core_web_sm-3.2.0-py3-none-any.whl

# 下载 NLTK 数据
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"

```

---

## 📊 数据库下载与准备
本项目支持 Arxiv 论文、BBC 新闻和对话数据。

```bash
# 使用 curl (Windows 自带) 下载
curl -L https://github.com/liyucheng09/Selective_Context/releases/download/v0.1.0rc1/datasets_dumps.zip -o datasets_dumps.zip

```

---

## 🚀 运行实验

### 1. 配置你的压缩算法
所有的“魔改逻辑”都位于 `my_compression.py` 中。你可以修改其中的卷积核大小（Kernel Size）或步长（Stride）。

### 2. 执行评测
运行以下命令，程序将自动完成 **压缩 -> 生成答案 -> 自动评估** 的全流程：

```bash
python main.py \
  datasets_dumps/arxiv \
  datasets_dumps/news \
  datasets_dumps/conversation \
  ./new_result \
  5 \
  gpt2

```

---

## 🧠 大模型设置说明
本项目目前默认使用 **GPT-2** 作为本地推理引擎，因为它对 `inputs_embeds` 的支持非常稳定。

- **本地权重加载**：程序在首次运行时会自动从 Hugging Face 下载 GPT-2 权重。若网络受限，可设置环境变量：`export HF_ENDPOINT=https://hf-mirror.com`。
- **显存优化**：本项目在 `qa_manager.py` 中内置了截断防护逻辑，确保拼接后的向量长度不会超过 GPT-2 的 **1024** 物理限制，有效避免了 `IndexError`。

---

## 📜 引用
如果您在研究中使用了本项目的逻辑，请同时引用原论文：

```コード スニペット
@misc{li2023compressing,
title={Compressing Context to Enhance Inference Efficiency of Large Language Models},
author={Yucheng Li and Bo Dong and Chenghua Lin and Frank Guerin},
year={2023},
eprint={2310.06201},
archivePrefix={arXiv},
primaryClass={cs.CL}
}

```

---
**你想让我继续帮你细化 my_compression.py 中关于卷积数学公式的描述，还是想直接开始下一轮的性能调优实验？**

---

*Exported from [Voyager](https://github.com/Nagi-ovo/gemini-voyager)*  
*Generated on March 17, 2026 at 01:49 PM*