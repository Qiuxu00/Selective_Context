# 解决 Python 项目环境配置

**Date**: March 17, 2026 at 01:51 PM
**Turns**: 1
**Source**: [Gemini Chat](https://gemini.google.com/app/98431610d951200a?hl=ja)

---

## Turn 1

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
*Generated on March 17, 2026 at 01:51 PM*