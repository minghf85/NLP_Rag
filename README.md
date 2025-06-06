# RAG基础复现
USTC自然语言处理课程

本项目基于ollama平台运行的embedding model:**bge-m3:large**和LLM:**qwen2.5:7b**对Rag进行基础功能原理的复现，未针对各个模块进行多余优化。

## 项目结构

- `main.py`: 主要实现脚本，在ollama平台运行embedding和llm实现知识库问答以及少量数据测试
- `testdata.txt`: 测试数据库，使用txt文本，包含中文和英文，从wikipedia等官方权威网站查找得到的信息，以换行为一个passage的分割。
- `unsupported_questions.txt`: 几个模型本身训练数据不包含的或者模型训练完成之后才出现的问题。
- `requirements.txt`: 项目依赖

## RAG模型简介

RAG结合了检索（Retrieval）和生成（Generation）两个过程:

1. **检索组件**: 使用双编码器检索系统（如DPR）检索与输入查询相关的文档，本项目复现使用embedding向量余弦相似度检索。
2. **生成组件**: 将检索到的文档与原始查询一起输入到序列到序列模型（如BART）中生成最终答案，本项目使用LLM自回归生产模型qwen2.5:7b生成。

RAG模型的优势:
- 通过检索外部知识增强生成能力
- 提高生成内容的事实准确性
- 使模型能够访问最新信息（如果检索语料库更新）

## 使用方法

1. 安装依赖:
```
pip install -r requirements.txt
```

2. 运行主脚本:
```
python main.py
```
可选择注释，切换问答和测试

## RAG的工作流程

1. 对知识库切分，向量化，存储
2. 对输入问题进行编码，获得问题的向量表示
3. 使用问题向量从文档库中检索相关文档
4. 将问题和检索到的文档一起输入生成模型
5. 生成模型产生最终答案

## 参考资料

- [RAG论文](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)
- [Hugging Face Transformers文档](https://huggingface.co/docs/transformers/model_doc/rag)
