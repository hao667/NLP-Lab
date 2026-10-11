# 作业一：文本分类实现

本仓库实现 NYT 新闻文本分类任务，分别使用词袋模型（BoW）、词向量（Word2Vec/GloVe）与
预训练语言模型（BERT）三种文本表示方法，并在 NYT 测试集上报告 Accuracy 与 Macro-F1。

## 目录结构

```
lab1/
├── HW-1/                 # 数据集（nyt.csv, ag.csv）
├── code/
│   ├── data_utils.py         # 数据加载、划分（80/10/10）、分词工具
│   ├── task1_bow.py          # Task 1: Bag-of-Words + Logistic Regression
│   ├── task2_word2vec.py     # Task 2: GloVe / Word2Vec 词向量 + Logistic Regression
│   ├── task3_bert.py         # Task 3: BERT Fine-tuning
│   ├── download_bert.py      # 从镜像下载 BERT 模型（网络受限时使用）
│   └── summarize_results.py  # 汇总并打印所有结果
├── cache/                # 缓存（数据划分、训练好的 Word2Vec 模型）
├── results/              # 各任务结果（json）
├── bert-base-uncased/    # 本地 BERT 模型
└── glove.6B.100d.txt     # GloVe 100 维词向量
```

## 环境依赖

- Python 3.11
- 主要依赖：`numpy`、`pandas`、`scikit-learn`、`nltk`、`gensim`、`torch`、`transformers`

安装命令：

```bash
pip install numpy pandas scikit-learn nltk gensim torch transformers
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## 数据与预训练资源准备

1. 数据集已置于 `HW-1/`：`nyt.csv`（text, label）、`ag.csv`（text）。
2. GloVe：将 `glove.6B.zip` 解压得到的 `glove.6B.100d.txt` 放到项目根目录，
   或直接运行 `task2_word2vec.py` 时会自动从 `glove.6B.zip` 中提取。
3. BERT：默认从 HuggingFace 下载 `google-bert/bert-base-uncased`；
   若网络受限，运行 `python code/download_bert.py` 从镜像下载到 `bert-base-uncased/`，
   `task3_bert.py` 会自动优先使用该本地模型。

## 运行方式

所有脚本均在 `code/` 目录下运行：

```bash
cd code

# Task 1: 三种 BoW + Logistic Regression
python task1_bow.py

# Task 2: 词向量 + Logistic Regression（三组实验）
python task2_word2vec.py          # 全部
python task2_word2vec.py ag nyt   # 仅 Word2Vec（AG/NYT），跳过 GloVe

# Task 3: BERT Fine-tuning（max_length=64, 3 epochs）
python task3_bert.py

# 汇总所有结果
python summarize_results.py
```

## 实验设置

- 数据划分：对 NYT 随机打乱后按 80% / 10% / 10% 划分训练/验证/测试集，
  使用固定随机种子（`random_state=42`）并缓存到 `cache/nyt_splits.pkl`，所有实验复用同一划分。
- 评价指标：Accuracy 与 Macro-F1，统一在 NYT 测试集上报告。
- Task 1：`Binary Bag of Words`、`Word Frequency`、`TF-IDF` 三种词袋表示，
  分类器统一为 `LogisticRegression`。
- Task 2：统一使用 100 维词向量，文档向量为词向量的平均，分类器为 `LogisticRegression`。
  - `Pre-trained GloVe`：glove.6B 100d
  - `Word2Vec on AG News`：用 AG News 训练 Word2Vec（CBOW, window=5, 100d）
  - `Word2Vec on NYT`：用 NYT 训练 Word2Vec（CBOW, window=5, 100d）
- Task 3：`google-bert/bert-base-uncased`，`max_length=64`，`epochs=3`。

## 结果

| Task | 方法 | Accuracy | Macro-F1 |
|------|------|----------|----------|
| Task 1 | Binary Bag of Words | 0.9887 | 0.9728 |
| Task 1 | Word Frequency | 0.9913 | 0.9779 |
| Task 1 | TF-IDF | 0.9896 | 0.9751 |
| Task 2 | Pre-trained GloVe | 0.9852 | 0.9645 |
| Task 2 | Word2Vec on AG News | 0.9783 | 0.9469 |
| Task 2 | Word2Vec on NYT | 0.9792 | 0.9473 |
| Task 3 | BERT | — | — |

> Task 3（BERT Fine-tuning）代码已完整实现（见 `code/task3_bert.py`），
> 但在纯 CPU 环境下微调 bert-base-uncased（3 个 epoch）耗时较长，此处未完成训练。
> 在有 GPU 或更长时间的环境下运行 `python code/task3_bert.py` 即可得到结果。
