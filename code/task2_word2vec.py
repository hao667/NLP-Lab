# -*- coding: utf-8 -*-
"""Task 2: Word Embedding（Word2Vec / GloVe）文本分类。

使用 100 维词向量表示文档，分类器统一为 Logistic Regression。三组实验：
  1. Pre-trained GloVe（glove.6B 100d）
  2. Word2Vec trained on AG News（100d）
  3. Word2Vec trained on NYT（100d）

文档向量 = 文档内所有有效单词词向量的平均。每种方法在 NYT Test Set 上
报告 Accuracy 与 Macro-F1。
"""
import json
import os
import zipfile

import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data_utils import (
    get_splits,
    load_ag,
    tokenize_corpus,
    BASE_DIR,
)

DIM = 100
RESULTS_PATH = os.path.join(BASE_DIR, "results", "task2_word2vec.json")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
GLOVE_ZIP = os.path.join(BASE_DIR, "glove.6B.zip")
GLOVE_TXT = os.path.join(BASE_DIR, "glove.6B.100d.txt")


# --------------------------------------------------------------------------- #
# 词向量加载与训练
# --------------------------------------------------------------------------- #
def load_glove(path=GLOVE_TXT, dim=DIM):
    """加载 GloVe 词向量，返回 {word: np.ndarray}。"""
    if not os.path.exists(path):
        # 若尚未解压，从 zip 中提取 100d 文件
        with zipfile.ZipFile(GLOVE_ZIP) as z:
            member = f"glove.6B.{dim}d.txt"
            with z.open(member) as src, open(path, "wb") as dst:
                dst.write(src.read())

    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.asarray(parts[1:], dtype="float32")
            if vec.shape[0] == dim:
                embeddings[word] = vec
    return embeddings


def train_word2vec(sentences, model_path, dim=DIM):
    """训练 Word2Vec 模型（CBOW），若已存在则直接加载。"""
    if os.path.exists(model_path):
        return Word2Vec.load(model_path)

    model = Word2Vec(
        sentences=sentences,
        vector_size=dim,
        window=5,
        min_count=1,
        sg=0,
        workers=4,
        seed=42,
        epochs=5,
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    return model


# --------------------------------------------------------------------------- #
# 文档向量与分类
# --------------------------------------------------------------------------- #
def doc_vector(tokens, wv, dim=DIM):
    """对一篇文档的所有有效单词词向量取平均。"""
    vecs = [wv[w] for w in tokens if w in wv]
    if len(vecs) == 0:
        return np.zeros(dim, dtype="float32")
    return np.mean(vecs, axis=0)


def build_doc_vectors(texts, wv, dim=DIM):
    """将文本集合转换为文档向量矩阵 (n_samples, dim)。"""
    tokenized = tokenize_corpus(texts)
    X = np.vstack([doc_vector(tokens, wv, dim) for tokens in tokenized])
    return X


def evaluate(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    return acc, macro_f1


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def load_existing_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def run_experiment(name, wv, train_df, test_df, y_train, y_test):
    print(f"[{name}] ...")
    X_train = build_doc_vectors(train_df["text"], wv)
    X_test = build_doc_vectors(test_df["text"], wv)
    acc, f1 = evaluate(X_train, y_train, X_test, y_test)
    print(f"      Accuracy={acc:.4f}, Macro-F1={f1:.4f}")
    return {"accuracy": acc, "macro_f1": f1}


def main():
    # 可选参数：glove / ag / nyt / all（默认 all）
    import sys

    choices = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    if "all" in choices:
        choices = ["glove", "ag", "nyt"]

    train_df, val_df, test_df = get_splits()
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    results = load_existing_results()

    if "glove" in choices:
        glove = load_glove()
        results["Pre-trained GloVe"] = run_experiment(
            "Pre-trained GloVe", glove, train_df, test_df, y_train, y_test
        )

    if "ag" in choices:
        ag_sentences = tokenize_corpus(load_ag()["text"])
        w2v_ag = train_word2vec(
            ag_sentences, os.path.join(CACHE_DIR, "word2vec_ag.model")
        )
        results["Word2Vec on AG News"] = run_experiment(
            "Word2Vec on AG News", w2v_ag.wv, train_df, test_df, y_train, y_test
        )

    if "nyt" in choices:
        nyt_sentences = tokenize_corpus(train_df["text"])
        w2v_nyt = train_word2vec(
            nyt_sentences, os.path.join(CACHE_DIR, "word2vec_nyt.model")
        )
        results["Word2Vec on NYT"] = run_experiment(
            "Word2Vec on NYT", w2v_nyt.wv, train_df, test_df, y_train, y_test
        )

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
