# -*- coding: utf-8 -*-
"""Task 1: Bag-of-Words 文本分类。

使用三种词袋表示 + Logistic Regression：
  1. Binary Bag of Words（只记录词是否出现）
  2. Word Frequency（记录词频）
  3. TF-IDF（词频-逆文档频率加权）

每种方法均在 NYT Test Set 上报告 Accuracy 与 Macro-F1。
"""
import json
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data_utils import get_splits, tokenize, BASE_DIR

RESULTS_PATH = os.path.join(BASE_DIR, "results", "task1_bow.json")


def evaluate(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    return acc, macro_f1


def main():
    train_df, val_df, test_df = get_splits()

    # 统一使用 nltk 分词作为 tokenizer（返回小写 token 列表）
    tokenizer = lambda text: tokenize(text)

    methods = {
        "Binary Bag of Words": CountVectorizer(
            tokenizer=tokenizer, binary=True, lowercase=False
        ),
        "Word Frequency": CountVectorizer(
            tokenizer=tokenizer, binary=False, lowercase=False
        ),
        "TF-IDF": TfidfVectorizer(tokenizer=tokenizer, lowercase=False),
    }

    results = {}
    for name, vectorizer in methods.items():
        X_train = vectorizer.fit_transform(train_df["text"])
        X_test = vectorizer.transform(test_df["text"])
        acc, macro_f1 = evaluate(
            X_train, train_df["label"], X_test, test_df["label"]
        )
        results[name] = {"accuracy": acc, "macro_f1": macro_f1}
        print(f"{name}: Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
