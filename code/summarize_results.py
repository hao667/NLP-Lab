# -*- coding: utf-8 -*-
"""汇总所有任务的结果并打印对比表格。"""
import json
import os

from data_utils import BASE_DIR

RESULTS_DIR = os.path.join(BASE_DIR, "results")

TASK_FILES = [
    ("Task 1: BoW", "task1_bow.json"),
    ("Task 2: Word2Vec", "task2_word2vec.json"),
    ("Task 3: BERT", "task3_bert.json"),
]


def main():
    rows = []
    for task_name, fname in TASK_FILES:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"[skip] {fname} 不存在")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for method, metrics in data.items():
            rows.append((task_name, method, metrics["accuracy"], metrics["macro_f1"]))

    print(f"{'Task':<18}{'Method':<28}{'Accuracy':>10}{'Macro-F1':>10}")
    print("-" * 66)
    for task_name, method, acc, f1 in rows:
        print(f"{task_name:<18}{method:<28}{acc:>10.4f}{f1:>10.4f}")


if __name__ == "__main__":
    main()
