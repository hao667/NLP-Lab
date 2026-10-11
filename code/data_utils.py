# -*- coding: utf-8 -*-
"""数据加载、划分与文本预处理工具。

本模块负责：
1. 读取 NYT / AG News 数据集；
2. 对 NYT 进行随机打乱并按 80% / 10% / 10% 划分为训练/验证/测试集；
3. 提供统一的文本分词函数。

为保证不同方法之间可比，划分结果使用固定随机种子并缓存到磁盘，
所有实验均复用同一份划分。
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import word_tokenize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW_DIR = os.path.join(BASE_DIR, "HW-1")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
NYT_PATH = os.path.join(HW_DIR, "nyt.csv")
AG_PATH = os.path.join(HW_DIR, "ag.csv")
SPLIT_PATH = os.path.join(CACHE_DIR, "nyt_splits.pkl")

RANDOM_STATE = 42


def load_nyt():
    """加载 NYT 数据集（包含 text 与 label 两列）。"""
    return pd.read_csv(NYT_PATH)


def load_ag():
    """加载 AG News 数据集（仅包含 text 一列）。"""
    return pd.read_csv(AG_PATH)


def get_splits():
    """获取 NYT 的训练/验证/测试划分（固定随机种子，缓存复用）。

    返回 (train_df, val_df, test_df)，三者的并集为完整 NYT 数据。
    """
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH, "rb") as f:
            train_df, val_df, test_df = pickle.load(f)
        return train_df, val_df, test_df

    df = load_nyt()
    # 先按 80/20 划分训练集与剩余部分，剩余部分再均分为验证/测试
    train_df, rest_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        rest_df, test_size=0.5, random_state=RANDOM_STATE, stratify=rest_df["label"]
    )
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(SPLIT_PATH, "wb") as f:
        pickle.dump((train_df, val_df, test_df), f)
    return train_df, val_df, test_df


def tokenize(text):
    """对单条文本进行分词并统一转为小写，返回 token 列表。"""
    return [w.lower() for w in word_tokenize(str(text))]


def tokenize_corpus(texts):
    """对文本序列进行分词，返回 token 列表的列表。"""
    return [tokenize(t) for t in texts]
