# -*- coding: utf-8 -*-
"""从 hf-mirror.com 镜像下载 BERT-base-uncased 到本地目录。

用于网络无法直接访问 huggingface.co 的环境。运行后会在项目根目录生成
`bert-base-uncased/` 目录，task3_bert.py 会优先使用该本地模型。
"""
import os
import urllib.request

from data_utils import BASE_DIR

MIRROR = "https://hf-mirror.com/google-bert/bert-base-uncased/resolve/main"
LOCAL_DIR = os.path.join(BASE_DIR, "bert-base-uncased")

FILES = [
    "config.json",
    "model.safetensors",
    "vocab.txt",
    "tokenizer.json",
    "tokenizer_config.json",
]


def download(url, dst, retries=3):
    for i in range(retries):
        try:
            print(f"  downloading {os.path.basename(dst)} ...")
            urllib.request.urlretrieve(url, dst)
            print(f"  ok ({os.path.getsize(dst)} bytes)")
            return
        except Exception as e:
            print(f"  attempt {i + 1} failed: {e}")
            if i == retries - 1:
                raise


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    for name in FILES:
        dst = os.path.join(LOCAL_DIR, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"skip {name} (already exists)")
            continue
        download(f"{MIRROR}/{name}", dst)
    print(f"Done. Model saved to {LOCAL_DIR}")


if __name__ == "__main__":
    main()
