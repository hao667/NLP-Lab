# -*- coding: utf-8 -*-
"""Task 3: 预训练语言模型（BERT）文本分类 Fine-tuning。

模型：google-bert/bert-base-uncased
配置：max_length=64，epochs=3

使用原生 PyTorch 训练循环（不依赖 datasets/accelerate）。
在 NYT Test Set 上报告 Accuracy 与 Macro-F1。
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score

from data_utils import get_splits, BASE_DIR

MODEL_NAME = "google-bert/bert-base-uncased"
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "bert-base-uncased")
MAX_LENGTH = 64
EPOCHS = 3
SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = os.path.join(BASE_DIR, "results", "bert")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "task3_bert.json")
CKPT_PATH = os.path.join(BASE_DIR, "results", "bert_ckpt.pt")


def build_label_mapping(labels):
    labels = sorted(set(labels))
    return {lbl: i for i, lbl in enumerate(labels)}, {i: lbl for i, lbl in enumerate(labels)}


def encode_texts(tokenizer, texts, labels, label2id):
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    lbls = torch.tensor([label2id[l] for l in labels], dtype=torch.long)
    return enc["input_ids"], enc["attention_mask"], lbls


def main():
    set_seed(SEED)
    train_df, val_df, test_df = get_splits()

    label2id, id2label = build_label_mapping(train_df["label"])
    num_labels = len(label2id)

    model_path = LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels
    )
    model.to(DEVICE)

    train_ids, train_mask, train_labels = encode_texts(
        tokenizer, train_df["text"], train_df["label"], label2id
    )
    test_ids, test_mask, test_labels = encode_texts(
        tokenizer, test_df["text"], test_df["label"], label2id
    )

    train_dataset = TensorDataset(train_ids, train_mask, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # 断点续训：若存在检查点则从上次进度恢复
    start_epoch = 0
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if (step + 1) % 50 == 0:
                print(
                    f"  epoch {epoch + 1}/{EPOCHS} step {step + 1}/{len(train_loader)} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )
        print(
            f"epoch {epoch + 1}/{EPOCHS} avg_loss={total_loss / len(train_loader):.4f}",
            flush=True,
        )
        # 每个 epoch 结束后保存检查点
        os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            },
            CKPT_PATH,
        )

    # 测试集评估
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_ids), BATCH_SIZE):
            input_ids = test_ids[i : i + BATCH_SIZE].to(DEVICE)
            attention_mask = test_mask[i : i + BATCH_SIZE].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            all_preds.append(torch.argmax(logits, dim=-1).cpu())
    preds = torch.cat(all_preds).numpy()

    acc = accuracy_score(test_labels.numpy(), preds)
    macro_f1 = f1_score(test_labels.numpy(), preds, average="macro")

    result = {"accuracy": acc, "macro_f1": macro_f1}
    print(f"BERT Test: Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {RESULTS_PATH}")

    # 清理检查点
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)


if __name__ == "__main__":
    main()
