import json
from datasets import Dataset
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
import numpy as np

path_treino = Path("docs/treino_ia_refatorado.json")
with open(path_treino, "r", encoding="utf-8") as f:
    treino_data = json.load(f)
treino_dataset = Dataset.from_list(treino_data)

# === Carregar dados de teste (original) ===
path_teste = Path("docs/treino_ia.json")
with open(path_teste, "r", encoding="utf-8") as f:
    teste_data = json.load(f)
teste_dataset = Dataset.from_list(teste_data)

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

def tokenize(example):
    return tokenizer(example["input"], truncation=True, padding=True, max_length=128)

tokenized_treino = treino_dataset.map(tokenize)
tokenized_teste = teste_dataset.map(tokenize)

todas_labels = sorted(set(example["label"] for example in treino_data + teste_data))
label2id = {label: idx for idx, label in enumerate(todas_labels)}
id2label = {v: k for k, v in label2id.items()}

def encode_label(example):
    example["label"] = label2id[example["label"]]
    return example

tokenized_treino = tokenized_treino.map(encode_label)
tokenized_teste = tokenized_teste.map(encode_label)

model = AutoModelForSequenceClassification.from_pretrained(
    "neuralmind/bert-base-portuguese-cased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def treinar_modelo():
    training_args = TrainingArguments(
        output_dir="./resultados",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=25,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_dir="./logs",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_treino,
        eval_dataset=tokenized_teste,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(" Avaliação final no dataset original:")
    print(metrics)

    model.save_pretrained("./modelo_final")
    tokenizer.save_pretrained("./modelo_final")

def resposta(entrada: str):
    print(entrada)
    model.eval()
    tokens = tokenizer(entrada, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        output = model(**tokens)

    scores = output.logits.softmax(dim=1).squeeze().cpu().numpy()
    pred_id = np.argmax(scores)

    print("🔍 Scores por personagem:")
    for idx, score in enumerate(scores):
        print(f"{id2label[idx]}: {score:.4f}")

    print(f"➡️ Previsão final: {id2label[pred_id]}")
    return id2label[pred_id]