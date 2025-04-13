import json
from datasets import Dataset
from pathlib import Path
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

#Carregar o json com os inputs de treinamento
path = Path("treino_ia.json")
with open(path, "rb") as arq:
    leitura = arq.read()

#Transformar o json em um dataset
json_de_treino = json.loads(leitura)
dataset = Dataset.from_list(json_de_treino)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize(example):
    return tokenizer(example["input"], truncation=True, padding="max_length", max_length=128)
tokenized_dataset = dataset.map(tokenize)

label2id = {label: idx for idx, label in enumerate(set(dataset["label"]))}
id2label = {v: k for k, v in label2id.items()}

def encode_label(example):
    example["label"] = label2id[example["label"]]
    return example

tokenized_dataset = tokenized_dataset.map(encode_label)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./resultados",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # se quiser, pode separar
)

trainer.train()

entrada = "Quero uma personagem feminina"

tokens = tokenizer(entrada, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
output = model(**tokens)
pred_id = output.logits.argmax(dim=1).item()
print(f"Recomendação: {id2label[pred_id]}")