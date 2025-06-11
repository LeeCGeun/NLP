import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from itertools import permutations
from sklearn.model_selection import train_test_split

# 1. 순열 매핑
perm_map = list(permutations([0, 1, 2, 3]))
perm2id = {p: i for i, p in enumerate(perm_map)}
id2perm = {i: p for i, p in enumerate(perm_map)}

# 2. 데이터 불러오기 및 전처리
train_df = pd.read_csv("./train.csv")
def row_to_input_label(row):
    sentences = [row[f"sentence_{i}"] for i in range(4)]
    input_text = " [SEP] ".join(sentences)
    answer = tuple([row[f"answer_{i}"] for i in range(4)])
    label = perm2id[answer]
    return {"input": input_text, "label": label}

inputs = train_df.apply(row_to_input_label, axis=1).tolist()
train_data, valid_data = train_test_split(inputs, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
valid_dataset = Dataset.from_pandas(pd.DataFrame(valid_data))

# 3. 모델 및 토크나이저 로드
model_name = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=24)

# 4. 토크나이징 함수 정의
def tokenize(batch):
    return tokenizer(batch["input"], padding="max_length", truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize, batched=True)
tokenized_valid = valid_dataset.map(tokenize, batched=True)

# 5. 학습 설정
training_args = TrainingArguments(
    output_dir="./bert_24cls_30epoch_results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none"
)

# 6. Trainer 정의 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained("./bert_24cls_30epoch_results")
tokenizer.save_pretrained("./bert_24cls_30epoch_results")

# 7. Inference (batch 처리로 OOM 방지)
test = pd.read_csv("./test.csv")
sentences = test[[f"sentence_{i}" for i in range(4)]].values.tolist()
inputs = [" [SEP] ".join(s) for s in sentences]

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 32
pred_ids = []
with torch.no_grad():
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        batch_enc = tokenizer(batch_inputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        batch_enc = {k: v.to(device) for k, v in batch_enc.items()}
        output = model(**batch_enc)
        batch_preds = output.logits.argmax(dim=1).cpu().tolist()
        pred_ids.extend(batch_preds)

pred_orders = [id2perm[i] for i in pred_ids]

submission = pd.read_csv("./sample_submission.csv")
for i in range(4):
    submission[f"answer_{i}"] = [pred[i] for pred in pred_orders]

submission.to_csv("bert_24cls_30epoch_results.csv", index=False)
print("✅ Submission saved to 'bert_24cls_30epoch_results.csv'")
