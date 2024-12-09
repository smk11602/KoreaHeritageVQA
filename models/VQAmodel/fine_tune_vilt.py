import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import ViltProcessor, ViltForQuestionAnswering
from sklearn.model_selection import train_test_split
from PIL import Image
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import re
import inflect

# 데이터 로드 함수
def load_data(filepath):
    data = pd.read_csv(filepath, dtype={'QuestionId': str})
    filtered_data = data[data['QuestionId'].str.startswith('1')].copy()

    def get_image_path(no):
        image_path = f/"your_dir/newdataset/newdata/{no}/{no}.jpg"
        return image_path if os.path.exists(image_path) else None

    filtered_data['image_path'] = filtered_data['No'].apply(get_image_path)
    filtered_data = filtered_data.dropna(subset=['image_path'])
    return filtered_data

# 답변 전처리 함수
def preprocess_answer(answer):
    answer = answer.lower()
    if answer == "rectangle":
        answer = "rectangular"
    p = inflect.engine()
    answer = ' '.join([p.number_to_words(word) if word.isdigit() else word for word in answer.split()])
    answer = re.sub(r'(?<!\d)\.(?!\d)', '', answer)
    answer = re.sub(r'\b(a|an|the)\b', '', answer)
    answer = re.sub(r"[^\w\s]", '', answer)
    answer = ' '.join(answer.split())
    return answer

# Train/Test Split 함수
def split_data(data, test_size=0.2, random_state=42):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data

# 모델 로드 및 label 확장
def load_model_with_labels(data):
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm")

    unique_answers = data['answer'].apply(preprocess_answer).unique()
    label2id = {label: idx for idx, label in enumerate(unique_answers)}
    id2label = {idx: label for label, idx in label2id.items()}

    model.config.label2id = label2id
    model.config.id2label = id2label

    num_labels = len(unique_answers)
    model.config.num_labels = num_labels
    model.classifier = nn.Linear(model.config.hidden_size, num_labels)

    print("Updated label2id and id2label with all unique answers.")
    print(f"Total unique answers (num_labels): {num_labels}")
    return processor, model

# 모델 학습 함수
def train_model(train_data, processor, model, epochs=1, learning_rate=5e-5, device='cpu'):
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for idx, row in train_data.iterrows():
            image = Image.open(row['image_path']).convert('RGB')
            question = row['question']
            answer = preprocess_answer(row['answer'])

            inputs = processor(image, question, return_tensors="pt").to(device)

            labels = torch.zeros((1, model.config.num_labels), device=device)
            labels[0, model.config.label2id[answer]] = 1.0

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{epochs} completed.")

# 성능 평가 및 결과 저장 함수
def evaluate_and_save_results(data, processor, model, epoch, output_dir="vilt_top3_eval", k=3, device='cpu'):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model.to(device)

    total = len(data)
    top1_correct = 0

    for idx, row in data.iterrows():
        image = Image.open(row['image_path']).convert('RGB')
        question = row['question']
        reference_answer = preprocess_answer(row['answer'])

        inputs = processor(image, question, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        top_k_indices = torch.topk(logits, k=k).indices.squeeze(0)
        predicted_answers = [model.config.id2label[idx.item()] for idx in top_k_indices]
        processed_predicted_answers = [preprocess_answer(ans) for ans in predicted_answers]

        if processed_predicted_answers[0] == reference_answer:
            top1_correct += 1

    accuracy_top1 = top1_correct / total * 100
    print(f"Epoch {epoch}: Top-1 Accuracy: {accuracy_top1:.2f}%")

    return accuracy_top1

# 데이터 로드 및 분할
data = load_data("koreaheritageVQAdataset\korea_heritage_VQA_final.csv")
train_data, test_data = split_data(data)

# 모델 및 프로세서 로드
processor, model = load_model_with_labels(data)

# 학습 및 평가
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 30
best_accuracy = 0.0
best_model_dir = "vilt_best_model"

for epoch in range(1, epochs + 1):
    # 모델 학습
    train_model(train_data, processor, model, epochs=1, learning_rate=5e-5, device=device)

    # 평가
    accuracy_top1 = evaluate_and_save_results(test_data, processor, model, epoch=epoch, output_dir="vilt_top3_eval", k=3, device=device)

    # Best 모델 저장
    if accuracy_top1 > best_accuracy:
        best_accuracy = accuracy_top1
        model.save_pretrained(best_model_dir)
        processor.save_pretrained(best_model_dir)
        print(f"New best model saved with Top-1 Accuracy: {best_accuracy:.2f}% at epoch {epoch}.")
