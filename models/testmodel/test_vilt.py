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

    # 이미지 경로 설정 및 존재하지 않는 이미지 제외
    def get_image_path(no):
        image_path = f"/your_dir/newdataset/newdata/{no}/{no}.jpg"
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

    # 모델의 num_labels와 classifier 재설정
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

            # One-hot encoding으로 labels 생성
            labels = torch.zeros((1, model.config.num_labels), device=device)  # [1, num_labels]
            labels[0, model.config.label2id[answer]] = 1.0  # 정답 인덱스에 1 할당

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{epochs} completed.")

# 성능 평가 및 결과 저장 함수
def evaluate_and_save_results(data, processor, model, epoch, output_dir="vilt_top3_eval_ep30", k=3, device='cpu'):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model.to(device)

    total = len(data)
    top1_correct = 0
    top3_correct = 0
    meteor_highest_scores = []
    rouge_l_highest_scores = []
    results = []

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for idx, row in data.iterrows():
        image = Image.open(row['image_path']).convert('RGB')
        question = row['question']
        reference_answer = preprocess_answer(row['answer'])
        question_id = row['QuestionId']

        # Top-3 답변 생성
        inputs = processor(image, question, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        top_k_indices = torch.topk(logits, k=k).indices.squeeze(0)
        predicted_answers = [model.config.id2label[idx.item()] for idx in top_k_indices]
        processed_predicted_answers = [preprocess_answer(ans) for ans in predicted_answers]

        # Top-1 정답 체크
        is_top1_correct = processed_predicted_answers[0] == reference_answer
        if is_top1_correct:
            top1_correct += 1

        # Top-3 정답 체크
        is_top3_correct = reference_answer in processed_predicted_answers
        if is_top3_correct:
            top3_correct += 1

        # METEOR 및 ROUGE-L 계산 (Top-3 중 가장 높은 점수)
        meteor_highest_scores.append(
            max(
                meteor_score([reference_answer.split()], pred.split())
                for pred in processed_predicted_answers
            )
        )
        rouge_l_highest_scores.append(
            max(
                scorer.score(reference_answer, pred)['rougeL'].fmeasure
                for pred in processed_predicted_answers
            )
        )

        # 결과 저장
        results.append({
            "QuestionId": question_id,
            "Question": question,
            "Actual Answer": row['answer'],
            "Processed Actual Answer": reference_answer,
            "Top-1 Answer": predicted_answers[0],
            "Top-2 Answer": predicted_answers[1] if len(predicted_answers) > 1 else None,
            "Top-3 Answer": predicted_answers[2] if len(predicted_answers) > 2 else None,
            "Processed Top-1 Answer": processed_predicted_answers[0],
            "Processed Top-2 Answer": processed_predicted_answers[1] if len(processed_predicted_answers) > 1 else None,
            "Processed Top-3 Answer": processed_predicted_answers[2] if len(processed_predicted_answers) > 2 else None,
            "Is Top-1 Correct": is_top1_correct,
            "Is Top-3 Correct": is_top3_correct
        })

    # 성능 계산
    accuracy_top1 = top1_correct / total * 100
    accuracy_top3 = top3_correct / total * 100
    meteor_score_highest_avg = sum(meteor_highest_scores) / len(meteor_highest_scores) * 100
    rouge_l_highest_avg = sum(rouge_l_highest_scores) / len(rouge_l_highest_scores) * 100

    # 결과 출력
    print(f"Epoch {epoch}: Top-1 Accuracy: {accuracy_top1:.2f}%")
    print(f"Epoch {epoch}: Top-3 Accuracy: {accuracy_top3:.2f}%")
    print(f"Epoch {epoch}: Highest METEOR Score (Top-3): {meteor_score_highest_avg:.2f}")
    print(f"Epoch {epoch}: Highest ROUGE-L Score (Top-3): {rouge_l_highest_avg:.2f}")

    # 결과 CSV로 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/epoch_{epoch}_results.csv", index=False, encoding="utf-8-sig")
    print(f"Epoch {epoch} results saved to {output_dir}/epoch_{epoch}_results.csv")

    return accuracy_top1, accuracy_top3, meteor_score_highest_avg, rouge_l_highest_avg

# 데이터 로드 및 분할
data = load_data("koreaheritageVQAdataset\korea_heritage_VQA_final.csv")
train_data, test_data = split_data(data)

# 모델 및 프로세서 로드
processor, model = load_model_with_labels(data)

# 학습 및 평가
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 30
for epoch in range(1, epochs + 1):
    # 모델 학습
    train_model(train_data, processor, model, epochs=1, learning_rate=5e-5, device=device)

    # 평가 및 저장
    evaluate_and_save_results(test_data, processor, model, epoch=epoch, output_dir="vilt_top3_eval_ep5", k=3, device=device)
