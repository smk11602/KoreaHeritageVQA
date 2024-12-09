import os
import pandas as pd
import torch
from transformers import AdamW
from PIL import Image
from sklearn.model_selection import train_test_split
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from torchvision import transforms
from fairseq.data.data_utils import post_process
import sys
sys.path.append("your_dir/OFA")
from tasks.ofa_task import OFATask
from tasks.ofa_task import OFATask
from fairseq import utils, checkpoint_utils
import re
import inflect

# 데이터 로드 및 전처리 함수
def load_data(filepath):
    data = pd.read_csv(filepath, dtype={'QuestionId': str})
    filtered_data = data[data['QuestionId'].str.startswith('1')].copy()
    filtered_data['image_path'] = filtered_data['No'].apply(
        lambda x: f"your_dir/newdataset/newdata/{x}/{x}.jpg"
    )
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

# OFA 모델 로드 함수
def load_ofa_model(model_path, use_cuda):
    # 모델과 태스크 로드
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        arg_overrides={"data": ""}
    )
    model = models[0]
    model.eval()
    if use_cuda:
        model.cuda()
    return model, task


# 이미지 전처리 함수
def preprocess_image(image_path, patch_image_size):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

# Top-K 답변 생성 함수
def generate_top_k_answers_ofa(task, model, image_tensor, question, k=3, use_cuda=False):
    sample = {
        "net_input": {
            "src_tokens": task.source_dictionary.encode_line(
                question, add_if_not_exist=False
            ).unsqueeze(0),
            "patch_images": image_tensor,
            "patch_masks": torch.tensor([True])
        }
    }
    if use_cuda:
        sample = utils.move_to_cuda(sample)
    
    with torch.no_grad():
        logits = model(**sample["net_input"])[0]
        top_k_logits = torch.topk(logits, k, dim=-1)[1]  # Top-K 답변 인덱스
        answers = [task.target_dictionary.string(answer).strip() for answer in top_k_logits]
    
    return [post_process(answer, 'sentencepiece') for answer in answers]

# 평가 및 저장 함수
def evaluate_and_save_results_ofa(data, task, model, epoch, output_dir="ofa_top3_eval", k=3, device="cpu"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    use_cuda = device == "cuda" and torch.cuda.is_available()

    total = len(data)
    top1_correct = 0
    top3_correct = 0
    meteor_highest_scores = []
    rouge_l_highest_scores = []
    results = []

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for idx, row in data.iterrows():
        image_tensor = preprocess_image(row['image_path'], patch_image_size=480)
        if use_cuda:
            image_tensor = image_tensor.cuda()

        question = row['question']
        reference_answer = preprocess_answer(row['answer'])
        question_id = row['QuestionId']

        # Top-K 답변 생성
        predicted_answers = generate_top_k_answers_ofa(task, model, image_tensor, question, k=k, use_cuda=use_cuda)
        processed_predicted_answers = [preprocess_answer(ans) for ans in predicted_answers]

        # Top-1 정답 체크
        is_top1_correct = processed_predicted_answers[0] == reference_answer
        if is_top1_correct:
            top1_correct += 1

        # Top-3 정답 체크
        is_top3_correct = reference_answer in processed_predicted_answers
        if is_top3_correct:
            top3_correct += 1

        # METEOR 및 ROUGE-L 계산
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

    print(f"Epoch {epoch}: Top-1 Accuracy: {accuracy_top1:.2f}%")
    print(f"Epoch {epoch}: Top-3 Accuracy: {accuracy_top3:.2f}%")
    print(f"Epoch {epoch}: Highest METEOR Score (Top-3): {meteor_score_highest_avg:.2f}")
    print(f"Epoch {epoch}: Highest ROUGE-L Score (Top-3): {rouge_l_highest_avg:.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/epoch_{epoch}_results.csv", index=False, encoding="utf-8-sig")
    return accuracy_top1, accuracy_top3, meteor_score_highest_avg, rouge_l_highest_avg

# 데이터 로드 및 분할
data = load_data("koreaheritageVQAdataset\korea_heritage_VQA_final.csv")
train_data, test_data = split_data(data)

# 모델 및 작업 로드
model_path = "your_dir/vqa_base_best.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, task = load_ofa_model(model_path, use_cuda=(device == "cuda"))

# 학습 및 평가 루프
epochs = 10
for epoch in range(1, epochs + 1):
    print(f"Starting evaluation for epoch {epoch}...")
    evaluate_and_save_results_ofa(
        data=test_data,
        task=task,
        model=model,
        epoch=epoch,
        output_dir="ofa_top3_eval",
        k=3,
        device=device
    )
