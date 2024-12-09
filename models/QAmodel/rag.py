import json
import pandas as pd
import logging
import re
import inflect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import torch

# 로그 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# JSON 파일에서 데이터셋 로드
def load_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# FLAN-T5 모델과 토크나이저 로드
t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")  # FLAN-T5 Base 사용
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# 요약 모델 로드 (Pegasus-XSum)
summarizer = pipeline("summarization", model="google/pegasus-xsum")

# 컨텍스트 요약 함수
def summarize_context(context, max_length=512):
    summary = summarizer(context, max_length=max_length, min_length=30, truncation=True)
    return summary[0]['summary_text']

# 긴 컨텍스트를 자르기
def truncate_context(context, max_length=512):
    return context[:max_length]  # 지정된 길이로 자름

# 답변 전처리 함수
def preprocess_answer(answer):
    answer = answer.lower()
    if answer == "rectangle":
        answer = "rectangular"
    p = inflect.engine()
    answer = ' '.join([p.number_to_words(word) if word.isdigit() else word for word in answer.split()])
    answer = re.sub(r'(?<!\d)\.(?!\d)', '', answer)  # 숫자가 아닌 점 제거
    answer = re.sub(r'\b(a|an|the)\b', '', answer)   # 관사 제거
    answer = re.sub(r"[^\w\s]", '', answer)         # 특수문자 제거
    answer = ' '.join(answer.split())               # 다중 공백 제거
    return answer

# 정확한 `context`에서 답변 생성
def generate_answer_from_exact_context(question, context):
    input_text = f"Answer the question based on the following context:\n\nContext:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    input_ids = t5_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids
    with torch.no_grad():
        outputs = t5_model.generate(input_ids, max_new_tokens=100)  # 더 긴 답변을 허용
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

# RAG 파이프라인: 요약과 비요약 모두 수행
def rag_pipeline(question, question_id, qa_dataset, use_summary=False):
    # 동일한 `question_id`를 가진 항목 찾기
    exact_match = next(item for item in qa_dataset if item['question_id'] == question_id)
    context = exact_match["context"]

    # 컨텍스트 요약 적용 여부
    if use_summary:
        context = summarize_context(context)

    # 답변 생성
    generated_answer = generate_answer_from_exact_context(question, context)

    return {
        "question": question,
        "retrieved_context": context,
        "ground_truth": exact_match["answers"][0],  # 실제 정답
        "generated_answer": generated_answer  # 생성된 답변
    }

# 평가 지표 계산
def calculate_metrics(generated_answer, ground_truth):
    # 정답 전처리
    generated_answer = preprocess_answer(generated_answer)
    ground_truth = preprocess_answer(ground_truth)

    # Accuracy
    accuracy = 1 if generated_answer == ground_truth else 0
    
    # METEOR
    meteor = meteor_score([ground_truth.split()], generated_answer.split())
    
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(ground_truth, generated_answer)['rougeL'].fmeasure
    
    return accuracy, meteor, rouge_l

# 전체 데이터셋에 대해 평가
def evaluate(qa_dataset, use_summary=False):
    results = []
    total_data = len(qa_dataset)
    summary_type = "with_summary" if use_summary else "without_summary"
    logging.info(f"Evaluating {total_data} questions ({summary_type}).")

    for idx, item in enumerate(qa_dataset):
        question = item["question"]
        question_id = item["question_id"]
        ground_truth = item["answers"][0]
        
        # RAG 파이프라인 실행
        result = rag_pipeline(question, question_id, qa_dataset, use_summary)
        
        # 평가지표 계산
        accuracy, meteor, rouge_l = calculate_metrics(result['generated_answer'], result['ground_truth'])

        # 결과 저장
        results.append({
            "question_id": question_id,
            "question": result['question'],
            "retrieved_context": result['retrieved_context'],
            "ground_truth": result['ground_truth'],
            "generated_answer": result['generated_answer'],
            "accuracy": accuracy,
            "meteor": meteor,
            "rouge_l": rouge_l
        })

        # 진행 상황 로그 기록
        if (idx + 1) % 100 == 0:  # 100개마다 로그 기록
            logging.info(f"Processed {idx + 1}/{total_data} samples ({summary_type}).")

    return results

# 전체 평가 점수 계산
def calculate_overall_metrics(results):
    df = pd.DataFrame(results)
    overall_accuracy = df["accuracy"].mean()
    overall_meteor = df["meteor"].mean()
    overall_rouge_l = df["rouge_l"].mean()
    return overall_accuracy, overall_meteor, overall_rouge_l

# 실행 및 결과 저장
def main():
    # 데이터셋 로드
    qa_dataset = load_data("koreaheritageVQAdataset\korea_heritage_VQA_finetune.json")

    # 요약 없는 평가
    results_no_summary = evaluate(qa_dataset, use_summary=False)
    df_no_summary = pd.DataFrame(results_no_summary)
    df_no_summary.to_csv('rag_eval_no_summary.csv', index=False)
    overall_accuracy, overall_meteor, overall_rouge_l = calculate_overall_metrics(results_no_summary)
    logging.info(f"Results without summary - Accuracy: {overall_accuracy}, METEOR: {overall_meteor}, ROUGE-L: {overall_rouge_l}")

    # 요약 있는 평가
    results_with_summary = evaluate(qa_dataset, use_summary=True)
    df_with_summary = pd.DataFrame(results_with_summary)
    df_with_summary.to_csv('rag_eval_with_summary.csv', index=False)
    overall_accuracy, overall_meteor, overall_rouge_l = calculate_overall_metrics(results_with_summary)
    logging.info(f"Results with summary - Accuracy: {overall_accuracy}, METEOR: {overall_meteor}, ROUGE-L: {overall_rouge_l}")

if __name__ == "__main__":
    main()
