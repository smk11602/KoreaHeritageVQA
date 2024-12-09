import os
import pandas as pd
import openai
import logging

# OpenAI API 키 설정
openai.api_key = "YOUR API key"

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CSV 파일 불러오기
df = pd.read_csv('koreaheritageVQAdataset\korea_heritage_VQA_R.csv', dtype={'QuestionId': str})

# QuestionId가 '0'으로 시작하는 데이터만 선택
contextual_questions = df[df['QuestionId'].str.startswith('0')]

# 필터링된 데이터프레임이 비어 있는지 확인
if contextual_questions.empty:
    logging.info("The filtered contextual_questions dataframe is empty.")
else:
    logging.info("Filtered dataframe is not empty. Proceeding with the first entry.")

    # 첫 번째 질문 선택
    sample_question = contextual_questions.iloc[0]

    # 평가 기준 설정 및 평가 함수 정의
    def evaluate_question(question_id, question, answer, heritage_name):
        prompt = (
            f"You are an assistant assigned to carefully evaluate automatically generated questions for the KoreaHeritageQA project. "
            f"These questions may be used as educational material, so please assess them thoughtfully and assign accurate scores.\n\n"
            f"Evaluate the following contextual question based on the cultural heritage item '{heritage_name}'.\n\n"
            f"Question: {question}\nAnswer: {answer}\n\n"
            "Rate this question on the following criteria from 1 to 100:\n"
            "1. Specificity - How specific is the question to this cultural heritage item? If this question can be used for other heritage, it is not quite specific\n"
            "2. Relevance - How relevant is the question in understanding this heritage's history or context?\n"
            "3. Conciseness - Is the question concise and to the point? Also consider the length of the answer. If it is more than 3 words, it might be hard to answer.\n\n"
            "Please provide scores as 'Specificity: X, Relevance: X, Conciseness: X'. Don't give explanation, just score\n\n"
        )

        # OpenAI API 호출 (chat completion 사용)
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0
        )

        # 응답에서 점수 추출
        evaluation_text = response.choices[0].message['content'].strip()
        logging.info(f"GPT-4 Response for QuestionID {question_id}:\n{evaluation_text}\n")

        # 점수 초기화
        specificity = relevance = conciseness = None

        # 응답 처리: 각 항목에서 'Specificity', 'Relevance', 'Conciseness' 값을 파싱
        for item in evaluation_text.split(","):
            item = item.strip()
            if "Specificity:" in item:
                specificity = int(item.split("Specificity:")[1].strip())
            elif "Relevance:" in item:
                relevance = int(item.split("Relevance:")[1].strip())
            elif "Conciseness:" in item:
                conciseness = int(item.split("Conciseness:")[1].strip())

        # 평균 점수 계산
        avg_score = round((specificity + relevance + conciseness) / 3, 2)

        # 로그 출력
        logging.info(f"Scores for QuestionID {question_id} - Specificity: {specificity}, Relevance: {relevance}, Conciseness: {conciseness}, Average: {avg_score}\n")

        return specificity, relevance, conciseness, avg_score

    # 평가 수행 및 데이터 저장
    results = []
    for _, row in contextual_questions.iterrows():
        specificity, relevance, conciseness, avg_score = evaluate_question(
            row['QuestionId'],
            row['question'],
            row['answer'],
            row['Name of Cultural Heritage']
        )
        results.append([row['QuestionId'], row['Name of Cultural Heritage'], row['question'], row['answer'], avg_score])

    # 새 데이터프레임 생성 및 CSV 저장
    final_df = pd.DataFrame(results, columns=['QuestionId', 'Name of Cultural Heritage', 'Question', 'Answer', 'AvgScore'])
    final_df.to_csv('koreaheritageVQAdataset\evaluated_contextual_questions.csv', index=False)