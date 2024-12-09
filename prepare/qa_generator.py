import os
import pandas as pd
import openai
import logging

# OpenAI API 키 설정
openai.api_key = "Your API key"


# 로그 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 데이터 로드
processing_data_path = "koreaheritageVQAdataset\processing_data.csv"
processing_data = pd.read_csv(processing_data_path)

# 결과를 저장할 리스트
results = []

# 특정 문화유산에 대해 테스트
for index, row in processing_data.iterrows():
    heritage_no = row['No']
    name = row['Name of Cultural Heritage'].split(',')[0].strip()
    korean_name = row['Korean']

    # 설명 파일 읽기
    description_path = os.path.join("newdataset", "newdata", str(heritage_no), f"{heritage_no}_description.txt")
    
    try:
        with open(description_path, 'r', encoding='utf-8') as f:
            description = f.read()
    except Exception as e:
        logging.error(f"설명 파일 로드 실패: {e}")
        continue  # 파일 로드 실패 시 다음 문화유산으로 넘어가기

    # GPT 프롬프트 생성
    prompt = (
        f"You are an assistant that generates questions based on visual observations of cultural heritage items. "
        f"Using the following description of '{name}', generate 3 simple visual questions that can be answered by observing the image (e.g., about color, shape, etc.). "
        "For visual questions, do not mention the heritage's name; instead, refer to it as 'this heritage.' "
        "Generate 3 contextual questions that require knowledge of the heritage, which cannot be answered from the image but from the description provided. "
        "For contextual questions, read the entire description and ensure that your answers are concise and meaningful. All answers must be strictly limited to 1-2 words.\n\nDescription:\n{description}\n"
        "Here is a reference format for the questions:\n"
        "n. Visual Question: What color is this heritage? | Answer: White\n"
        "n. Visual Question: What is the shape of the roof of this heritage? | Answer: Dome\n"
        "n. Contextual Question: Who composed the inscription on the stele? | Answer: Choe Chi-won\n"
        "n. Contextual Question: When was the Godalsa Temple built? | Answer: 600 AD\n"
        "Please provide the questions and answers in the following format:\n"
        "1. Visual Question: ... | Answer: ...\n"
        "2. Visual Question: ... | Answer: ...\n"
        "3. Visual Question: ... | Answer: ...\n"
        "4. Contextual Question: ... | Answer: ...\n"
        "5. Contextual Question: ... | Answer: ...\n"
        "6. Contextual Question: ... | Answer: ..."
    )

    visual_count = 0
    contextual_count = 0

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )

        # 응답 출력
        generated_text = response.choices[0].message["content"].strip()
        logging.info(f"GPT 응답: {generated_text} \n")  # 응답 내용 출력

        # 응답 처리
        for line in generated_text.split('\n'):
            # '|'를 기준으로 질문과 답변을 나누기
            if '| Answer: ' in line:  # 기존의 파싱 방식
                question, answer = line.split('| Answer: ')
                question_id = f"{'0' if 'Contextual' in question else '1'}{heritage_no:03}{len(results):02}"  # 0: Contextual, 1: Visual
                results.append([question_id, row['Classification'], name, korean_name, question.strip(), answer.strip()])
                
                # 질문 유형에 따라 카운트 증가
                if 'Visual Question' in question:  # Visual questions
                    visual_count += 1
                else:  # Contextual questions
                    contextual_count += 1

        # 추가적인 조건으로 질문 수 확인
        if visual_count != 3 or contextual_count != 3:
            logging.warning(f"Heritage No {heritage_no} has {visual_count} visual and {contextual_count} contextual questions. Expected 3 each.")

    except Exception as e:
        logging.error(f"질문 생성 실패: {e}")

    # 결과 출력
    logging.info(f"Heritage No {heritage_no}: {visual_count} visual questions and {contextual_count} contextual questions generated.")

# DataFrame으로 변환
columns = ['QuestionId', 'Classification', 'Name of Cultural Heritage', 'Korean', 'question', 'answer']
final_dataframe = pd.DataFrame(results, columns=columns)

# CSV 파일로 저장
final_dataframe.to_csv("korea_heritage_VQA.csv", index=False)
print("테스트 CSV 파일이 성공적으로 저장되었습니다: korea_heritage_VQA.csv")
