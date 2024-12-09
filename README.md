# KoreaHeritageVQA
**2024-2 융합콘텐츠학과 졸업논문 프로젝트**

---

## 📜 소개
**KoreaHeritageVQA**는 한국 문화유산 관련 시각적 질문 응답(VQA) 시스템을 개발하기 위해 구축된 데이터셋과 모델을 포함합니다.

### **프로젝트 목표**
1. **한국 문화유산 데이터셋 구축**  
   문화유산 description을 기반으로 시각적 질문과 컨텍스트 기반 질문을 생성.
   
   ![A4 - 1 (2)](https://github.com/user-attachments/assets/29b1e63f-ed3c-40db-b829-7c2d45dbbfc1)

2. **질문 유형 기반 VQA 모델 설계**  
   질문 유형에 따라 시각적 정보 또는 description을 활용한 답변 생성.

   ![A4 - 2 (2)](https://github.com/user-attachments/assets/5fe82131-6c71-4b71-ac81-e4a20e743d8a)

---

## 🛠️ 구현 정보

### 1. 구현 환경
- **운영체제(OS):** Rocky Linux 9.4  
- **사용 GPU:** NVIDIA RTX 3090, NVIDIA A6000  
- **사용 언어:** Python 3.9.0  

### 2. 주요 패키지 및 라이브러리

| **패키지 이름**      | **버전**  |
|-----------------------|-----------|
| PyTorch              | 2.5.1     |
| Transformers         | 4.46.3    |
| Pandas               | 1.3.5     |
| NumPy                | 1.26.4    |
| Scikit-learn         | 1.5.2     |


---

## 📂 프로젝트 구조
📂 프로젝트 구조

koreaheritageVQAdataset
├── evaluated_contextual_questions.csv        
├── korea_heritage_VQA_final.csv              # 최종 VQA 데이터셋
├── korea_heritage_VQA_finetune.json          # RAG용 데이터파일
├── korea_heritage_VQA_R.csv                  # 평가 관련 데이터
├── performance_reservation_list.csv          # 한국문화재단 데이터셋
├── processing_data.csv                       
models
├── question_classifier.py                    # 질문 유형 분류 모델 코드
├── train_classifier.py                       # Classifier 모델 학습 코드
├── QAModel
│   ├── rag.py                                # RAG 기반 QA 모델 코드
│   └── logs                                  
├── testmodel
│   ├── test_blip.py                          # BLIP 모델 테스트 코드
│   ├── test_ofa.py                           # OFA 모델 테스트 코드
│   └── test_vilt.py                          # ViLT 모델 테스트 코드
├── utils
│   └── VQA_data_loader.py                    
└── VQAModel
    ├── fine_tune_vilt.py                     # ViLT 모델 Fine-tuning 코드
    ├── vilt.py                               
    └── finetune_vilt
        ├── config.json                       # Fine-tuning 설정 파일
        ├── preprocessor_config.json          
        ├── special_tokens_map.json           
        ├── tokenizer.json                   
        ├── tokenizer_config.json            
        └── vocab.txt                        
prepare
├── contextual_eval.py                        # 컨텍스트 질문 생성 및 정제 코드
├── data_crawl.py                             # 데이터(이미지, 설명) 크롤링 스크립트
└── qa_generator.py                           # 질문 생성 코드
VQAdataset
├── ok_vqa
│   ├── ok_vqa_sampled.json                   # 샘플링된 OK-VQA 데이터셋
│   ├── OpenEnded_mscoco_train2014_questions.json # COCO Train2014 질문 데이터
│   └── OpenEnded_mscoco_val2014_questions.json   # COCO Val2014 질문 데이터
└── vqa_v2
    ├── v2_OpenEnded_mscoco_test2015_questions.json # COCO Test2015 질문 데이터
    ├── v2_OpenEnded_mscoco_train2014_questions.json # COCO Train2014 질문 데이터
    └── vqa_v2_sampled.json                       # 샘플링된 VQA v2 데이터셋



---

## 🔗 데이터셋 사용 방법

### **1. 데이터 다운로드**
`/prepare/data_crawl.py`를 실행하여 이미지(.jpg)와 설명(.txt)를 다운로드합니다.

---

### **2. 테스트 실행**
1에서 다운로드한 데이터와 `/koreaheritageVQAdataset/korea_heritage_VQA_final.csv`에 있는 질문을 이용하여 모델을 테스트합니다.

---

### **3. Fine-tuning**
1. `/models/train_classifier.py`를 실행하여 `question_classifier.pth`를 생성합니다.  
2. 생성한 모델을 이용해 Question Classifier를 실행합니다.  
3. `/models/VQAmodel/fine_tune_vilt.py`를 실행하여 fine-tuned VILT 모델을 얻습니다.  
4. `/models/VQAmodel/vilt.py`를 실행하여 결과를 확인합니다.

---

### **4. GPT를 이용한 질문 생성**
1. 다운로드한 데이터를 경로로 설정합니다.  
2. `/prepare/qa_generator.py`와 `/prepare/contextual_eval.py`를 실행하여 질문 생성 및 정제 과정을 수행합니다.

---

## 📊 구축 데이터셋 예시

![A4 - 3 (1)](https://github.com/user-attachments/assets/bdb18b97-9eb3-48aa-a277-77be9a40bc19)

