import torch
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

# ViLT 모델 및 프로세서 초기화
vilt_model_path = "models\VQAmodel\finetune_vilt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vilt_processor = ViltProcessor.from_pretrained(vilt_model_path)
vilt_model = ViltForQuestionAnswering.from_pretrained(vilt_model_path).to(device)
vilt_model.eval()

def process_with_vilt(question, metadata, reference_answer, topk=3):
   
    #data crawling을 통해 다운 받은 img 파일 경로 입력
    image_path = f"/your_dir/{metadata['No']}.jpg"
    
    try:
        # 이미지 로드
        image = Image.open(image_path).convert("RGB")
        
        # 질문과 이미지를 ViLT 프로세서로 전처리
        inputs = vilt_processor(images=image, text=question, return_tensors="pt").to(device)
        
        # ViLT 모델 추론
        with torch.no_grad():
            outputs = vilt_model(**inputs)
            logits = outputs.logits.squeeze(0)
        
        # Top-k 예측값 추출
        topk_ids = torch.topk(logits, topk).indices.tolist()
        topk_outputs = [vilt_processor.tokenizer.decode([idx]) for idx in topk_ids]
        top1_output = topk_outputs[0]

        # 평가 지표 초기화
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        # METEOR 계산 (Top-1만)
        meteor = meteor_score([reference_answer], top1_output)

        # ROUGE 계산 (Top-1만)
        rouge_scores = scorer.score(reference_answer, top1_output)
        rouge1 = rouge_scores['rouge1'].fmeasure
        rouge2 = rouge_scores['rouge2'].fmeasure
        rougel = rouge_scores['rougeL'].fmeasure

        # Accuracy 계산
        top1_correct = reference_answer.strip().lower() == top1_output.strip().lower()
        topk_correct = any(reference_answer.strip().lower() == output.strip().lower() for output in topk_outputs)

        return {
            "predicted_answers": topk_outputs,
            "accuracy_top1": int(top1_correct),
            "accuracy_topk": int(topk_correct),
            "meteor": meteor,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougel": rougel
        }
    
    except FileNotFoundError:
        return {"error": f"Image not found: {image_path}"}
    except Exception as e:
        return {"error": str(e)}
