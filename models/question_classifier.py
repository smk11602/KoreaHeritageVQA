import torch
import torch.nn as nn
from transformers import AutoModel

class QuestionClassifier(nn.Module):
    def __init__(self):
        super(QuestionClassifier, self).__init__()
        # DeBERTa Base 모델 불러오기
        self.deberta = AutoModel.from_pretrained("microsoft/deberta-base")
        self.classifier = nn.Linear(self.deberta.config.hidden_size, 2)  # 2 classes: contextual, visual

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_output)
