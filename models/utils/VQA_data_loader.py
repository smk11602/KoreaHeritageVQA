from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import json
import random
import torch
from transformers import BertTokenizer

class VQADataset(Dataset):
    def __init__(self, data_path, label, tokenizer, max_len=128):
        with open(data_path, "r") as f:
            data = json.load(f)["questions"]
        self.questions = [item["question"] for item in data]
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        inputs = self.tokenizer(
            question,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "label": torch.tensor(self.label, dtype=torch.long),
        }

def create_dataloaders(batch_size=32, test_ratio=0.2, seed=42):
    # Set random seeds for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load training datasets
    ok_vqa_train_dataset = VQADataset("VQAdataset\ok_vqa\ok_vqa_sampled.json", label=0, tokenizer=tokenizer)  # contextual
    vqa_v2_train_dataset = VQADataset("VQAdataset\vqa_v2\vqa_v2_sampled.json", label=1, tokenizer=tokenizer)  # visual

    # Load test datasets
    ok_vqa_test_path = "VQAdataset\ok_vqa\OpenEnded_mscoco_val2014_questions.json"
    vqa_v2_test_path = "VQAdataset\vqa_v2\v2_OpenEnded_mscoco_test2015_questions.json"
    ok_vqa_test_dataset = VQADataset(ok_vqa_test_path, label=0, tokenizer=tokenizer)
    vqa_v2_test_dataset = VQADataset(vqa_v2_test_path, label=1, tokenizer=tokenizer)

    # Determine sample size for test split
    train_sample_size = len(ok_vqa_train_dataset) + len(vqa_v2_train_dataset)
    test_sample_size = int(train_sample_size * test_ratio)

    # Sample indices for test datasets
    ok_vqa_test_sample_size = test_sample_size // 2
    vqa_v2_test_sample_size = test_sample_size - ok_vqa_test_sample_size

    ok_vqa_test_indices = random.sample(range(len(ok_vqa_test_dataset)), ok_vqa_test_sample_size)
    vqa_v2_test_indices = random.sample(range(len(vqa_v2_test_dataset)), vqa_v2_test_sample_size)

    # Create subsets based on sampled indices
    ok_vqa_test_sampled = Subset(ok_vqa_test_dataset, ok_vqa_test_indices)
    vqa_v2_test_sampled = Subset(vqa_v2_test_dataset, vqa_v2_test_indices)

    # Combine sampled test subsets into a single dataset
    combined_test_dataset = ConcatDataset([ok_vqa_test_sampled, vqa_v2_test_sampled])

    # Create DataLoaders
    train_loader = DataLoader(ConcatDataset([ok_vqa_train_dataset, vqa_v2_train_dataset]), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader