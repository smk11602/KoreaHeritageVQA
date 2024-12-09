import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from models.question_classifier import QuestionClassifier
from utils.VQA_data_loader import create_dataloaders

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuestionClassifier().to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)
    criterion = CrossEntropyLoss()

    # 데이터 로더 생성
    train_loader, test_loader = create_dataloaders()

    # Best 모델 저장을 위한 변수 초기화
    best_accuracy = 0.0
    model_save_path = "your_dir/question_classifier.pth"

    # 모델 학습
    for epoch in range(20):  # epochs를 필요에 따라 조절
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}")

        # 모델 평가
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}, Test Accuracy: {accuracy * 100:.2f}%")

        # Best 모델 저장
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with accuracy: {accuracy * 100:.2f}% to {model_save_path}")

    print(f"Training completed. Best accuracy: {best_accuracy * 100:.2f}%")
