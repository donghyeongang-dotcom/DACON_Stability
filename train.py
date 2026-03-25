import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import StructuralStabilityDataset, get_train_transform, get_valid_transform
from model import DualViewPredictor

def train():
    # 학습 설정
    num_epochs = 30
    batch_size = 16
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 모델 선언
    model = DualViewPredictor(pretrained=True).to(device)

    # 2. 데이터셋 및 데이터로더 설정 (데이터 디렉토리는 대회 규격에 맞춰 수정)
    train_dataset = StructuralStabilityDataset("dummy_path", "dummy.csv", transform=get_train_transform(), is_train=True)
    dev_dataset = StructuralStabilityDataset("dummy_path", "dummy.csv", transform=get_valid_transform(), is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # 3. 손실 함수와 최적화 도구 
    # Label이 0(Stable)과 1(Unstable)일 경우
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Learning Rate Scheduler 활용하여 후반부 미세 조정
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_loss = float('inf')

    # 4. 학습 루프
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for view1, view2, labels in train_loader:
            view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(view1, view2)
            loss = criterion(outputs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        train_loss = running_loss / max(1, len(train_loader))
        
        # 5. 검증 루프
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for view1, view2, labels in dev_loader:
                view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)
                outputs = model(view1, view2)
                
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                
                # 시그모이드를 거쳐 0.5 확률 기준으로 라벨 결정
                preds = (torch.sigmoid(outputs).squeeze() > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_loss = val_loss / max(1, len(dev_loader))
        val_acc = (correct / total * 100) if total > 0 else 0
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
        scheduler.step(val_loss)

        # Early Stopping & Best Model Save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(">>> Saved best model!")

if __name__ == "__main__":
    print("Train baseline script.")
    # 더미 데이터를 사용하여 학습 루프 테스트 실행
    train()
