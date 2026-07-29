import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from src.loader import get_dataloaders
from models.bilstm import BiLSTM

def get_model(model_name, num_classes):
    if model_name.lower() == 'bilstm':
        
        return BiLSTM(num_joints=76, coords=3, hidden_size=256, num_classes=num_classes)
   
    else:
        raise ValueError(f"Model {model_name} không được hỗ trợ!")

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")
    
    # 1. Load Data
    print(f"Loading data from {config.TRAIN_DATA_PATH} and {config.VAL_DATA_PATH}...")
    if not os.path.exists(config.TRAIN_DATA_PATH) or not os.path.exists(config.VAL_DATA_PATH):
        print("❌ Lỗi: Không tìm thấy file dữ liệu train/val. Vui lòng kiểm tra thư mục data/processed/")
        return

    train_loader, val_loader = get_dataloaders(
        config.TRAIN_DATA_PATH, 
        config.VAL_DATA_PATH, 
        batch_size=args.batch_size, 
        max_len=config.SEQUENCE_LENGTH
    )
    
    # 2. Init Model
    model = get_model(args.model, config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 3. Training Loop
    best_acc = 0.0
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    save_path = os.path.join(config.SAVED_MODELS_DIR, f"{args.model}_best.pth")
    
    print(f"\n🚀 Bắt đầu huấn luyện mô hình {args.model.upper()}...")
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs} [Train]"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=-1)
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)

        train_acc = 100 * correct_train / total_train
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                preds = outputs.argmax(dim=-1)
                correct_val += (preds == y_batch).sum().item()
                total_val += y_batch.size(0)

        val_acc = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(val_acc)

        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> [Saved Model] Đạt Val Acc cao nhất: {best_acc:.2f}%")

    print(f"\n🎉 Hoàn tất! Mô hình đã được lưu tại: {save_path} (Best Acc: {best_acc:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training VSL400 Models")
    parser.add_argument('--model', type=str, default='bilstm', help="Tên mô hình muốn train (vd: bilstm, gcn, ...)")
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help="Số lượng epoch")
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help="Learning rate")
    
    args = parser.parse_args()
    train(args)
