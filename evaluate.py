import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from src.loader import get_dataloaders
from train import get_model # Re-use the factory function

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")
    
    # 1. Load Test Data
    print(f"Loading test data from {config.TEST_DATA_PATH}...")
    if not os.path.exists(config.TEST_DATA_PATH):
        print("❌ Lỗi: Không tìm thấy file dữ liệu test. Vui lòng kiểm tra thư mục data/processed/")
        return

    # We only need test loader, so we can just pass test_path to get_dataloaders
    # or create a temporary dummy train/val just to get test loader. 
    # Let's import dataset directly to be cleaner.
    from src.loader import VSL400KeypointDataset3D
    from torch.utils.data import DataLoader
    
    test_dataset = VSL400KeypointDataset3D(config.TEST_DATA_PATH, max_len=config.SEQUENCE_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Init Model & Load Weights
    model = get_model(args.model, config.NUM_CLASSES).to(device)
    
    weights_path = args.weights
    if not weights_path:
        weights_path = os.path.join(config.SAVED_MODELS_DIR, f"{args.model}_best.pth")
        
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"✅ Đã nạp thành công weights từ: {weights_path}")
    else:
        print(f"❌ Lỗi: KHÔNG tìm thấy file model tại {weights_path}. Vui lòng chạy train trước!")
        return

    # 3. Evaluation
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Testing"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            test_loss += loss.item()
            preds = outputs.argmax(dim=-1)
            
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    test_acc = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    
    # 4. In kết quả
    print("\n" + "="*50)
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
    print("="*50)
    print(f"Tổng số mẫu test:   {total}")
    print(f"Test Loss:          {avg_test_loss:.4f}")
    print(f"Test Accuracy:      {test_acc:.2f}%")
    print("="*50)

    # Xem thử vài dự đoán
    print("\n🔍 XEM THỬ 10 DỰ ĐOÁN MẪU:")
    print(f"{'STT':<5} | {'Nhãn Thực Tế':<20} | {'Dự Đoán':<20} | {'Kết quả'}")
    print("-" * 60)
    for i in range(min(10, len(all_preds))):
        actual_name = config.ACTIONS[all_targets[i]] if config.ACTIONS else f"Class {all_targets[i]}"
        pred_name   = config.ACTIONS[all_preds[i]] if config.ACTIONS else f"Class {all_preds[i]}"
        status      = "✅ ĐÚNG" if all_targets[i] == all_preds[i] else "❌ SAI"
        print(f"{i+1:<5} | {actual_name:<20} | {pred_name:<20} | {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VSL400 Models")
    parser.add_argument('--model', type=str, default='bilstm', help="Tên mô hình muốn evaluate (vd: bilstm, gcn)")
    parser.add_argument('--weights', type=str, default='', help="Đường dẫn file .pth (nếu trống sẽ lấy mặc định trong saved_models)")
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help="Batch size")
    
    args = parser.parse_args()
    evaluate(args)
