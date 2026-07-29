import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def calculate_metrics(targets, predictions, class_names=None):
    """
    Tính toán các chỉ số đánh giá mô hình phân loại.
    
    Args:
        targets (list or np.array): Nhãn thực tế (Ground truth).
        predictions (list or np.array): Nhãn dự đoán của mô hình.
        class_names (list, optional): Danh sách tên các nhãn để in report chi tiết.
        
    Returns:
        dict: Chứa các chỉ số (accuracy, precision, recall, f1).
    """
    targets = np.array(targets)
    predictions = np.array(predictions)
    
    acc = accuracy_score(targets, predictions)
    # macro average để tính trung bình không trọng số của các lớp
    precision = precision_score(targets, predictions, average='macro', zero_division=0)
    recall = recall_score(targets, predictions, average='macro', zero_division=0)
    f1 = f1_score(targets, predictions, average='macro', zero_division=0)
    
    print("\n" + "="*50)
    print("📊 BÁO CÁO CÁC CHỈ SỐ ĐÁNH GIÁ (METRICS)")
    print("="*50)
    print(f"Accuracy (Độ chính xác tổng thể):  {acc * 100:.2f}%")
    print(f"Precision (Độ chuẩn xác trung bình): {precision * 100:.2f}%")
    print(f"Recall (Độ phủ trung bình):          {recall * 100:.2f}%")
    print(f"F1-Score (Trung bình điều hòa):      {f1 * 100:.2f}%")
    
    # In ra báo cáo chi tiết cho từng lớp nếu số lượng lớp không quá khổng lồ
    if class_names is not None and len(class_names) <= 50:
        print("\n--- Báo cáo chi tiết từng lớp ---")
        print(classification_report(targets, predictions, target_names=class_names, zero_division=0))
    elif class_names is not None:
         print(f"\n[!] Bỏ qua in chi tiết vì số lượng lớp ({len(class_names)}) quá lớn, có thể gây nhiễu log.")
         
    print("="*50)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
