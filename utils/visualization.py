import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix(targets, predictions, class_names, save_path="logs/confusion_matrix.png"):
    """
    Vẽ và lưu biểu đồ Ma trận nhầm lẫn (Confusion Matrix).
    
    Args:
        targets (list or np.array): Nhãn thực tế.
        predictions (list or np.array): Nhãn dự đoán.
        class_names (list): Danh sách tên các nhãn.
        save_path (str): Đường dẫn lưu file ảnh.
    """
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(targets, predictions)
    
   
    fig_size = max(10, len(class_names) * 0.4)
    plt.figure(figsize=(fig_size, fig_size))
    
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='g', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)', fontsize=16)
    plt.xlabel('Dự đoán (Predicted)', fontsize=12)
    plt.ylabel('Thực tế (Actual)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Đã lưu Ma trận nhầm lẫn tại: {save_path}")

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path="logs/training_history.png"):
    """
    Vẽ biểu đồ Loss và Accuracy qua các Epochs trong quá trình huấn luyện.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Biểu đồ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, 'bo-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'rs-', label='Validation Accuracy')
    plt.title('Độ chính xác qua từng vòng lặp')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Biểu đồ Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
    plt.plot(epochs, val_losses, 'rs-', label='Validation Loss')
    plt.title('Hàm mất mát qua từng vòng lặp')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Đã lưu Biểu đồ huấn luyện tại: {save_path}")
