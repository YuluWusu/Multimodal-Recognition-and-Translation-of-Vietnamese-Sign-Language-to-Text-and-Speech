import matplotlib.pyplot as plt
import re
import os

def plot_logs(log_file_path, output_image_path):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epochs = []

    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Regex to extract metrics:
    # Example: Epoch 01/30 | Train Loss: 5.1727 - Train Acc: 3.28% | Val Loss: 4.3432 - Val Acc: 7.76%
    pattern = re.compile(r"Epoch (\d+)/\d+ \| Train Loss: ([\d.]+) - Train Acc: ([\d.]+)% \| Val Loss: ([\d.]+) - Val Acc: ([\d.]+)%")

    for line in lines:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            train_accs.append(float(match.group(3)))
            val_losses.append(float(match.group(4)))
            val_accs.append(float(match.group(5)))

    if not epochs:
        print(f"No log data found in {log_file_path}.")
        return

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, label='Train Accuracy', marker='o', markersize=4)
    plt.plot(epochs, val_accs, label='Val Accuracy', marker='s', markersize=4)
    plt.title('Độ chính xác (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=4)
    plt.plot(epochs, val_losses, label='Val Loss', marker='s', markersize=4)
    plt.title('Hàm mất mát (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f'Kết quả huấn luyện: {os.path.basename(os.path.dirname(log_file_path))}')
    plt.tight_layout()
    plt.savefig(output_image_path)
    print(f"Saved chart to {os.path.basename(output_image_path)}")
    plt.close() # Đóng plot thay vì plt.show() để chạy ngầm không bị chặn

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    log_30ep = os.path.join(base_dir, "30ep", "quatrinh+ketqua.txt")
    img_30ep = os.path.join(base_dir, "30ep_chart.png")
    
    log_60ep = os.path.join(base_dir, "60ep", "quatrinh+ketqua.txt")
    img_60ep = os.path.join(base_dir, "60ep_chart.png")

    if os.path.exists(log_30ep):
        plot_logs(log_30ep, img_30ep)
    
    if os.path.exists(log_60ep):
        plot_logs(log_60ep, img_60ep)
