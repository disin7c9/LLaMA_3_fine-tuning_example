import re
import matplotlib.pyplot as plt

# 파일 경로
file_path = "loss기록.txt"

# 정규표현식을 사용해 train 손실, validation 손실, epoch 값 추출
train_pattern = r"\{'loss': ([\d\.]+),.*'epoch': ([\d\.]+)\}"
valid_pattern = r"\{'eval_loss': ([\d\.]+),.*'epoch': ([\d\.]+)\}"

train_losses = []
valid_losses = []
train_epochs = []
valid_epochs = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Train loss 추출
        train_match = re.search(train_pattern, line)
        if train_match:
            train_losses.append(float(train_match.group(1)))
            train_epochs.append(float(train_match.group(2)))
        
        # Validation loss 추출
        valid_match = re.search(valid_pattern, line)
        if valid_match:
            valid_losses.append(float(valid_match.group(1)))
            valid_epochs.append(float(valid_match.group(2)))

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(train_epochs, train_losses, label="Train Loss", linestyle='-')
plt.plot(valid_epochs, valid_losses, label="Validation Loss", linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Llama-3.1-8B-Instruct fine-tuning loss")
plt.xlim(0, 3)
plt.ylim(1, 2.5)
plt.legend()
plt.grid()
plt.tight_layout()

# 그래프 저장 또는 표시
plt.savefig("Llama-3.1-8B-Instruct_fine-tuning_loss.png")  # 그래프를 파일로 저장
plt.show()  # 그래프를 화면에 표시
