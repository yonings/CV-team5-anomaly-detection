# download_model.py
import os
import gdown

def ensure_model():
    file_id = "1zr8qpXomwC4I_zWTatgQgIE2m9mjBnWJ"
    
    output_dir = "checkpoints/autoencoder"
    output_path = os.path.join(output_dir, "best_model_epoch_100.pth")

    # 폴더가 없다면 생성
    os.makedirs(output_dir, exist_ok=True)

    # 모델이 없으면 다운로드
    if not os.path.exists(output_path):
        print("Downloading best_model_epoch_100.pth...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print("Download complete!")
    else:
        print("Model already exists:", output_path)

if __name__ == "__main__":
    ensure_model()
