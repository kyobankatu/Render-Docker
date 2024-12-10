# ベースイメージ
FROM python:3.9-slim

# 必要なLinuxパッケージをインストール
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-jpn \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

# 作業ディレクトリを設定
WORKDIR /app

# 必要なファイルをコンテナにコピー
COPY requirements.txt /app/requirements.txt
COPY main.py /app/main.py

# Pythonライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install opencv-python-headless

# アプリケーションのエントリーポイント
CMD ["python", "main.py"]