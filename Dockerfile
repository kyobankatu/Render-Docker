# ベースイメージ
FROM python:3.9-slim

# 必要なLinuxパッケージをインストール
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && apt-get clean

# 作業ディレクトリを設定
WORKDIR /app

# プロジェクトファイルをコンテナにコピー
COPY . /app

# Pythonライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのエントリーポイント
CMD ["python", "app.py"]
