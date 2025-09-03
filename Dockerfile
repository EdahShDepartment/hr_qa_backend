# 使用官方 Python 基底映像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt 進容器
COPY requirements.txt .


RUN apt-get update && apt-get install -y tini
# 安裝 Python 套件
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案所有檔案
COPY . .

ENTRYPOINT ["/usr/bin/tini", "--"]

# 開放 Flask 預設的 5000 port
EXPOSE 5001

# 啟動指令 (用 wsgi.py 啟動 Flask 專案)
CMD ["python", "wsgi.py"]