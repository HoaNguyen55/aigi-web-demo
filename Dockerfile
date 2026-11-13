# Ảnh CPU cho demo local. Source + checkpoint gốc KHÔNG nằm trong image: gắn thư mục aigi_final vào /aigi (chỉ đọc).
#   docker build -t aigi-web .
#   docker run --rm -p 127.0.0.1:8000:8000 -v /mnt/d/xulyanh/extracted/aigi_final:/aigi:ro aigi-web
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 AIGI_ROOT=/aigi HOST=0.0.0.0 PORT=8000
WORKDIR /app/backend
COPY backend/requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY backend/app.py .
COPY --from=frontend /frontend/dist /app/frontend/dist
RUN useradd --create-home demo
USER demo
EXPOSE 8000
# start-period: nạp 4 checkpoint mất ~17 giây trên máy thử
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]
CMD ["python", "app.py"]
