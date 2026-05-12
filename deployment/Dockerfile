FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    MODEL_PATH=/app/model/trained_model.pth

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -v

COPY app ./app
COPY model ./model

EXPOSE 7860

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
CMD ["python", "app.py"]

