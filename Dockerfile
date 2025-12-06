FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

RUN python setup_knowledge_base.py || true

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
