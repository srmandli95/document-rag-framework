FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    chainlit \
    httpx

COPY . /app

EXPOSE 8501

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8501"]