FROM python:3.11-slim

WORKDIR /app

COPY Pipfile /app/Pipfile

RUN pip install --no-cache-dir pipenv \
    && pipenv install --system --skip-lock

COPY chainlit_ui /app

EXPOSE 8501

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8501"]
