# Dockerfile
# Containerisation of the Flask API
# Reference: Docker documentation
# https://docs.docker.com/

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/app.py .
COPY model/model.pkl model/model.pkl

ENV MODEL_PATH=model/model.pkl

EXPOSE 5000

CMD ["python", "app.py"]
