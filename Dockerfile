FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY data/ ./data/
COPY src/preprocess.py .
COPY src/train.py .
COPY src/app.py .

RUN python preprocess.py
RUN python train.py

ENV MODEL_PATH=model/model.pkl
ENV DATA_PATH=data/processed.csv

EXPOSE 5000

CMD ["python", "app.py"]
