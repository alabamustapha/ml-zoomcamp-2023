FROM python:3.9-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "predict.py", "model_C=1.0.bin", "./"]

RUN pipenv install --system --deploy

EXPOSE 9090

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9090", "predict:app"]