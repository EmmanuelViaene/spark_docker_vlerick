FROM python:3.9

WORKDIR /app

COPY . /app

USER root

RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get install -y ant && \
    apt-get clean;

RUN useradd -ms /bin/bash python-api
USER python-api

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python", "/app/main.py"]

