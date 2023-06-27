From python:3.8-slim-buster
COPY  . /app
WORKDIR /app
RUN apt update -y && apt install awscli -y
RUN apt-get update && pip install -r requirements.txt
CMD ["python","app.py"]