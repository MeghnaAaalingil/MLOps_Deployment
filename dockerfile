# FROM huggingface/transformers-pytorch-cpu:latest
FROM python:3.10-slim

# Install build dependencies and system tools
RUN apt-get update && apt-get install -y git curl gcc libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

COPY ./ /app
WORKDIR /app
COPY models/model.onnx.dvc /app/models/

# install requirements
RUN pip install "dvc[gdrive]"
RUN pip install -r requirements_prod.txt

# initialise dvc
RUN dvc init --no-scm

RUN echo "DVC Version Info:" && dvc version

# configuring remote server in dvc
RUN dvc remote add -d storage gdrive://1drs_lQqAQjMeh54QyhMH6cWKb0gNYcmW
RUN dvc remote modify storage gdrive_use_service_account true
RUN dvc remote modify storage gdrive_service_account_json_file_path creds.json

RUN cat .dvc/config
# pulling the trained model
RUN dvc pull models/model.onnx.dvc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# running the application
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]