FROM huggingface/transformers-pytorch-cpu:latest

COPY ./ /app
WORKDIR /app

# install requirements
RUN pip install "dvc[gdrive]"
RUN pip install -r requirements_prod.txt

# initialise dvc
RUN dvc init --no-scm
# configuring remote server in dvc
RUN dvc remote add -d storage gdrive://1drs_lQqAQjMeh54QyhMH6cWKb0gNYcmW
RUN dvc remote modify storage gdrive_use_service_account true
RUN dvc remote modify storage gdrive_service_account_json_file_path creds.json

# pulling the trained model
RUN dvc pull dvcfiles/model.onnx.dvc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# running the application
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]