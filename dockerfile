FROM huggingface/transformers-pytorch-cpu:latest
COPY ./ /app
WORKDIR /app

RUN pip install "dvc[gdrive]==3.59.2"
RUN pip install -r requirements_prod.txt

# initialise dvc
RUN dvc init --no-scm
# configuring remote server in dvc
RUN dvc remote add -d storage gdrive://1drs_lQqAQjMeh54QyhMH6cWKb0gNYcmW
RUN dvc remote modify storage gdrive_use_service_account true
RUN dvc remote modify storage gdrive_service_account_json_file_path /app/secret.json

RUN dvc pull dvcfiles/trained_model.dvc

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host","--port", "8000"]