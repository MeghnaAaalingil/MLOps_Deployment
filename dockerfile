FROM huggingface/transformers-pytorch-cpu:latest
COPY ./ /app
WORKDIR /app
RUN pip install -r requirements_prod.txt
EXPOSE 8000
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
CMD ["uvicorn", "app:app", "--host","--port", "8000"]