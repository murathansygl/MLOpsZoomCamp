FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim
WORKDIR /app
COPY [ "model2.bin", "model.bin" ]