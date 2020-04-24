FROM pure/python:3.7-cuda10.2-base

RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr libtesseract-dev git && \
    mkdir /app && \
    cd /app && \
    git clone https://github.com/tiagordc/yolo-detection-api.git . && \
    pip install --upgrade pip && \
    pip --no-cache-dir install -r requirements.txt && \
    apt-get purge --autoremove -y git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD [ "application.py" ]
