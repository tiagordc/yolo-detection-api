FROM python:3.7

COPY . /app
WORKDIR /app
RUN pip --no-cache-dir install -r requirements.txt

# ssh
ENV SSH_PASSWD "root:Docker!"
RUN apt-get update \
        && apt-get install -y --no-install-recommends dialog \
        && apt-get update \
    && apt-get install -y --no-install-recommends openssh-server tesseract-ocr libtesseract-dev \
    && echo "$SSH_PASSWD" | chpasswd 

COPY ./docker/sshd_config /etc/ssh/
COPY ./docker/init.sh /usr/local/bin/

RUN chmod u+x /usr/local/bin/init.sh
EXPOSE 5000 2222

ENTRYPOINT ["init.sh"]
