FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m lstm

RUN chown -R lstm:lstm /home/lstm/

COPY --chown=lstm *.* /home/lstm/ner_tagger/

USER lstm

RUN cd /home/lstm/ner_tagger && pip3 install -r requirements.txt
RUN pip3 install mkl

COPY app /home/lstm/ner_tagger/app
WORKDIR /home/lstm/ner_tagger/app/
ENV PATH="/home/lstm/.local/bin:$PATH"

EXPOSE 8080

CMD ["gunicorn", "lstm_ner_app:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080"]