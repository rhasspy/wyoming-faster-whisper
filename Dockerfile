# ctranslate2 does not seem to support newer versions of python
FROM python:3.9 AS builder

WORKDIR /src

COPY . .

RUN ./script/setup && ./script/package

# ctranslate2 does not seem to support newer versions of python
FROM python:3.9

RUN --mount=type=bind,from=builder,target=/mnt/builder pip3 install /mnt/builder/src/dist/*.whl

WORKDIR /src
COPY docker-entrypoint.sh .

EXPOSE 10300/tcp
VOLUME /data
LABEL org.opencontainers.image.source=https://github.com/rhasspy/wyoming-faster-whisper

ENTRYPOINT ["./docker-entrypoint.sh"]
