# Wyoming Faster Whisper

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the [faster-whisper](https://github.com/guillaumekln/faster-whisper/) speech to text system.

## Home Assistant Add-on

[![Show add-on](https://my.home-assistant.io/badges/supervisor_addon.svg)](https://my.home-assistant.io/redirect/supervisor_addon/?addon=core_whisper)

[Source](https://github.com/home-assistant/addons/tree/master/whisper)

## Local Install

Clone the repository and set up Python virtual environment:

``` sh
git clone https://github.com/rhasspy/wyoming-faster-whisper.git
cd wyoming-faster-whisper
script/setup
```

Run a server anyone can connect to:

```sh
script/run --model tiny-int8 --language en --uri 'tcp://0.0.0.0:10300' --data-dir /data --download-dir /data
```

The `--model` can also be a HuggingFace model like `Systran/faster-distil-whisper-small.en`

## Docker

### CPU

#### Build Docker Image
``` sh
docker build -t wyoming-whisper -f Dockerfile.cpu .
```

#### Run Docker Image 
``` sh
docker run -it -p 10300:10300 -v /path/to/local/data:/data rhasspy/wyoming-whisper \
    --model tiny-int8 --language en --data-dir /path/to/local/data
```

#### Docker Compose

When using docker compose for CPU, 

``` sh
docker compose up -f docker-compose.cpu.yml -d
```

### GPU

#### Build Docker Image
``` sh
docker build -t wyoming-whisper -f Dockerfile.gpu .
```

#### Run Docker Image (GPU)

``` sh
docker run -it -p 10300:10300 -v /path/to/local/data:/data wyoming-whisper \
    --model tiny-int8 --language en --device cuda --data-dir /path/to/local/data
```

#### Docker Compose

``` sh
docker compose up -d
```

**NOTE**: Models are downloaded temporarily to the `HF_HUB_CACHE` directory, which defaults to `~/.cache/huggingface/hub`.
You may need to adjust this environment variable when using a read-only root filesystem (e.g., `HF_HUB_CACHE=/tmp`).

[Source](https://github.com/rhasspy/wyoming-addons/tree/master/whisper)
