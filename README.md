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

**NOTE**: Models are downloaded to the first `--data-dir` directory.

## Docker Image

``` sh
docker run -it -p 10300:10300 -v /path/to/local/data:/data rhasspy/wyoming-whisper \
    --model tiny-int8 --language en
```

**NOTE**: Models are downloaded to `/data`, so make sure this points to a Docker volume.

[Source](https://github.com/rhasspy/wyoming-addons/tree/master/whisper)

## Running on Intel GPUs

This may not be the only way to run a server that makes use of an Intel GPU, but
it is known to work.

This method uses the [OpenAI Whisper](https://github.com/openai/whisper) library
for inference, which is generally not as fast as Faster Whisper, but it does use
PyTorch, which is able to run models on Intel GPUs.

After running `scripts/setup`, activate the Python virtual environment created
by it. This will ensure that the following steps install Python packages within 
it.

```sh
source .venv/bin/activate
```

Follow the steps at https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html

Install OpenAI Whisper:

```sh
pip install openai-whisper
```

When you run `scripts/run` be sure to pass the `--stt-library whisper` flag to
have it use the OpenAI Whisper library. And pass `--device xpu` to run the model
on an Intel GPU.
