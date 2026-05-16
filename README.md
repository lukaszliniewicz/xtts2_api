# XTTS FastAPI

OpenAI-compatible TTS server powered by [Coqui XTTS v2](https://github.com/coqui-ai/TTS).
Environment managed by [pixi](https://pixi.sh).

## Quick Start

```bash
# Windows (double-click run.bat, or run from terminal)
run.bat

# Linux / macOS
bash run.sh

# Force CPU (no GPU detection)
run.bat --cpu

# Use an existing pixi binary (skip pixi download)
run.bat --pixi-path "D:\\tools\\pixi.exe"
bash run.sh --pixi-path /usr/local/bin/pixi
```

The bootstrapper downloads pixi, creates a Python 3.12 environment, detects your
hardware (NVIDIA CUDA, AMD ROCm, or CPU), installs the appropriate PyTorch wheel
and coqui-tts, preloads the default model into `models/XTTS_2.0.2/` by default
(unless `XTTS_COQUI_TOS_AGREED=false`), then starts the server on `http://0.0.0.0:8020`.

## Platform & Backend Compatibility

| Platform | Backend | Status |
|----------|---------|--------|
| Windows | CUDA (NVIDIA) | ✅ Tested |
| Windows | CPU | ✅ Tested |
| Windows | ROCm (AMD) | ⚠️ Untested, should work with HIP SDK |
| Linux | CUDA (NVIDIA) | ⚠️ Untested, expected to work |
| Linux | ROCm (AMD) | ⚠️ Untested, expected to work |
| Linux | CPU | ⚠️ Untested |
| macOS | CPU / MPS | ⚠️ Untested (MPS is buggy with XTTS) |

Only **Windows CPU and CUDA** have been tested. Other platforms should work but
are untested. Pull requests welcome.

## Manual Steps (if bootstrapper fails)

```bash
# 1. Ensure pixi is in bin/
bin\pixi install

# 2. Install PyTorch for your backend
bin\pixi run pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# 3. Install coqui-tts + pin transformers
bin\pixi run pip install coqui-tts
bin\pixi run pip install "transformers>=4,<5"

# 4. (CUDA only) Install DeepSpeed
bin\pixi run pip install --no-deps deepspeed==0.16.5

# 5. Start
bin\pixi run python -m uvicorn src.xtts_fastapi.main:app --host 0.0.0.0 --port 8020
```

## API Reference

### `GET /health`

Server health and status.

```json
{
  "status": "ok",
  "version": "0.1.0",
  "model": "tts_models/multilingual/multi-dataset/xtts_v2",
  "device": "cuda",
  "deepspeed": true,
  "voices": 5
}
```

### `GET /v1/models`

List available models (OpenAI-compatible).

### `POST /v1/files`

Upload a file (OpenAI-compatible multipart endpoint).

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Single uploaded file |
| `purpose` | string | OpenAI-style file purpose (use `user_data` for voice references) |
| `name` | string | Optional filename override (stored in response as `filename`) |

Only WAV uploads are accepted on this endpoint.

- non-WAV uploads are rejected with `422 unsupported_file_type`
- WAV bytes are normalized into a voice ID and stored as `voices/<voice_id>/<voice_id>.wav`
- the endpoint returns a **voice** object (not a file object)

### `GET /v1/files`

List uploaded files (OpenAI-compatible). Supports `limit`, `after`, `order`, and
`purpose` query parameters. This is primarily useful for existing legacy file IDs.

### `GET /v1/files/{file_id}`

Retrieve legacy file metadata.

### `GET /v1/files/{file_id}/content`

Download legacy file bytes.

### `DELETE /v1/files/{file_id}`

Delete a legacy file.

### `GET /v1/voices`

List uploaded voices.

### `POST /v1/voices`

Upload voice samples. Accepts multipart form data.

| Field | Type | Description |
|-------|------|-------------|
| `files` | File[] | One or more WAV audio samples |
| `voice_id` | string | Optional custom ID (normalized to lowercase slug) |
| `language` | string | Optional language code |

If `voice_id` is omitted, the server derives it from the first uploaded filename
stem and normalizes it (lowercase, punctuation/spaces collapsed to `-`).

### `DELETE /v1/voices/{voice_id}`

Delete a voice.

### `POST /v1/audio/speech`

Generate speech (OpenAI-compatible).

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | required | Model ID (e.g. `tts_models/multilingual/multi-dataset/xtts_v2`) |
| `input` | string | required | Text to synthesize (max 4096 chars) |
| `voice` | string \| object | required | Voice ID or `{"id":"..."}` (legacy file IDs still resolve if present) |
| `language` | string | `"en"` | Language code |
| `response_format` | string | `"wav"` | `wav`, `mp3`, `opus`, `aac`, `flac`, `pcm` |
| `speed` | float | `1.0` | Playback speed (0.25–4.0) |
| `stream_format` | string | — | `audio` for raw stream, `sse` for SSE events |
| `speaker_wav` | string[] | — | Direct paths to reference audio (bypasses voice store) |
| `xtts` | object | — | Per-request XTTS parameter overrides (see below) |
| `instructions` | string | — | Optional JSON workaround for LiteLLM speech (`{"language":"en","xtts":{...}}`) |

**Response:** Raw audio bytes with `Content-Type: audio/wav` (or the requested format).

## Recommended SDK Flow

Use `/v1/voices` when your client can call it directly. `/v1/files` now accepts
WAV only and returns a voice object (`id` is the voice ID to pass as `voice`).

### OpenAI Python SDK

```python
import requests
from openai import OpenAI

client = OpenAI(api_key="sk-local", base_url="http://127.0.0.1:8020/v1")

with open("voice.wav", "rb") as f:
    upload = requests.post(
        "http://127.0.0.1:8020/v1/files",
        files={"file": ("voice.wav", f, "audio/wav")},
        data={"purpose": "user_data"},
        timeout=30,
    )
upload.raise_for_status()
voice_id = upload.json()["id"]

speech = client.audio.speech.create(
    model="tts_models/multilingual/multi-dataset/xtts_v2",
    input="Hello from XTTS.",
    voice={"id": voice_id},
    response_format="wav",
)
speech.write_to_file("speech.wav")

requests.delete(f"http://127.0.0.1:8020/v1/voices/{voice_id}", timeout=30)
```

### LiteLLM Python SDK

Some LiteLLM versions do not forward `extra_body` on `aspeech()` for
OpenAI-compatible TTS. Use `instructions` with a JSON object as a workaround.

```python
import asyncio
import json
import litellm
import requests


async def main():
    with open("voice.wav", "rb") as f:
        upload = requests.post(
            "http://127.0.0.1:8020/v1/files",
            files={"file": ("voice.wav", f, "audio/wav")},
            data={"purpose": "assistants"},
            timeout=30,
        )
    upload.raise_for_status()
    voice_id = upload.json()["id"]

    speech = await litellm.aspeech(
        model="openai/tts_models/multilingual/multi-dataset/xtts_v2",
        input="Hello from LiteLLM.",
        voice=voice_id,
        response_format="wav",
        speed=1.04,
        instructions=json.dumps(
            {
                "language": "en",
                "xtts": {
                    "temperature": 0.66,
                    "top_p": 0.88,
                    "repetition_penalty": 4.1,
                },
            }
        ),
        api_base="http://127.0.0.1:8020/v1",
        api_key="sk-local",
    )
    speech.write_to_file("speech.wav")

    requests.delete(f"http://127.0.0.1:8020/v1/voices/{voice_id}", timeout=30)

asyncio.run(main())
```

### XTTS Parameters

Pass via `xtts` in the speech request body. Values here override the server defaults
for that request only.

LiteLLM workaround: place the same payload inside `instructions` as JSON,
for example `{"xtts":{"temperature":0.65}}`. You can also use
`{"temp":0.65}` as an alias for `temperature`.

Use `xtts` directly whenever your client supports it. The `instructions` JSON path
is a compatibility fallback for LiteLLM `aspeech()` versions that do not forward
`extra_body`.

`gpt_cond_len`, `gpt_cond_chunk_len`, `max_ref_length`, and `sound_norm_refs` are
conditioning-audio options. They are applied when extracting speaker latents and
are not forwarded to HuggingFace `generate()` kwargs.

| Parameter | Server Default | Range | Description |
|-----------|---------------|-------|-------------|
| `temperature` | `0.7` | 0.0–2.0 | Softmax temperature. Lower = more stable, higher = more expressive but riskier |
| `top_p` | `0.85` | 0.0–1.0 | Nucleus sampling cutoff |
| `top_k` | `50` | 0+ | Top-k sampling limit |
| `repetition_penalty` | `5.0` | 0.0–100.0 | Penalize repeated tokens. Higher = fewer stutters/hallucinations |
| `length_penalty` | `1.0` | -10.0–10.0 | Length penalty. Slightly above 1.0 can prevent early cutoffs |
| `do_sample` | *(true)* | bool | Enable sampling (required for temp/top_p/top_k to work) |
| `num_beams` | `1` | 1+ | Beam search width. >1 is slower with marginal quality gain |
| `enable_text_splitting` | `false` | bool | Split text into sentences for unlimited length. Disabled = better prosody but ~400 token limit |
| `gpt_cond_len` | `12` | 1+ sec | Seconds of reference audio for GPT conditioning |
| `gpt_cond_chunk_len` | `6` | 1+ sec | Chunk size for conditioning audio processing |
| `max_ref_length` | `12` | 1+ sec | Max seconds of reference audio for decoder embedding (`max_ref_len` alias is accepted) |
| `sound_norm_refs` | `true` | bool | Normalize reference audio loudness |
| `librosa_trim_db` | — | 0+ | Optional silence-trim threshold for reference audio preprocessing |
| `stream_chunk_size` | `20` | 1+ tokens | GPT tokens per stream chunk (lower = lower latency) |
| `overlap_wav_len` | `1024` | 0+ samples | Cross-fade samples between stream chunks (~42ms at 24kHz) |
| `hf_generate_kwargs` | — | object | Raw kwargs passed to HuggingFace `generate()` |

Example with overrides:

```json
{
  "model": "tts_models/multilingual/multi-dataset/xtts_v2",
  "input": "Hello, this is a test.",
  "voice": "my_voice",
  "xtts": {
    "temperature": 0.65,
    "repetition_penalty": 5.0,
    "enable_text_splitting": true
  }
}
```

## Configuration

Set via environment variables with `XTTS_` prefix or in a `.env` file:

```env
XTTS_DEVICE=auto
XTTS_COQUI_TOS_AGREED=true
XTTS_TEMPERATURE=0.7
XTTS_REPETITION_PENALTY=5.0
```

| Variable | Default | Description |
|----------|---------|-------------|
| `XTTS_HOST` | `0.0.0.0` | Bind address |
| `XTTS_PORT` | `8020` | Port |
| `XTTS_DEVICE` | `auto` | `auto`, `cuda`, `cpu` |
| `XTTS_USE_DEEPSPEED` | `true` | Enable DeepSpeed (CUDA only) |
| `XTTS_COQUI_TOS_AGREED` | `true` | Accept Coqui terms of service (set `false` to opt out) |
| `XTTS_DEFAULT_LANGUAGE` | `en` | Default language |
| `XTTS_DEFAULT_MODEL` | `tts_models/multilingual/multi-dataset/xtts_v2` | Default model |
| `XTTS_DEFAULT_MODEL_LOCAL_DIR` | `XTTS_2.0.2` | Local folder name under `models/` for the default model |
| `XTTS_TEMPERATURE` | `0.7` | See XTTS Parameters |
| `XTTS_TOP_P` | `0.85` | |
| `XTTS_TOP_K` | `50` | |
| `XTTS_REPETITION_PENALTY` | `5.0` | |
| `XTTS_LENGTH_PENALTY` | `1.0` | |
| `XTTS_ENABLE_TEXT_SPLITTING` | `false` | |
| `XTTS_GPT_COND_LEN` | `12` | |
| `XTTS_GPT_COND_CHUNK_LEN` | `6` | |
| `XTTS_MAX_REF_LENGTH` | `12` | (legacy `XTTS_MAX_REF_LEN` is also accepted) |
| `XTTS_SOUND_NORM_REFS` | `true` | |
| `XTTS_LIBROSA_TRIM_DB` | — | Optional silence trim threshold |
| `XTTS_MIN_REF_AUDIO_SECONDS` | `0.5` | Minimum reference clip duration |
| `XTTS_STREAM_CHUNK_SIZE` | `20` | |
| `XTTS_OVERLAP_WAV_LEN` | `1024` | |

When `XTTS_COQUI_TOS_AGREED=true` (default), the launcher and server set
`COQUI_TOS_AGREED=1` automatically before model download/load.

## Output Formats

`wav`, `mp3`, `opus`, `aac`, `flac`, `pcm`. Non-WAV formats require `ffmpeg` on
your system PATH.

## Supported Languages

`ar`, `cs`, `de`, `en`, `es`, `fr`, `hi`, `hu`, `it`, `ja`, `ko`, `nl`, `pl`,
`pt`, `ru`, `tr`, `zh-cn`

## Streaming

Two streaming modes via `stream_format`:

- **`audio`** — raw audio chunks as they're generated (`Content-Type: audio/wav`)
- **`sse`** — OpenAI-compatible server-sent events (`data: {"type":"speech.audio.delta","audio":"<base64>"}`)

## Model Cache

The launcher pre-downloads the default model (~1.6 GB) into
`models/XTTS_2.0.2/` during install/start by default.
It prefers direct HuggingFace download into a temporary project folder, then
moves the result into `models/XTTS_2.0.2/` to avoid duplicate disk usage.
If direct download is skipped or fails, it falls back to coqui-tts download and
the server can still download on first inference.
Existing `models/v2.0.2/` is reused and copied forward automatically.
On startup, the launcher validates required model files (`config.json`,
`model.pth`, `speakers_xtts.pth`, `vocab.json`) and scans `models/` recursively
for a complete bundle before attempting a new download.

Custom models can be placed in `models/<model_id>/` with `config.json`,
`model.pth`, `speakers_xtts.pth`, and `vocab.json`.

## DeepSpeed

DeepSpeed 0.16.5 is automatically installed for CUDA backends. This project pins
PyTorch 2.6.x to match the available Windows DeepSpeed wheel build. NVIDIA
driver required; CUDA toolkit is not needed at runtime.

The server keeps DeepSpeed enabled by default on CUDA and automatically retries
checkpoint loading without DeepSpeed if initialization fails.

## Notes

- `/v1/files` can store any file type, but XTTS voice cloning works best with clean mono WAV references.
- Voice samples should be mono WAV files. Any sample rate is fine (resampled to 24kHz).
- Reference audio quality matters more than quantity. A single clean 6-second
  clip often beats a noisy 30-second clip.
- If CUDA runtime errors occur during speaker conditioning/inference, the server
  automatically retries on CPU for that model instance.
- `enable_text_splitting: false` (default) caps input to ~400 tokens but gives
  natural prosody across sentences. Set to `true` for long-form content.
