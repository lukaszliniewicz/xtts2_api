# xtts2_api

OpenAI-compatible TTS server using Coqui XTTS v2. Managed by [pixi](https://pixi.sh).

## Install

```bash
# Windows
scripts\install.bat --cuda

# Linux (CUDA)
./scripts/install.sh --cuda

# Linux (CPU/ROCm)
./scripts/install.sh --cpu
```

Installer: downloads pixi → creates Python 3.12 env → installs torch + coqui-tts + deepspeed.

## Run

```bash
scripts\run.bat    # Windows
./scripts/run.sh   # Linux/macOS
```

Server starts on `http://0.0.0.0:8020`.

## Usage

### Upload a voice
```bash
curl -X POST http://localhost:8020/v1/voices \
  -F "files=@sample.wav" \
  -F "voice_id=my_voice"
```

### Generate speech
```bash
curl -X POST http://localhost:8020/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "v2.0.2",
    "input": "Hello, this is a test.",
    "voice": "my_voice",
    "language": "en",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### Stream (SSE)
```bash
curl -N http://localhost:8020/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "v2.0.2",
    "input": "Hello world",
    "voice": "my_voice",
    "stream_format": "sse"
  }'
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status |
| GET | `/v1/models` | List models |
| GET | `/v1/voices` | List uploaded voices |
| POST | `/v1/voices` | Upload voice samples |
| DELETE | `/v1/voices/{id}` | Delete voice |
| POST | `/v1/audio/speech` | Generate speech |

## Models

Place XTTS v2 weights in `models/<model_id>/`:

```
models/
  v2.0.2/
    config.json
    model.pth
    speakers_xtts.pth
    vocab.json
```

Model ID = folder name. The registry scans this directory at startup.

## XTTS Parameters

Pass via `xtts` object in the speech request:

```json
{
  "xtts": {
    "temperature": 0.85,
    "top_p": 0.85,
    "top_k": 50,
    "repetition_penalty": 2.0,
    "length_penalty": 1.0,
    "enable_text_splitting": true
  }
}
```

## Configuration

Set via env vars with `XTTS_` prefix or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `XTTS_HOST` | `0.0.0.0` | Bind address |
| `XTTS_PORT` | `8020` | Port |
| `XTTS_DEVICE` | `cuda` | `cuda`, `cpu`, `mps` |
| `XTTS_USE_DEEPSPEED` | `true` | Enable DeepSpeed (Windows pre-built wheel) |
| `XTTS_COQUI_TOS_AGREED` | `false` | Accept Coqui terms |

## Supported Languages

`ar`, `cs`, `de`, `en`, `es`, `fr`, `hi`, `hu`, `it`, `ja`, `ko`, `nl`, `pl`, `pt`, `ru`, `tr`, `zh-cn`

## Format Conversion

Output formats: `wav`, `mp3`, `opus`, `aac`, `flac`, `pcm`. Requires `ffmpeg` for non-WAV formats.

## Notes

- DeepSpeed 0.16.5 ships pre-built Windows cp312 wheel. CUDA toolkit is not required at runtime — only the NVIDIA driver.
- Voice samples should be WAV files, mono, any sample rate (resampled to 24kHz).
