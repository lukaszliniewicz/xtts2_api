# XTTS FastAPI Server - Implementation Plan

## 1. Goals

- Build a FastAPI server for XTTS v2 (and compatible Coqui models) with OpenAI-compatible speech API shape for LiteLLM interoperability.
- Do **not** map OpenAI model aliases (`tts-1`, etc.) to internal models.
- Auto-discover local models by folder and expose them via `/v1/models`.
- Support uploaded voice references and expose them in `/v1/voices`, including filenames.
- Cross-platform setup for Windows/Linux/macOS with launcher/install scripts (`.bat` + `.sh`) that:
  - download Pixi into `bin/`
  - initialize environment
  - install dependencies
  - prompt for `cpu` or `cuda` (or accept `--cpu` / `--gpu`)
  - default to GPU installation mode

## 2. Non-goals (current phase)

- Fine-tuning endpoints/jobs (deferred).
- Auth/multi-tenant access control.
- Distributed/multi-node inference.

## 3. Platform & Backend Strategy

### 3.1 User-facing install modes
- Prompt mode: `cpu` or `cuda` (default `cuda`).
- Non-interactive: `--cpu` or `--gpu` (alias `--cuda`).

### 3.2 Backend resolution by platform
- **Windows**
  - `cuda` => install CUDA PyTorch wheels.
  - `cpu` => install CPU PyTorch wheels.
- **Linux**
  - `cuda` => install CUDA PyTorch if NVIDIA available.
  - If `cuda` selected but NVIDIA unavailable:
    - detect ROCm capability; fallback to ROCm (with warning), else CPU fallback.
- **macOS**
  - `cuda` selection is not available; map to:
    - Apple Silicon: PyTorch default (MPS-capable) with warning that CUDA is not supported.
    - Intel mac: CPU fallback with warning.
- DeepSpeed remains optional and CUDA-focused (no hard dependency in bootstrap).

## 4. Repo Scaffold

```text
xtts_fastapi/
  PLAN.md
  README.md
  .gitignore
  pixi.toml
  src/
    xtts_fastapi/
      __init__.py
      main.py
      settings.py
      api_models.py
      errors.py
      registry.py
      engine.py
      audio.py
      voices.py
      model_loader.py
  scripts/
    install.sh
    install.bat
    run.sh
    run.bat
  models/
    .gitkeep
  voices/
    .gitkeep
  tests/
    test_health.py
    test_models.py
    test_voices.py
    test_speech_contract.py
```

## 5. Pixi Environment Design

- Base dependencies in `pixi.toml`:
  - python 3.12
  - fastapi, uvicorn, pydantic, pydantic-settings
  - python-multipart
  - soundfile, numpy
  - ffmpeg (conda-forge)
  - coqui-tts
- Torch is installed by installer task (backend-specific), not blindly pinned in base env.
- Pixi tasks:
  - `serve`
  - `test`
  - `install-torch-cpu`
  - `install-torch-cuda`
  - `install-torch-rocm`
  - `check-runtime` (prints torch backend/device info)

## 6. Installer/Launcher Behavior

### 6.1 scripts/install.sh and scripts/install.bat
- Parse args: `--cpu`, `--gpu|--cuda`, `--yes`, `--force`.
- Detect OS/arch.
- Download Pixi binary into `bin/` from latest GitHub release.
- Run `pixi install`.
- Run backend-specific torch install task.
- Run `pixi run check-runtime`.
- Print next commands (`run` script, API URL).

### 6.2 run.sh and run.bat
- Ensure Pixi exists (call installer if missing).
- Start server via `pixi run serve`.

## 7. API Design

### 7.1 Endpoints
- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/speech`
- `GET /v1/voices`
- `POST /v1/voices`
- `DELETE /v1/voices/{voice_id}` (recommended)

### 7.2 `/v1/models`
- Auto-discovered model IDs from folders under `models/`.
- Model ID = folder name.
- Return OpenAI-style list object with local IDs.

### 7.3 `/v1/audio/speech` request contract
OpenAI-compatible core fields:
- `model`, `input`, `voice`, `response_format`, `speed`, `stream_format`, `instructions`
Extensions:
- `language` (top-level)
- `xtts` object for XTTS-specific controls

### 7.4 XTTS-specific parameters (`xtts`)
- Generation:
  - `temperature`
  - `top_p`
  - `top_k`
  - `repetition_penalty`
  - `length_penalty`
  - `do_sample`
  - `num_beams`
  - `enable_text_splitting`
- Voice conditioning:
  - `gpt_cond_len`
  - `gpt_cond_chunk_len`
  - `max_ref_len`
  - `sound_norm_refs`
- Streaming:
  - `stream_chunk_size`
  - `overlap_wav_len`

Validation:
- strict type/range checks
- OpenAI-style error payloads on invalid params

### 7.5 Voice model
- `POST /v1/voices` multipart upload of 1..N wav files.
- Persist under `voices/{voice_id}/`.
- Store metadata including source filenames.
- `GET /v1/voices` returns `voice_id`, file list, created time, optional model/language tags.
- Accept `voice` in speech requests as:
  - uploaded `voice_id`
  - known speaker name from model
  - filename alias (if unique), else ambiguity error

## 8. Model Discovery & Loading

- Discover valid model folder by presence of required files (`config.json`, checkpoint, vocab).
- Lazy-load model on first request.
- Keep in-memory cache keyed by model ID.
- Per-model lock for inference serialization safety.
- Optional latent cache for `(model_id, voice_id, conditioning params)`.

## 9. Streaming + Formats

- `stream_format=audio`: chunked binary stream.
- `stream_format=sse`: OpenAI-style SSE events.
- Output `response_format`: mp3, opus, aac, flac, wav, pcm.
- Use ffmpeg conversion pipeline for non-wav output.

## 10. Testing Matrix

- Unit tests:
  - model discovery
  - request validation
  - voice upload/list/delete contract
  - OpenAI-style error schema
- Smoke runs:
  - Windows + CUDA/CPU
  - Linux + CUDA and CPU (ROCm-capable path where available)
  - macOS Apple Silicon (MPS path), macOS Intel (CPU path)

## 11. Delivery Steps

1. Scaffold files and project structure.
2. Implement install/run scripts.
3. Implement settings + model registry.
4. Implement voice storage endpoints.
5. Implement speech endpoint + XTTS parameter bridge.
6. Implement streaming + format conversion.
7. Add tests and README usage examples.
