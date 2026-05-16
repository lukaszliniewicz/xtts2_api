import numpy as np
import pytest
import soundfile as sf
import torch

import src.xtts_fastapi.model_loader as model_loader
from src.xtts_fastapi.api_models import XTTSParams
from src.xtts_fastapi.engine import engine
from src.xtts_fastapi.errors import APIError
from src.xtts_fastapi.model_loader import XTTSWrapper


def test_inference_kwargs_exclude_voice_conditioning_fields():
    params = XTTSParams(
        temperature=0.55,
        gpt_cond_len=11,
        gpt_cond_chunk_len=3,
        max_ref_length=10,
        sound_norm_refs=True,
    )

    kwargs = engine._build_inference_kwargs(params, for_stream=False)

    assert kwargs["temperature"] == 0.55
    assert "gpt_cond_len" not in kwargs
    assert "gpt_cond_chunk_len" not in kwargs
    assert "max_ref_length" not in kwargs
    assert "sound_norm_refs" not in kwargs


def test_voice_kwargs_include_voice_conditioning_fields():
    params = XTTSParams(
        gpt_cond_len=11,
        gpt_cond_chunk_len=3,
        max_ref_length=10,
        sound_norm_refs=True,
    )

    kwargs = engine._build_voice_kwargs(params)

    assert kwargs["gpt_cond_len"] == 11
    assert kwargs["gpt_cond_chunk_len"] == 3
    assert kwargs["max_ref_length"] == 10
    assert kwargs["sound_norm_refs"] is True


def test_xtts_params_accepts_legacy_max_ref_len_alias():
    params = XTTSParams(max_ref_len=9)
    assert params.max_ref_length == 9


def test_xtts_wrapper_cuda_error_detection():
    assert XTTSWrapper._is_cuda_runtime_error(RuntimeError("CUDA error: unknown error"))
    assert XTTSWrapper._is_cuda_runtime_error(RuntimeError("cuDNN_STATUS_EXECUTION_FAILED"))
    assert XTTSWrapper._is_cuda_runtime_error(RuntimeError("nvrtc: error: failed to load builtins"))
    assert XTTSWrapper._is_cuda_runtime_error(
        RuntimeError(
            "RuntimeError:\n#ifdef __HIPCC__\n"
            "#define CUDA_OR_ROCM_NUM_THREADS 256\n"
            "extern \"C\" __global__ void abs_kernel_vectorized4_kernel()"
        )
    )
    assert not XTTSWrapper._is_cuda_runtime_error(RuntimeError("plain value error"))


def test_get_conditioning_latents_retries_after_cuda_error(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.calls = 0

        def get_conditioning_latents(self, audio_path, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("CUDA error: unknown error")
            return "gpt", "spk"

    wrapper = XTTSWrapper.__new__(XTTSWrapper)
    wrapper.device = "cuda"
    wrapper.use_deepspeed = True
    wrapper.model_info = None
    wrapper.xtts_model = FakeModel()
    wrapper._speaker_manager = None
    wrapper._loaded = True

    fallback_count = {"value": 0}

    def fake_fallback(exc, stage):
        fallback_count["value"] += 1
        wrapper.device = "cpu"
        return True

    monkeypatch.setattr(wrapper, "load", lambda: None)
    monkeypatch.setattr(wrapper, "_fallback_to_cpu", fake_fallback)
    monkeypatch.setattr(wrapper, "_supported_kwargs", lambda fn, kwargs: kwargs)

    gpt, spk = wrapper.get_conditioning_latents(["sample.wav"], gpt_cond_len=12)

    assert (gpt, spk) == ("gpt", "spk")
    assert wrapper.xtts_model.calls == 2
    assert fallback_count["value"] == 1


def test_synthesize_retries_after_cuda_error(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.calls = 0

        def inference(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("CUDA error: unknown error")
            return {"wav": [0.0, 0.1]}

    wrapper = XTTSWrapper.__new__(XTTSWrapper)
    wrapper.device = "cuda"
    wrapper.use_deepspeed = True
    wrapper.model_info = None
    wrapper.xtts_model = FakeModel()
    wrapper._speaker_manager = None
    wrapper._loaded = True

    fallback_count = {"value": 0}

    def fake_fallback(exc, stage):
        fallback_count["value"] += 1
        wrapper.device = "cpu"
        return True

    monkeypatch.setattr(wrapper, "load", lambda: None)
    monkeypatch.setattr(wrapper, "_fallback_to_cpu", fake_fallback)
    monkeypatch.setattr(wrapper, "_supported_kwargs", lambda fn, kwargs: kwargs)

    result = wrapper.synthesize(
        text="hello",
        language="en",
        gpt_cond_latent=torch.tensor([1.0]),
        speaker_embedding=torch.tensor([2.0]),
        temperature=0.6,
    )

    assert result["wav"] == [0.0, 0.1]
    assert wrapper.xtts_model.calls == 2
    assert fallback_count["value"] == 1


def test_synthesize_stream_retries_after_cuda_error(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.calls = 0

        def inference_stream(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("CUDA error: unknown error")
            yield torch.tensor([0.0, 0.1])

    wrapper = XTTSWrapper.__new__(XTTSWrapper)
    wrapper.device = "cuda"
    wrapper.use_deepspeed = True
    wrapper.model_info = None
    wrapper.xtts_model = FakeModel()
    wrapper._speaker_manager = None
    wrapper._loaded = True

    fallback_count = {"value": 0}

    def fake_fallback(exc, stage):
        fallback_count["value"] += 1
        wrapper.device = "cpu"
        return True

    monkeypatch.setattr(wrapper, "load", lambda: None)
    monkeypatch.setattr(wrapper, "_fallback_to_cpu", fake_fallback)
    monkeypatch.setattr(wrapper, "_supported_kwargs", lambda fn, kwargs: kwargs)

    chunks = list(
        wrapper.synthesize_stream(
            text="hello",
            language="en",
            gpt_cond_latent=torch.tensor([1.0]),
            speaker_embedding=torch.tensor([2.0]),
            temperature=0.6,
        )
    )

    assert len(chunks) == 1
    assert wrapper.xtts_model.calls == 2
    assert fallback_count["value"] == 1


def test_reference_audio_short_path_rejected(tmp_path):
    short_wav = tmp_path / "short.wav"
    sf.write(short_wav, np.zeros(2400, dtype=np.float32), 24000)

    with pytest.raises(APIError) as exc:
        engine._validate_reference_audio_paths([str(short_wav)])

    assert exc.value.code == "reference_audio_too_short"


def test_load_checkpoint_falls_back_when_deepspeed_fails(monkeypatch):
    class FakeModel:
        def __init__(self, fail_on_deepspeed=False):
            self.fail_on_deepspeed = fail_on_deepspeed
            self.calls = []

        def load_checkpoint(self, config, checkpoint_dir, use_deepspeed):
            self.calls.append(use_deepspeed)
            if use_deepspeed and self.fail_on_deepspeed:
                raise RuntimeError("deepspeed failed")

    wrapper = XTTSWrapper.__new__(XTTSWrapper)
    wrapper.device = "cuda"
    wrapper.use_deepspeed = True

    first_model = FakeModel(fail_on_deepspeed=True)
    fallback_model = FakeModel()

    fake_xtts = type(
        "FakeXtts",
        (),
        {"init_from_config": staticmethod(lambda cfg: fallback_model)},
    )
    monkeypatch.setattr(model_loader, "Xtts", fake_xtts)

    loaded_model = wrapper._load_checkpoint_with_fallback(
        model=first_model,
        config=object(),
        checkpoint_dir=".",
    )

    assert loaded_model is fallback_model
    assert first_model.calls == [True]
    assert fallback_model.calls == [False]
    assert wrapper.use_deepspeed is False
