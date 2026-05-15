from src.xtts_fastapi.api_models import XTTSParams
from src.xtts_fastapi.engine import engine


def test_inference_kwargs_exclude_voice_conditioning_fields():
    params = XTTSParams(
        temperature=0.55,
        gpt_cond_len=11,
        gpt_cond_chunk_len=3,
        max_ref_len=10,
        sound_norm_refs=True,
    )

    kwargs = engine._build_inference_kwargs(params, for_stream=False)

    assert kwargs["temperature"] == 0.55
    assert "gpt_cond_len" not in kwargs
    assert "gpt_cond_chunk_len" not in kwargs
    assert "max_ref_len" not in kwargs
    assert "sound_norm_refs" not in kwargs


def test_voice_kwargs_include_voice_conditioning_fields():
    params = XTTSParams(
        gpt_cond_len=11,
        gpt_cond_chunk_len=3,
        max_ref_len=10,
        sound_norm_refs=True,
    )

    kwargs = engine._build_voice_kwargs(params)

    assert kwargs["gpt_cond_len"] == 11
    assert kwargs["gpt_cond_chunk_len"] == 3
    assert kwargs["max_ref_len"] == 10
    assert kwargs["sound_norm_refs"] is True
