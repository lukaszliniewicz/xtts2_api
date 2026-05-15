from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "xtts-fapi"


class ModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModel]


class FileObject(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: str = "processed"
    expires_at: int | None = None
    status_details: str | None = None


class FileListResponse(BaseModel):
    object: str = "list"
    data: list[FileObject]
    has_more: bool = False


class FileDeletedResponse(BaseModel):
    id: str
    object: str = "file"
    deleted: bool = True


class VoiceFile(BaseModel):
    filename: str
    size: int


class Voice(BaseModel):
    voice_id: str
    object: str = "voice"
    files: list[VoiceFile]
    created: int
    model: str | None = None
    language: str | None = None


class VoiceList(BaseModel):
    object: str = "list"
    data: list[Voice]


class VoiceCreateResponse(BaseModel):
    id: str
    object: str = "voice"
    model: str | None = None
    language: str | None = None
    sample_count: int
    created: int


class XTTSParams(BaseModel):
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    repetition_penalty: float | None = Field(default=None, ge=0.0, le=100.0)
    length_penalty: float | None = Field(default=None, ge=-10.0, le=10.0)
    do_sample: bool | None = None
    num_beams: int | None = Field(default=None, ge=1)
    enable_text_splitting: bool | None = None
    gpt_cond_len: int | None = Field(default=None, ge=1)
    gpt_cond_chunk_len: int | None = Field(default=None, ge=1)
    max_ref_length: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("max_ref_length", "max_ref_len"),
    )
    sound_norm_refs: bool | None = None
    librosa_trim_db: int | None = Field(default=None, ge=0)
    stream_chunk_size: int | None = Field(default=None, ge=1)
    overlap_wav_len: int | None = Field(default=None, ge=0)
    hf_generate_kwargs: dict[str, Any] | None = None


class VoiceIdentifier(BaseModel):
    id: str


class CreateSpeechRequest(BaseModel):
    model: str
    input: str = Field(..., max_length=4096)
    voice: str | VoiceIdentifier
    language: str = "en"
    response_format: str = Field(default="wav", pattern=r"^(mp3|opus|aac|flac|wav|pcm)$")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream_format: str | None = Field(default=None, pattern=r"^(audio|sse)$")
    instructions: str | None = Field(default=None, max_length=4096)
    xtts: XTTSParams | None = None
    speaker_wav: list[str] | None = None

    @property
    def voice_id(self) -> str:
        if isinstance(self.voice, VoiceIdentifier):
            return self.voice.id
        return self.voice
