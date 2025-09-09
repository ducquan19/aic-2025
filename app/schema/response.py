from pydantic import BaseModel, Field
from typing import List, Optional


class KeyframeServiceReponse(BaseModel):
    key: int = Field(..., description="Keyframe key")
    video_num: int = Field(..., description="Video ID")
    group_num: int = Field(..., description="Group ID")
    keyframe_num: int = Field(..., description="Keyframe number")
    confidence_score: float = Field(..., description="Keyframe number")
    prefix: str = Field(default="L")


class SingleKeyframeDisplay(BaseModel):
    path: str
    score: float


class KeyframeDisplay(BaseModel):
    results: list[SingleKeyframeDisplay]
    export_csv: str | None = None


class TrakeItem(BaseModel):
    path: str
    score: float
    group_num: int
    video_num: int
    keyframe_num: int
    stage_index: int = Field(..., description="Vị trí sự kiện trong chuỗi (0-based)")


class TrakeDisplay(BaseModel):
    video_group: int
    video_num: int
    results: List[TrakeItem]
    export_csv: str | None = None
