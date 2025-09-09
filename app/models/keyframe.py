from beanie import Document, Indexed
from typing import Annotated
from pydantic import BaseModel, Field


class Keyframe(Document):
    key: Annotated[int, Indexed(unique=True)]
    video_num: Annotated[int, Indexed()]
    group_num: Annotated[int, Indexed()]
    keyframe_num: Annotated[int, Indexed()]
    prefix: str = Field(default="L")

    class Settings:
        name = "keyframes"
