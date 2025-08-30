import os

from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
from typing import ClassVar

load_dotenv()


class MongoDBSettings(BaseSettings):
    MONGO_HOST: str = Field(..., alias="MONGO_HOST")
    MONGO_PORT: int = Field(..., alias="MONGO_PORT")
    MONGO_DB: str = Field(..., alias="MONGO_DB")
    MONGO_USER: str = Field(..., alias="MONGO_USER")
    MONGO_PASSWORD: str = Field(..., alias="MONGO_PASSWORD")


class IndexPathSettings(BaseSettings):
    FAISS_INDEX_PATH: str | None
    USEARCH_INDEX_PATH: str | None


class KeyFrameIndexMilvusSetting(BaseSettings):
    COLLECTION_NAME: str = "keyframe"
    HOST: str = "localhost"
    PORT: str = "19530"
    METRIC_TYPE: str = "COSINE"
    INDEX_TYPE: str = "FLAT"
    BATCH_SIZE: int = 10000
    SEARCH_PARAMS: dict = {}


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class AppSettings(BaseSettings):
    # DATA_FOLDER: str = "data/keyframes"
    # ID2INDEX_PATH: str = "data/id2index.json"
    # MODEL_NAME: str = "ViT-B-32"
    # FRAME2OBJECT: str = "data/detections.json"
    # ASR_PATH: str = "data/asr_proc.json"

    ROOT_DIR: ClassVar[str] = ROOT_DIR  # <-- đây là biến helper, không phải field

    DATA_FOLDER: str = os.path.join(ROOT_DIR, "data/keyframes")
    ID2INDEX_PATH: str = os.path.join(ROOT_DIR, "data/id2index.json")
    # MODEL_NAME: str = "hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup"  # hoặc model khác bạn dùng
    MODEL_NAME: str = "ViT-B-32"
    FRAME2OBJECT: str = os.path.join(ROOT_DIR, "data/detections.json")
    ASR_PATH: str = os.path.join(ROOT_DIR, "data/asr_proc.json")
