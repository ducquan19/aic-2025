from pydantic import BaseModel, Field, conlist
from typing import List, Optional


class BaseSearchRequest(BaseModel):
    """Base search request with common parameters"""

    query: str = Field(
        ..., description="Search query text", min_length=1, max_length=1000
    )
    top_k: int = Field(
        default=10, ge=1, le=500, description="Number of top results to return"
    )
    score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence score threshold"
    )


class TextSearchRequest(BaseSearchRequest):
    """Simple text search request"""

    pass


class TextSearchWithExcludeGroupsRequest(BaseSearchRequest):
    """Text search request with group exclusion"""

    exclude_groups: List[int] = Field(
        default_factory=list,
        description="List of group IDs to exclude from search results",
    )


class TextSearchWithSelectedGroupsAndVideosRequest(BaseSearchRequest):
    """Text search request with specific group and video selection"""

    include_groups: List[int] = Field(
        default_factory=list,
        description="List of group IDs to include in search results",
    )
    include_videos: List[int] = Field(
        default_factory=list,
        description="List of video IDs to include in search results",
    )


class TrakeSearchRequest(BaseModel):
    """Temporal Retrieval and Alignment of Key Events (TRAKE)"""

    events: List[str] = Field(
        ..., min_length=1, max_length=5, description="search queries"
    )
    top_k: int = Field(
        default=10, ge=1, le=500, description="Number of top results to return"
    )
    score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence score threshold"
    )
    max_kf_gap: int = Field(
        default=200,
        ge=1,
        description="Giới hạn chênh lệch keyframes"
    )
