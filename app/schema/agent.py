from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    refined_query: str = Field(..., description="The rephrased response")
    list_of_objects: list[str] | None = Field(
        None,
        description="The list of objects for filtering(Object from coco class), optionall",
    )


class AgentQueryRequest(BaseModel):
    """Request model for agent queries"""

    query: str = Field(
        ..., description="Natural language query", min_length=1, max_length=2000
    )


class AgentQueryResponse(BaseModel):
    """Response model for agent queries"""

    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
