from fastapi import APIRouter, Depends, HTTPException

from schema.agent import AgentQueryRequest, AgentQueryResponse
from controller.agent_controller import AgentController
from core.logger import SimpleLogger
from core.dependencies import get_agent_controller


router = APIRouter(
    prefix="/agent",
    tags=["agent"],
    responses={404: {"description": "Not found"}},
)
logger = SimpleLogger(__name__)


@router.post(
    "/search",
    response_model=AgentQueryResponse,
    summary="Intelligent keyframe search with AI agent",
    description="""
    Use the AI agent to search for keyframes and generate comprehensive answers.

    The agent will:
    1. Extract and rephrase visual elements from your query for better search
    2. Search for relevant keyframes using semantic similarity
    3. Score and select the best video based on keyframe relevance
    4. Optionally apply COCO object filtering to refine results
    5. Generate a comprehensive answer using visual context and metadata

    **Parameters:**
    - **query**: Natural language query describing what you're looking for

    **Example:**
    ```json
    {
        "query": "Show me scenes with people walking in a park during sunset"
    }
    ```
    """,
    response_description="AI-generated answer based on relevant keyframes",
)
async def agent_search(
    request: AgentQueryRequest,
    controller: AgentController = Depends(get_agent_controller),
):
    """Process natural language queries using the intelligent agent."""

    logger.info(f"Agent query request: '{request.query}'")

    # try:
    answer = await controller.search_and_answer(request.query)

    logger.info(f"Agent generated answer for query: '{request.query}'")

    return AgentQueryResponse(query=request.query, answer=answer)

    # except Exception as e:
    #     logger.error(f"Error processing agent query '{request.query}': {str(e)}")
    #     raise HTTPException(
    #         status_code=500,
    #         detail=f"Error processing query: {str(e)}"
    #     )
