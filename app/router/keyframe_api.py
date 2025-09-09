from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
from pathlib import Path

from schema.request import (
    TextSearchRequest,
    TextSearchWithExcludeGroupsRequest,
    TextSearchWithSelectedGroupsAndVideosRequest,
    TrakeSearchRequest,
)
from schema.response import (
    KeyframeServiceReponse,
    SingleKeyframeDisplay,
    KeyframeDisplay,
    TrakeDisplay,
    TrakeItem,
)
from controller.query_controller import QueryController
from core.dependencies import get_query_controller
from core.logger import SimpleLogger


logger = SimpleLogger(__name__)


router = APIRouter(
    prefix="/keyframe",
    tags=["keyframe"],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/search",
    response_model=KeyframeDisplay,
    summary="Simple text search for keyframes",
    description="""
    Perform a simple text-based search for keyframes using semantic similarity.

    This endpoint converts the input text query to an embedding and searches for
    the most similar keyframes in the database.

    **Parameters:**
    - **query**: The search text (1-1000 characters)
    - **top_k**: Maximum number of results to return (1-100, default: 10)
    - **score_threshold**: Minimum confidence score (0.0-1.0, default: 0.0)

    **Returns:**
    List of keyframes with their metadata and confidence scores, ordered by similarity.

    **Example:**
    ```json
    {
        "query": "person walking in the park",
        "top_k": 5,
        "score_threshold": 0.7
    }
    ```
    """,
    response_description="List of matching keyframes with confidence scores",
)
async def search_keyframes(
    request: TextSearchRequest,
    controller: QueryController = Depends(get_query_controller),
):
    """
    Search for keyframes using text query with semantic similarity.
    """

    logger.info(
        f"Text search request: query='{request.query}', top_k={request.top_k}, threshold={request.score_threshold}"
    )

    results = await controller.search_text(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
    )

    logger.info(f"Found {len(results)} results for query: '{request.query}'")
    display_results = list(
        map(
            lambda pair: SingleKeyframeDisplay(path=pair[0], score=pair[1]),
            map(controller.convert_model_to_path, results),
        )
    )
    # NEW: export CSV top-100 theo yêu cầu
    export_path = controller._export_topk_csv(results, k=request.top_k)
    export_fname = Path(export_path).name
    return KeyframeDisplay(results=display_results, export_csv=export_fname)


@router.post(
    "/search/exclude-groups",
    response_model=KeyframeDisplay,
    summary="Text search with group exclusion",
    description="""
    Perform text-based search for keyframes while excluding specific groups.

    This endpoint allows you to search for keyframes while filtering out
    results from specified groups (e.g., to avoid certain video categories).

    **Parameters:**
    - **query**: The search text
    - **top_k**: Maximum number of results to return
    - **score_threshold**: Minimum confidence score
    - **exclude_groups**: List of group IDs to exclude from results

    **Use Cases:**
    - Exclude specific video categories or datasets
    - Filter out content from certain time periods
    - Remove specific collections from search results

    **Example:**
    ```json
    {
        "query": "sunset landscape",
        "top_k": 15,
        "score_threshold": 0.6,
        "exclude_groups": [1, 3, 7]
    }
    ```
    """,
    response_description="List of matching keyframes excluding specified groups",
)
async def search_keyframes_exclude_groups(
    request: TextSearchWithExcludeGroupsRequest,
    controller: QueryController = Depends(get_query_controller),
):
    """
    Search for keyframes with group exclusion filtering.
    """

    logger.info(
        f"Text search with group exclusion: query='{request.query}', exclude_groups={request.exclude_groups}"
    )

    results: list[KeyframeServiceReponse] = (
        await controller.search_text_with_exlude_group(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            list_group_exlude=request.exclude_groups,
        )
    )

    logger.info(
        f"Found {len(results)} results excluding groups {request.exclude_groups}"
    )

    display_results = list(
        map(
            lambda pair: SingleKeyframeDisplay(path=pair[0], score=pair[1]),
            map(controller.convert_model_to_path, results),
        )
    )

    export_path = controller._export_topk_csv(results, k=request.top_k)
    export_fname = Path(export_path).name
    return KeyframeDisplay(results=display_results, export_csv=export_fname)


@router.post(
    "/search/selected-groups-videos",
    response_model=KeyframeDisplay,
    summary="Text search within selected groups and videos",
    description="""
    Perform text-based search for keyframes within specific groups and videos only.

    This endpoint allows you to limit your search to specific groups and videos,
    effectively creating a filtered search scope.

    **Parameters:**
    - **query**: The search text
    - **top_k**: Maximum number of results to return
    - **score_threshold**: Minimum confidence score
    - **include_groups**: List of group IDs to search within
    - **include_videos**: List of video IDs to search within

    **Behavior:**
    - Only keyframes from the specified groups AND videos will be searched
    - If a keyframe belongs to an included group OR an included video, it will be considered
    - Empty lists mean no filtering for that category

    **Use Cases:**
    - Search within specific video collections
    - Focus on particular time periods or datasets
    - Limit search to curated content sets

    **Example:**
    ```json
    {
        "query": "car driving on highway",
        "top_k": 20,
        "score_threshold": 0.5,
        "include_groups": [2, 4, 6],
        "include_videos": [101, 102, 203, 204]
    }
    ```
    """,
    response_description="List of matching keyframes from selected groups and videos",
)
async def search_keyframes_selected_groups_videos(
    request: TextSearchWithSelectedGroupsAndVideosRequest,
    controller: QueryController = Depends(get_query_controller),
):
    """
    Search for keyframes within selected groups and videos.
    """

    logger.info(
        f"Text search with selection: query='{request.query}', include_groups={request.include_groups}, include_videos={request.include_videos}"
    )

    results = await controller.search_with_selected_video_group(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        list_of_include_groups=request.include_groups,
        list_of_include_videos=request.include_videos,
    )

    logger.info(f"Found {len(results)} results within selected groups/videos")

    display_results = list(
        map(
            lambda pair: SingleKeyframeDisplay(path=pair[0], score=pair[1]),
            map(controller.convert_model_to_path, results),
        )
    )

    export_path = controller._export_topk_csv(results, k=request.top_k)
    export_fname = Path(export_path).name
    return KeyframeDisplay(results=display_results, export_csv=export_fname)


@router.post(
    "/trake_search",
    response_model=TrakeDisplay,
    summary="TRAKE: Retrieval + Alignment using Beam Search",
    description="Nhận 1-5 sự kiện, trả về 1 keyframe cho mỗi sự kiện, tất cả thuộc cùng 1 video và theo thứ tự thời gian",
)
async def trake_search(
    request: TrakeSearchRequest,
    controller: QueryController = Depends(get_query_controller),
):
    seq = await controller.trake_search(
        events=request.events,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        max_kf_gap=request.max_kf_gap,
    )
    if not seq:
        return TrakeDisplay(video_group=-1, video_num=-1, results=[])

    vg = seq[0].group_num
    vn = seq[0].video_num
    items: list[TrakeItem] = []
    for i, kf in enumerate(seq):
        path, score = controller.convert_model_to_path(kf)
        items.append(
            TrakeItem(
                path=path,
                score=score,
                group_num=kf.group_num,
                video_num=kf.video_num,
                keyframe_num=kf.keyframe_num,
                stage_index=i,
            )
        )

    export_path = controller._export_topk_csv(seq, k=len(seq))
    export_fname = Path(export_path).name
    return TrakeDisplay(
        video_group=vg, video_num=vn, results=items, export_csv=export_fname
    )


@router.get("/download")
def download_csv(
    fname: str, controller: QueryController = Depends(get_query_controller)
):
    results_dir = Path(controller.app_settings.RESULT_DIR)
    safe_name = Path(fname).name
    full = results_dir / safe_name
    if not full.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        full,
        media_type="text/csv",
        filename=safe_name,
    )
