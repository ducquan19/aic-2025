import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)

from typing import List, cast
from llama_index.core.llms import LLM

from .agent import VisualEventExtractor, AnswerGenerator

from service.search_service import KeyframeQueryService
from service.model_service import ModelService
from schema.response import KeyframeServiceReponse


def apply_object_filter(
    keyframes: List[KeyframeServiceReponse],
    objects_data: dict[str, list[str]],
    target_objects: List[str],
) -> List[KeyframeServiceReponse]:

    if not target_objects:
        return keyframes

    target_objects_set = {obj.lower() for obj in target_objects}
    filtered_keyframes = []

    for kf in keyframes:
        keyy = f"L{kf.group_num:02d}/L{kf.group_num:02d}_V{kf.video_num:03d}/{kf.keyframe_num:03d}.jpg"
        keyframe_objects = objects_data.get(keyy, [])
        print(f"{keyy=}")
        print(f"{keyframe_objects=}")
        keyframe_objects_set = {obj.lower() for obj in keyframe_objects}

        if target_objects_set.intersection(keyframe_objects_set):
            filtered_keyframes.append(kf)

    print(f"{filtered_keyframes=}")
    return filtered_keyframes


class KeyframeSearchAgent:
    def __init__(
        self,
        llm: LLM,
        keyframe_service: KeyframeQueryService,
        model_service: ModelService,
        data_folder: str,
        objects_data: dict[str, list[str]],
        asr_data: dict[str, str | list[dict[str, str]]],
        top_k: int = 10,
    ):
        self.llm = llm
        self.keyframe_service = keyframe_service
        self.model_service = model_service
        self.data_folder = data_folder
        self.top_k = top_k

        self.objects_data = objects_data or {}
        self.asr_data = asr_data or {}

        self.query_extractor = VisualEventExtractor(llm)
        self.answer_generator = AnswerGenerator(llm, data_folder)

    async def process_query(self, user_query: str) -> str:
        """
        Main agent flow:
        1. Extract visual/event elements and rephrase query
        2. Search for top-K keyframes using rephrased query
        3. Score videos by averaging keyframe scores, select best video
        4. Optionally apply COCO object filtering
        5. Generate final answer with visual context
        """

        agent_response = await self.query_extractor.extract_visual_events(user_query)
        search_query = agent_response.refined_query
        suggested_objects = agent_response.list_of_objects

        print(f"{search_query=}")
        print(f"{suggested_objects=}")

        embedding = self.model_service.embedding(search_query).tolist()[0]
        top_k_keyframes = await self.keyframe_service.search_by_text(
            text_embedding=embedding, top_k=self.top_k, score_threshold=0.1
        )

        video_scores = self.query_extractor.calculate_video_scores(top_k_keyframes)
        _, best_video_keyframes = video_scores[0]

        final_keyframes = best_video_keyframes
        print(f"Length of keyframes before objects {len(final_keyframes)}")
        if suggested_objects:
            filtered_keyframes = apply_object_filter(
                keyframes=best_video_keyframes,
                objects_data=self.objects_data,
                target_objects=suggested_objects,
            )
            if filtered_keyframes:
                final_keyframes = filtered_keyframes
        print(f"Length of keyframes after objects {len(final_keyframes)}")

        smallest_kf = min(final_keyframes, key=lambda x: int(x.keyframe_num))
        max_kf = max(final_keyframes, key=lambda x: int(x.keyframe_num))

        print(f"{smallest_kf=}")
        print(f"{max_kf=}")

        group_num = smallest_kf.group_num
        video_num = smallest_kf.video_num

        print(f"{group_num}")
        print(f"{video_num}")
        print(f"L{group_num:02d}/L{group_num:02d}_V{video_num:03d}")

        answer = await self.answer_generator.generate_answer(
            original_query=user_query,
            final_keyframes=final_keyframes,
            objects_data=self.objects_data,
            # asr_data=asr_text
        )

        return cast(str, answer)
