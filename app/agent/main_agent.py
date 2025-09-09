import os
import sys
import numpy as np

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
        keyy = f"{kf.prefix}{kf.group_num:02d}/{kf.prefix}{kf.group_num:02d}_V{kf.video_num:03d}/{kf.keyframe_num:03d}.jpg"
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
        asr_data: dict[str, str | dict],
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

    async def process_query1(self, user_query: str) -> str:
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

        prefix = smallest_kf.prefix
        group_num = smallest_kf.group_num
        video_num = smallest_kf.video_num

        print(f"{group_num}")
        print(f"{video_num}")
        print(f"{prefix}{group_num:02d}/{prefix}{group_num:02d}_V{video_num:03d}")

        answer = await self.answer_generator.generate_answer(
            original_query=user_query,
            final_keyframes=final_keyframes,
            objects_data=self.objects_data,
            asr_data=self.asr_data,
        )

        return cast(str, answer)

    async def process_query(self, user_query: str) -> str:
        agent_response = await self.query_extractor.extract_visual_events(user_query)
        search_query = agent_response.refined_query
        suggested_objects = agent_response.list_of_objects

        # Embed 1 lần cho query dùng lại
        q_emb = self.model_service.embedding(search_query).tolist()[0]

        top_k_keyframes = await self.keyframe_service.search_by_text(
            text_embedding=q_emb, top_k=self.top_k, score_threshold=0.1
        )

        # Tính điểm theo VIDEO (visual_avg) như cũ
        video_scores = self.query_extractor.calculate_video_scores(top_k_keyframes)

        # Kết hợp với ASR
        TOP_VIDEOS = 30
        alpha = 0.7  # trọng số ảnh; 0.3 cho ASR
        best_video_keyframes = None
        best_final = -1.0

        # Đánh giá top 10 video đầu tiên đủ rồi (tối ưu tốc độ)
        for vis_avg, kfs in video_scores[:TOP_VIDEOS]:
            p = kfs[0].prefix
            g = kfs[0].group_num
            v = kfs[0].video_num
            asr_text = self._get_asr_text_for_video(p, g, v)[:2000]

            asr_sim = 0.0
            if asr_text:
                # Cắt gọn để tránh tokenizer CLIP truncate quá dài
                asr_emb = self.model_service.embedding(asr_text).tolist()[0]
                asr_sim = self._cosine(np.array(q_emb), np.array(asr_emb))

            final_score = alpha * vis_avg + (1 - alpha) * asr_sim
            if final_score > best_final:
                best_final = final_score
                best_video_keyframes = kfs

        final_keyframes = best_video_keyframes or video_scores[0][1]

        # Lọc theo COCO object nếu agent gợi ý
        if suggested_objects:
            filtered_keyframes = apply_object_filter(
                keyframes=final_keyframes,
                objects_data=self.objects_data,
                target_objects=suggested_objects,
            )
            if filtered_keyframes:
                final_keyframes = filtered_keyframes

        answer = await self.answer_generator.generate_answer(
            original_query=user_query,
            final_keyframes=final_keyframes,
            objects_data=self.objects_data,
            asr_data=self.asr_data,  # <-- TRUYỀN ASR VÀO PROMPT
        )

        return cast(str, answer)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a, b))

    def _get_asr_text_for_video(
        self, prefix: str, group_num: int, video_num: int
    ) -> str:
        key_mp4 = f"{prefix}{group_num:02d}_V{video_num:03d}.mp4"
        rec = self.asr_data.get(key_mp4, None)
        if isinstance(rec, dict):
            return (rec.get("asr_clean") or rec.get("asr_raw") or "").strip()
        elif isinstance(rec, str):
            return rec.strip()
        return ""

    # def _get_ocr_texts_for_video(self, g: int, v: int, kfs: list) -> list[str]:
    #     """
    #     Thu OCR text theo từng keyframe của video ứng viên.
    #     Key trong OCR map là 'Lxx/Lxx_Vyyy/nnn.jpg'
    #     """
    #     texts = []
    #     for kf in kfs:
    #         img_key = f"L{g:02d}/L{g:02d}_V{v:03d}/{kf.keyframe_num:03d}.jpg"
    #         txt = self.ocr_data.get(img_key, "")
    #         if txt:
    #             texts.append(txt)
    #     return texts
