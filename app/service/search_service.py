import os
import sys

from typing import List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)


from repository.milvus import KeyframeVectorRepository
from repository.milvus import MilvusSearchRequest
from repository.mongo import KeyframeRepository

from schema.response import KeyframeServiceReponse


class KeyframeQueryService:
    def __init__(
        self,
        keyframe_vector_repo: KeyframeVectorRepository,
        keyframe_mongo_repo: KeyframeRepository,
    ):

        self.keyframe_vector_repo = keyframe_vector_repo
        self.keyframe_mongo_repo = keyframe_mongo_repo

    async def _retrieve_keyframes(self, ids: list[int]):
        keyframes = await self.keyframe_mongo_repo.get_keyframe_by_list_of_keys(ids)
        print(keyframes[:5])

        keyframe_map = {k.key: k for k in keyframes}
        return_keyframe = [keyframe_map[k] for k in ids]
        return return_keyframe

    async def _search_keyframes(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None = None,
        exclude_indices: list[int] | None = None,
    ) -> list[KeyframeServiceReponse]:

        search_request = MilvusSearchRequest(
            embedding=text_embedding, top_k=top_k, exclude_ids=exclude_indices
        )

        search_response = await self.keyframe_vector_repo.search_by_embedding(
            search_request
        )

        filtered_results = [
            result
            for result in search_response.results
            if score_threshold is None or result.distance > score_threshold
        ]

        sorted_results = sorted(
            filtered_results, key=lambda r: r.distance, reverse=True
        )

        sorted_ids = [result.id_ for result in sorted_results]

        keyframes = await self._retrieve_keyframes(sorted_ids)

        keyframe_map = {k.key: k for k in keyframes}
        response = []

        for result in sorted_results:
            keyframe = keyframe_map.get(result.id_)
            if keyframe is not None:
                response.append(
                    KeyframeServiceReponse(
                        key=keyframe.key,
                        video_num=keyframe.video_num,
                        group_num=keyframe.group_num,
                        keyframe_num=keyframe.keyframe_num,
                        confidence_score=result.distance,
                        prefix=keyframe.prefix if hasattr(keyframe, "prefix") else "L",
                    )
                )
        return response

    async def search_by_text(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None = 0.5,
    ):
        return await self._search_keyframes(
            text_embedding, top_k, score_threshold, None
        )

    async def search_by_text_range(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        range_queries: list[tuple[int, int]],
    ):
        """
        range_queries: a bunch of start end indices, and we just search inside these, ignore everything
        """

        all_ids = self.keyframe_vector_repo.get_all_id()
        allowed_ids = set()
        for start, end in range_queries:
            allowed_ids.update(range(start, end + 1))

        exclude_ids = [id_ for id_ in all_ids if id_ not in allowed_ids]

        return await self._search_keyframes(
            text_embedding, top_k, score_threshold, exclude_ids
        )

    async def search_by_text_exclude_ids(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        exclude_ids: list[int] | None,
    ):
        """
        range_queries: a bunch of start end indices, and we just search inside these, ignore everything
        """
        return await self._search_keyframes(
            text_embedding, top_k, score_threshold, exclude_ids
        )

    async def trake_beam_search(
        self,
        stage_embeddings: List[List[float]],
        beam_width: int = 50,
        score_threshold: float = 0.0,
        max_kf_gap: int = 200,
    ) -> List[KeyframeServiceReponse]:
        """
        Beam search qua chuỗi sự kiện:
        - Giai đoạn 1: lấy top-k keyframe
        - Các giai đoạn sau: chỉ giữ ứng viên cùng video với phần tử trước,
          keyframe_num tăng dần và chênh lệch <= max_kf_gap.
        Trả về chuỗi keyframe tốt nhất (1 phần tử cho mỗi sự kiện).
        """
        if not stage_embeddings:
            return []

        # stage 1
        first = await self._search_keyframes(
            text_embedding=stage_embeddings[0],
            top_k=min(beam_width, 100),
            score_threshold=score_threshold,
            exclude_indices=None,
        )
        if not first:
            return []

        # beam = [(tổng_điểm, [seq])]
        beams: List[Tuple[float, List[KeyframeServiceReponse]]] = [
            (r.confidence_score, [r]) for r in first
        ]

        # các stage tiếp theo
        for s in range(1, len(stage_embeddings)):
            emb = stage_embeddings[s]
            # mở rộng ứng viên rộng hơn trước khi lọc
            candidates = await self._search_keyframes(
                text_embedding=emb,
                top_k=min(beam_width * 20, 200),
                score_threshold=score_threshold,
                exclude_indices=None,
            )
            next_beams: List[Tuple[float, List[KeyframeServiceReponse]]] = []

            for total, seq in beams:
                last = seq[-1]
                for c in candidates:
                    if (c.group_num != last.group_num) or (
                        c.video_num != last.video_num
                    ):
                        continue
                    if int(c.keyframe_num) <= int(last.keyframe_num):
                        continue
                    if int(c.keyframe_num) - int(last.keyframe_num) > max_kf_gap:
                        continue
                    next_beams.append((total + c.confidence_score, seq + [c]))

            # giữ lại beam tốt nhất
            next_beams.sort(key=lambda x: x[0], reverse=True)
            beams = next_beams[:beam_width]
            if not beams:
                return []

        # chọn best
        _, best_seq = max(beams, key=lambda x: x[0])
        return best_seq
