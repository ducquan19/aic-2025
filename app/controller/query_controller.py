from pathlib import Path
from functools import lru_cache
from typing import List
import json

import os
import sys

from utils.map_index import n_to_frame_idx
import csv
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

sys.path.insert(0, ROOT_DIR)

from service import ModelService, KeyframeQueryService
from schema.response import KeyframeServiceReponse


class QueryController:

    def __init__(
        self,
        data_folder: Path,
        id2index_path: Path,
        model_service: ModelService,
        keyframe_service: KeyframeQueryService,
    ):
        self.data_folder = data_folder
        self.id2index = json.load(open(id2index_path, "r"))
        self.model_service = model_service
        self.keyframe_service = keyframe_service

        # New
        from core.settings import AppSettings

        self.app_settings = AppSettings()
        os.makedirs(self.app_settings.RESULT_DIR, exist_ok=True)

    def _video_name(self, prefix: str, group_num: int, video_num: int) -> str:
        return f"{prefix}{group_num:02d}_V{video_num:03d}"

    def _export_topk_csv(
        self, items: list[KeyframeServiceReponse], k: int = 100
    ) -> str:
        """
        Ghi file CSV dạng: <video_name>, <frame_idx>
        video_name: 'Lxx_Vyyy'
        frame_idx: lấy từ data/map-keyframes/Lxx_Vyyy.csv, map cột 'n' == keyframe_num.
        """
        out_path = os.path.join(
            self.app_settings.RESULT_DIR,
            f"query_top{min(k, len(items))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        rows = []
        for kf in items[:k]:
            video_name = self._video_name(kf.prefix, kf.group_num, kf.video_num)
            frame_idx = n_to_frame_idx(
                self.app_settings.MAP_KEYFRAME_DIR,
                kf.prefix,
                kf.group_num,
                kf.video_num,
                kf.keyframe_num,
            )
            if frame_idx is None:
                # fallback (nếu thiếu mapping), có thể ghi -1
                frame_idx = -1
            rows.append((video_name, frame_idx))

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # writer.writerow(["video_name", "frame_idx"])
            writer.writerows(rows)
        return out_path

    # --------

    def convert_model_to_path(self, model: KeyframeServiceReponse) -> tuple[str, float]:
        return (
            os.path.join(
                self.data_folder,
                f"{model.prefix}{model.group_num:02d}/{model.prefix}{model.group_num:02d}_V{model.video_num:03d}/{model.keyframe_num:03d}.jpg",
            ),
            model.confidence_score,
        )

    async def search_text(self, query: str, top_k: int, score_threshold: float):
        embedding = self.model_service.embedding(query).tolist()[0]

        result = await self.keyframe_service.search_by_text(
            embedding, top_k, score_threshold
        )
        return result

    async def search_text_with_exlude_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_group_exlude: list[int],
    ):
        exclude_ids = [
            int(k)
            for k, v in self.id2index.items()
            if int(v.split("/")[0]) in list_group_exlude
        ]

        embedding = self.model_service.embedding(query).tolist()[0]
        result = await self.keyframe_service.search_by_text_exclude_ids(
            embedding, top_k, score_threshold, exclude_ids
        )
        return result

    async def search_with_selected_video_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_of_include_groups: list[int],
        list_of_include_videos: list[int],
    ):

        exclude_ids = None
        if len(list_of_include_groups) > 0 and len(list_of_include_videos) == 0:
            print("hi")
            exclude_ids = [
                int(k)
                for k, v in self.id2index.items()
                if int(v.split("/")[0]) not in list_of_include_groups
            ]

        elif len(list_of_include_groups) == 0 and len(list_of_include_videos) > 0:
            exclude_ids = [
                int(k)
                for k, v in self.id2index.items()
                if int(v.split("/")[1]) not in list_of_include_videos
            ]

        elif len(list_of_include_groups) == 0 and len(list_of_include_videos) == 0:
            exclude_ids = []
        else:
            exclude_ids = [
                int(k)
                for k, v in self.id2index.items()
                if (
                    int(v.split("/")[0]) not in list_of_include_groups
                    or int(v.split("/")[1]) not in list_of_include_videos
                )
            ]

        embedding = self.model_service.embedding(query).tolist()[0]
        result = await self.keyframe_service.search_by_text_exclude_ids(
            embedding, top_k, score_threshold, exclude_ids
        )
        return result

    @lru_cache(maxsize=512)
    def _load_map_for_video(
        self, prefix: str, group_num: int, video_num: int
    ) -> dict[int, int]:
        """
        Đọc file data/map-keyframes/L{group:02d}_V{video:03d}.csv
        trả về {keyframe_num -> frame_idx}
        """
        fname = f"{prefix}{group_num:02d}_V{video_num:03d}.csv"
        map_path = os.path.join(self.data_folder.parent, "map-keyframes", fname)
        mapping: dict[int, int] = {}
        if os.path.exists(map_path):
            with open(map_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # file mẫu có cột n, pts_time, fps, frame_idx
                for row in reader:
                    n = int(row["n"])
                    mapping[n] = int(float(row["frame_idx"]))
        return mapping

    def _frame_idx_of(
        self, prefix: str, group_num: int, video_num: int, keyframe_num: int
    ) -> int | None:
        mp = self._load_map_for_video(prefix, group_num, video_num)
        return mp.get(keyframe_num)

    async def trake_search(
        self,
        events: List[str],
        top_k: int,
        score_threshold: float,
        max_kf_gap: int,
    ) -> List[KeyframeServiceReponse]:
        # embed từng stage
        stage_embeddings = [
            self.model_service.embedding(ev).tolist()[0] for ev in events
        ]
        seq = await self.keyframe_service.trake_beam_search(
            stage_embeddings=stage_embeddings,
            beam_width=top_k,
            score_threshold=score_threshold,
            max_kf_gap=max_kf_gap,
        )
        return seq
