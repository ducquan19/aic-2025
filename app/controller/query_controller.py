from pathlib import Path
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

    def _video_name(self, group_num: int, video_num: int) -> str:
        return f"L{group_num:02d}_V{video_num:03d}"

    def _export_topk_csv(self, items: list[KeyframeServiceReponse], k: int = 100) -> str:
        """
        Ghi file CSV dạng: <video_name>, <frame_idx>
        video_name: 'Lxx_Vyyy'
        frame_idx: lấy từ data/map-keyframes/Lxx_Vyyy.csv, map cột 'n' == keyframe_num.
        """
        out_path = os.path.join(
            self.app_settings.RESULT_DIR,
            f"query_top{min(k, len(items))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        rows = []
        for kf in items[:k]:
            video_name = self._video_name(kf.group_num, kf.video_num)
            frame_idx = n_to_frame_idx(
                self.app_settings.MAP_KEYFRAME_DIR, kf.group_num, kf.video_num, kf.keyframe_num
            )
            if frame_idx is None:
                # fallback (nếu thiếu mapping), có thể ghi -1
                frame_idx = -1
            rows.append((video_name, frame_idx))

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_name", "frame_idx"])
            writer.writerows(rows)
        return out_path

    # --------

    def convert_model_to_path(self, model: KeyframeServiceReponse) -> tuple[str, float]:
        return (
            os.path.join(
                self.data_folder,
                f"L{model.group_num:02d}/L{model.group_num:02d}_V{model.video_num:03d}/{model.keyframe_num:03d}.jpg",
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
