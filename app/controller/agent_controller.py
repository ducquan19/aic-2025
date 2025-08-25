import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)

from typing import Dict, List, Optional
from pathlib import Path
import json

from agent.main_agent import KeyframeSearchAgent
from service.search_service import KeyframeQueryService
from service.model_service import ModelService
from llama_index.core.llms import LLM


class AgentController:

    def __init__(
        self,
        llm: LLM,
        keyframe_service: KeyframeQueryService,
        model_service: ModelService,
        data_folder: str,
        objects_data_path: Optional[Path] = None,
        asr_data_path: Optional[Path] = None,
        top_k: int = 200,
    ):

        objects_data = (
            self._load_json_data(objects_data_path) if objects_data_path else {}
        )
        asr_data = self._load_json_data(asr_data_path) if asr_data_path else {}

        self.agent = KeyframeSearchAgent(
            llm=llm,
            keyframe_service=keyframe_service,
            model_service=model_service,
            data_folder=data_folder,
            objects_data=objects_data,
            asr_data=asr_data,
            top_k=top_k,
        )

    def _load_json_data(self, path: Path):
        return json.load(open(path))

    async def search_and_answer(self, user_query: str) -> str:
        return await self.agent.process_query(user_query)
