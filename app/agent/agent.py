import re
from typing import cast
from llama_index.core.llms import LLM
from llama_index.core import PromptTemplate
from schema.agent import AgentResponse
from pathlib import Path

from typing import Dict, List, Tuple
from collections import defaultdict
from schema.response import KeyframeServiceReponse
import os
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock, MessageRole


COCO_CLASS = """
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
"""


class VisualEventExtractor:

    def __init__(self, llm: LLM):
        self.llm = llm
        self.extraction_prompt = PromptTemplate(
            """
            Extract visual elements and events from the following query.
            Focus on concrete, searchable visual descriptions and actions.

            COCO: {coco}
            Query: {query}


            Please extract the key visual elements, and events/actions within a query. And then rephrase them in a way that it is effective for embedding search. And Optionally, based on the original query, if you think we SHOULD use these COCO objects as the last filter to narrow down the search keyframes and have a better answer for the user questions, then feel free to provide a list!
            No explanation, just the rephrased query, and the optional list of coco class
            """
        )

    async def extract_visual_events(self, query: str) -> AgentResponse:
        prompt = self.extraction_prompt.format(query=query, coco=COCO_CLASS)
        response = await self.llm.as_structured_llm(AgentResponse).acomplete(prompt)
        obj = cast(AgentResponse, response.raw)
        return obj

    @staticmethod
    def calculate_video_scores(
        keyframes: List[KeyframeServiceReponse],
    ) -> List[Tuple[float, List[KeyframeServiceReponse]]]:
        """
        Calculate average scores for each video and return sorted by score

        Returns:
            List of tuples: (video_num, average_score, keyframes_in_video)
        """
        video_keyframes: Dict[str, List[KeyframeServiceReponse]] = defaultdict(list)

        for keyframe in keyframes:
            video_keyframes[f"{keyframe.group_num}/{keyframe.video_num}"].append(
                keyframe
            )

        video_scores: List[Tuple[float, List[KeyframeServiceReponse]]] = []
        for _, video_keyframes_list in video_keyframes.items():
            avg_score = sum(kf.confidence_score for kf in video_keyframes_list) / len(
                video_keyframes_list
            )
            video_scores.append((avg_score, video_keyframes_list))

        video_scores.sort(key=lambda x: x[0], reverse=True)

        return video_scores


class AnswerGenerator:
    """Generates final answers based on refined keyframes"""

    def __init__(self, llm: LLM, data_folder: str):
        self.data_folder = data_folder
        self.llm = llm
        self.answer_prompt = PromptTemplate(
            """
            Based on the user's query and the relevant keyframes found, generate a comprehensive answer.

            Original Query and questions: {query}

            Relevant Keyframes:
            {keyframes_context}

            Please provide a detailed answer that:
            1. Directly addresses the user's query
            2. References specific information from the keyframes
            3. Synthesizes information across multiple keyframes if relevant
            4. Mentions which videos/keyframes contain the most relevant content

            Keep the answer informative but concise.
            """
        )

    async def generate_answer(
        self,
        original_query: str,
        final_keyframes: List[KeyframeServiceReponse],
        objects_data: Dict[str, List[str]],
        asr_data: Dict[str, dict | str],
    ):
        chat_messages = []
        for kf in final_keyframes:
            keyy = f"{kf.prefix}{kf.group_num:02d}/{kf.prefix}{kf.group_num:02d}_V{kf.video_num:03d}/{kf.keyframe_num:03d}.jpg"
            objects = objects_data.get(keyy, [])

            image_path = os.path.join(
                self.data_folder,
                f"{kf.prefix}{kf.group_num:02d}/{kf.prefix}{kf.group_num:02d}_V{kf.video_num:03d}/{kf.keyframe_num:03d}.jpg",
            )

            # Lấy ASR theo video
            vkey = f"{kf.prefix}{kf.group_num:02d}_V{kf.video_num:03d}.mp4"
            asr_rec = asr_data.get(vkey, {})
            if isinstance(asr_rec, dict):
                asr_text = (
                    asr_rec.get("asr_clean") or asr_rec.get("asr_raw") or ""
                ).strip()
            else:
                asr_text = str(asr_rec).strip() if asr_rec else ""
            asr_snippet = (asr_text[:400] + "…") if len(asr_text) > 400 else asr_text

            context_text = f"""
            Keyframe {kf.key} from Video {kf.video_num} (Confidence: {kf.confidence_score:.3f}):
            - Detected Objects: {', '.join(objects) if objects else 'None detected'}
            - ASR Snippet: {asr_snippet if asr_snippet else "N/A"}
            """

            if os.path.exists(image_path):
                message_content = [
                    ImageBlock(path=Path(image_path)),
                    TextBlock(text=context_text),
                ]
            else:
                message_content = [
                    TextBlock(text=context_text + "\n(Image not available)")
                ]

            user_message = ChatMessage(role=MessageRole.USER, content=message_content)
            chat_messages.append(user_message)

        final_prompt = self.answer_prompt.format(
            query=original_query,
            keyframes_context="See the keyframes and their context above",
        )
        query_message = ChatMessage(
            role=MessageRole.USER, content=[TextBlock(text=final_prompt)]
        )
        chat_messages.append(query_message)

        response = await self.llm.achat(chat_messages)
        return response.message.content
