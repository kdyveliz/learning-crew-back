import logging
import asyncio
from typing import List, Union
from PIL import Image
from google import genai
from google.genai import types
import app_config

logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self):
        self.api_semaphore = asyncio.Semaphore(1)
        self.sleep_time = 6

    def call_gemini_api(
        self, system_prompt: str, contents: List[Union[str, Image.Image]]
    ) -> str:
        """Google GenAI API를 호출합니다."""
        if system_prompt == "ERROR: PROMPT NOT LOADED":
            raise ValueError("시스템 프롬프트가 올바르게 로드되지 않았습니다.")

        img_count = sum(1 for i in contents if isinstance(i, Image.Image))
        logger.info(f"Google AI API 호출 중... (텍스트 + 이미지 {img_count}장)")

        try:
            client = genai.Client(api_key=app_config.API_KEY)
            response = client.models.generate_content(
                model=app_config.API_MODEL,
                config=types.GenerateContentConfig(system_instruction=system_prompt),
                contents=contents,
            )
            return response.text
        except Exception as e:
            logger.error(f"API 호출 실패: {e}")
            raise

    async def call_gemini_api_async(
        self, system_prompt: str, contents: List[Union[str, Image.Image]]
    ) -> str:
        """비동기 래퍼"""
        return await asyncio.to_thread(self.call_gemini_api, system_prompt, contents)

    async def process_with_rate_limit(self, key: str, func, *args, **kwargs):
        """
        세마포어를 사용하여 API 호출 빈도를 제어하는 래퍼 메서드입니다.
        작업 완료 후 대기하여 분당 요청 횟수(RPM) 제한을 준수합니다.
        """
        async with self.api_semaphore:
            try:
                logger.info(f"[{key}] 속도 제한 래퍼 진입. 처리 시작...")
                result = await func(*args, **kwargs)

                # API 호출 후 강제 대기 (Gemini Free Tier: 약 10 RPM)
                logger.info(
                    f"[{key}] API 호출 완료. Rate Limit 준수를 위해 {self.sleep_time}초 대기..."
                )
                await asyncio.sleep(self.sleep_time)

                return result
            except Exception as e:
                logger.error(f"[{key}] 처리 중 예외 발생: {e}")
                raise e
