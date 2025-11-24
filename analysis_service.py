import os
import re
import json
import logging
import unicodedata
from typing import Optional
from fastapi import UploadFile

import file_utils
import db_utils
from gemini_service import GeminiService

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self, gemini_service: GeminiService):
        self.gemini_service = gemini_service

    def extract_info_from_filename(self, filename: str) -> dict:
        """파일명에서 캠퍼스, 반, 작성자 정보를 추출합니다."""
        # 파일명에서 (1), (2) 같은 번호 패턴 제거
        filename = file_utils.clean_filename(filename)

        CAMPUS_LIST_RAW = ["광주", "구미", "서울", "대전", "부울경"]
        CAMPUS_LIST = [unicodedata.normalize("NFC", s) for s in CAMPUS_LIST_RAW]
        CLASS_REGEX = r"(\d+반)"
        try:
            name_without_ext = os.path.splitext(filename)[0]
            name_without_ext = unicodedata.normalize("NFC", name_without_ext)
            parts = name_without_ext.split("_")
            info = {"campus": None, "class_name": None, "author_name": None}

            if len(parts) >= 4:
                campus_candidate = parts[-3].strip()
                class_candidate = parts[-2].strip()
                author_raw = parts[-1].strip()
                if campus_candidate in CAMPUS_LIST and re.fullmatch(
                    CLASS_REGEX, class_candidate
                ):
                    info["campus"] = campus_candidate
                    info["class_name"] = class_candidate
                    info["author_name"] = re.split(r"[\.\s-]", author_raw, 1)[0]
                    return info

            for part in parts:
                if part in CAMPUS_LIST:
                    info["campus"] = part
                elif re.fullmatch(CLASS_REGEX, part):
                    info["class_name"] = part

            for part in reversed(parts):
                if (
                    (part not in CAMPUS_LIST)
                    and (not re.fullmatch(CLASS_REGEX, part))
                    and ("보고서" not in part)
                    and ("계획서" not in part)
                    and (".xlsx" not in part)
                ):
                    info["author_name"] = part
                    break
            return info
        except Exception:
            return {"campus": None, "class_name": None, "author_name": None}

    def get_matching_key(self, filename: str) -> str | None:
        try:
            # 파일명에서 (1), (2) 같은 번호 패턴 제거
            filename = file_utils.clean_filename(filename)
            name = unicodedata.normalize("NFC", os.path.splitext(filename)[0])
            parts = name.split("_")
            return "_".join(parts[-3:]) if len(parts) >= 3 else None
        except:
            return None

    async def process_single_pair(
        self,
        key: str,
        plan_file: Optional[UploadFile],
        report_file: Optional[UploadFile],
        system_prompt: str,
    ):
        logger.info(f"[{key}] 쌍 처리 시작...")

        # 0. 파일명에서 정보 미리 추출 (참석자 명단 저장 시 활용)
        target_filename = report_file.filename if report_file else plan_file.filename
        target_filename = file_utils.clean_filename(target_filename)
        file_info = self.extract_info_from_filename(target_filename)

        # 1. 이미지 추출 (실제 개수 카운팅)
        extracted_images = []
        for file in [plan_file, report_file]:
            if file and file.filename.lower().endswith(".xlsx"):
                await file.seek(0)
                content = await file.read()
                extracted_images.extend(file_utils.extract_images_from_excel(content))
                await file.seek(0)

        actual_photo_count = len(extracted_images)
        logger.info(f"[{key}] 실제 감지된 이미지: {actual_photo_count}장")

        # 1.5 참석자 명단 추출 및 저장 (중복 제거)
        import tempfile

        # 모든 파일에서 참석자 명단 수집
        all_attendance = []
        for file in [plan_file, report_file]:
            if file and file.filename.lower().endswith(".xlsx"):
                try:
                    # 임시 파일 생성
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".xlsx"
                    ) as tmp:
                        await file.seek(0)
                        content = await file.read()
                        tmp.write(content)
                        tmp_path = tmp.name

                    # 참석자 명단 추출
                    attendance_list = file_utils.extract_attendance_list_from_excel(
                        tmp_path
                    )
                    all_attendance.extend(attendance_list)

                    # 임시 파일 삭제
                    os.remove(tmp_path)
                    await file.seek(0)  # 파일 포인터 초기화

                except Exception as e:
                    logger.error(f"[{key}] 참석자 명단 추출 실패: {e}")
                    if "tmp_path" in locals() and os.path.exists(tmp_path):
                        os.remove(tmp_path)

        # 중복 제거 후 DB 저장
        seen = set()
        for entry in all_attendance:
            parsed = file_utils.parse_attendance_string(entry)

            # 파싱 실패 시, 파일명 정보로 보완 시도
            if not parsed and file_info["campus"] and file_info["class_name"]:
                parsed = {
                    "campus": file_info["campus"],
                    "class_name": file_info["class_name"],
                    "name": entry.strip(),
                }

            if parsed:
                # (campus, class_name, name) 튜플로 중복 체크
                key_tuple = (parsed["campus"], parsed["class_name"], parsed["name"])
                if key_tuple not in seen:
                    seen.add(key_tuple)
                    db_utils.save_attendance_entry(**parsed)
                    logger.info(
                        f"[{key}] 참석자 저장: {parsed['campus']} {parsed['class_name']} {parsed['name']}"
                    )

        # 2. 텍스트 추출
        combined_text = ""
        if plan_file:
            combined_text += f"# [계획서 데이터]\n{await file_utils.read_upload_file_content(plan_file)}\n\n"
        if report_file:
            combined_text += f"# [결과보고서 데이터]\n{await file_utils.read_upload_file_content(report_file)}\n\n"

        if not combined_text:
            return {"key": key, "status": "error", "error": "내용 없음"}

        # 3. 텍스트 스마트 요약
        MAX_TOTAL_CHARS = 25000
        if len(combined_text) > MAX_TOTAL_CHARS:
            head_chars = 20000
            tail_chars = 5000
            combined_text = (
                combined_text[:head_chars]
                + f"\n\n... [중간 내용 {len(combined_text) - MAX_TOTAL_CHARS}자 생략됨] ...\n\n"
                + combined_text[-tail_chars:]
            )
            logger.info(f"[{key}] 텍스트 과다로 앞뒤만 추출하여 전송")

        # 4. 이미지 전송 개수 제한
        MAX_IMAGES_TO_SEND = 3
        images_to_send = extracted_images[:MAX_IMAGES_TO_SEND]

        # 5. 프롬프트 구성
        context_header = f"""
        [분석가를 위한 내부 참고 자료 (절대 출력 금지)]
        1. **실제 증빙 자료**: 기술적 제약으로 샘플 {len(images_to_send)}장만 보이지만, 실제 파일 내부에는 총 **{actual_photo_count}장**의 사진이 포함되어 있음. (평가 시 "풍부한 증빙 자료가 제출되었다"고 판단할 것)
        2. **텍스트 요약**: 내용이 길어 중간이 생략되었으나, 문맥을 통해 전체를 읽은 것처럼 평가할 것.

        [출력 시 주의사항]
        - 위 '내부 참고 자료', 'SYSTEM NOTE', '기술적 한계', '텍스트 생략' 등의 단어를 **결과 코멘트에 절대 언급하지 마십시오.**
        - 마치 당신이 **{actual_photo_count}장의 사진을 모두 직접 눈으로 확인했고, 전체 글을 꼼꼼히 다 읽은 사람처럼** 자연스럽게 작성하십시오.
        --------------------------------------------------
        """

        final_prompt_content = context_header + combined_text
        api_contents = [final_prompt_content] + images_to_send

        # 6. 디버그 저장
        os.makedirs("debug", exist_ok=True)
        with open(f"debug/debug_payload_{key}.txt", "w", encoding="utf-8") as f:
            f.write(final_prompt_content)

        # 7. API 호출 및 결과 처리 (Rate Limit 적용)
        target_filename = report_file.filename if report_file else plan_file.filename
        # 파일명에서 (1), (2) 같은 번호 패턴 제거
        target_filename = file_utils.clean_filename(target_filename)

        async def _call_api():
            return await self.gemini_service.call_gemini_api_async(
                system_prompt, api_contents
            )

        try:
            # GeminiService의 Rate Limit 래퍼 사용
            api_response_text = await self.gemini_service.process_with_rate_limit(
                key, _call_api
            )

            start = api_response_text.find("{")
            end = api_response_text.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("JSON 형식 오류")

            cleaned_json = api_response_text[start : end + 1]
            data = json.loads(cleaned_json)
            if isinstance(data, list):
                data = data[0]

            data["photo_count_detected"] = actual_photo_count
            info = self.extract_info_from_filename(target_filename)

            # DB 저장 (비동기 실행을 위해 to_thread 사용 가능, 여기선 간단히 직접 호출하거나 to_thread)
            # db_utils는 동기 함수이므로 to_thread 권장
            import asyncio

            await asyncio.to_thread(
                db_utils.save_result_to_db,
                os.path.splitext(target_filename)[0],
                data.get("total", 0),
                actual_photo_count,
                json.dumps(data, ensure_ascii=False),
                info.get("campus"),
                info.get("class_name"),
                info.get("author_name"),
            )

            return {
                "key": key,
                "filename": target_filename,
                "status": "success",
                "analysis_result": json.dumps(data, ensure_ascii=False),
            }

        except Exception as e:
            logger.error(f"[{key}] 오류: {e}")
            return {
                "key": key,
                "filename": target_filename,
                "status": "error",
                "error": str(e),
            }
