# main.py
import logging
from google import genai
from google.genai import types

# 로컬 모듈 임포트 (이름 변경)
import app_config  # 'config' 대신 'app_config'를 임포트
import file_utils

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- API 호출 함수 ---
def call_gemini_api(system_prompt: str, content: str) -> str:
    """Google AI API를 호출하고 응답 텍스트를 반환합니다."""
    logger.info("Google AI API 호출 중...")
    try:
        # app_config에서 API_KEY와 MODEL을 가져옵니다.
        client = genai.Client(api_key=app_config.API_KEY)
        response = client.models.generate_content(
            model=app_config.API_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
            contents=content,
        )
        logger.info("API 호출 성공")
        return response.text
    except Exception as e:
        logger.error(f"API 호출 실패: {e}")
        raise


# --- 메인 실행 로직 ---
def main():
    """메인 실행 함수"""
    logger.info("프로세스 시작...")

    # 1. 시스템 프롬프트 로드
    try:
        # --- 디버깅 코드는 이제 제거해도 됩니다 ---
        # print("--- config 모듈 속성 검사 ---")
        # print(dir(app_config)) # 필요하다면 app_config로 테스트
        # ------------------------------------

        # app_config에서 SYSTEM_PROMPT_PATH를 가져옵니다.
        system_prompt = file_utils.load_system_prompt(app_config.SYSTEM_PROMPT_PATH)
        logger.info("시스템 프롬프트 로드 완료")
    except FileNotFoundError as e:
        logger.error(f"시스템 프롬프트 파일을 찾을 수 없습니다: {e}")
        print(
            f"오류: 시스템 프롬프트 파일을 찾을 수 없습니다. ({app_config.SYSTEM_PROMPT_PATH})"
        )
        return

    # 2. 보고서 파일 검색
    # app_config에서 경로와 키워드를 가져옵니다.
    report_file_path = file_utils.find_file_by_keywords(
        app_config.DOWNLOADS_PATH, app_config.TARGET_FILE_KEYWORDS
    )

    # 3. 보고서 내용 읽기
    report_content = app_config.DEFAULT_CONTENT  # 기본값으로 초기화

    if report_file_path:
        try:
            report_content = file_utils.get_file_content(report_file_path)
            print(f"파일을 성공적으로 읽었습니다: {report_file_path}")
            logger.info(f"파일 읽기 완료: {report_file_path}")
        except Exception as e:
            logger.error(f"파일 읽기 중 오류 발생: {e}. 기본값을 사용합니다.")
            print(f"파일을 읽는 중 오류 발생({e}). 기본값을 사용합니다.")
    else:
        logger.warning(
            f"파일을 찾을 수 없어 기본값을 사용합니다. (경로: {app_config.DOWNLOADS_PATH})"
        )
        print(f"파일을 찾을 수 없습니다. 기본값을 사용합니다.")

    # 4. API 호출 및 결과 출력
    try:
        api_response = call_gemini_api(system_prompt, report_content)

        print("\n[Gemini API 응답]============================\n")
        print(api_response)
        print("\n============================================\n")

    except Exception as e:
        print(f"API 호출 중 심각한 오류가 발생했습니다: {e}")

    logger.info("프로세스 완료")


if __name__ == "__main__":
    main()
