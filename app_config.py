# config.py
import os
from dotenv import load_dotenv  # ✨ 신규: 환경변수 로드용

# --- ✨ 환경 변수 로드 ---
# .env 파일을 찾아서 환경 변수로 설정합니다. (로컬 개발용, Railway에서는 선택사항)
load_dotenv()

# --- 경로 설정 ---
# 1. config.py 파일이 있는 현재 디렉터리 (프로젝트 루트로 간주)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 2. prompts 폴더 경로 조합 (PROJECT_ROOT 기준)
SYSTEM_PROMPT_PATH = os.path.join(PROJECT_ROOT, "prompts/evaluation_prompt.txt")

# 다운로드 경로는 그대로 유지
DOWNLOADS_PATH = os.path.join(os.path.expanduser("~"), "Downloads")


# --- Google AI API 설정 (환경 변수 로드) ---
API_KEY = os.getenv("GOOGLE_API_KEY")  # 환경 변수에서 API 키 로드
API_MODEL = os.getenv(
    "GEMINI_MODEL", "gemini-2.5-flash"
)  # 환경 변수에서 모델 로드 (없으면 기본값 사용)

if not API_KEY:
    # 환경 변수 로드 실패 시 예외 발생
    error_msg = """
    GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.
    
    로컬 환경: .env 파일에 GOOGLE_API_KEY=<your-api-key>를 추가하세요.
    Railway 배포: Variables 탭에서 GOOGLE_API_KEY 환경 변수를 추가하고 재배포하세요.
    
    현재 환경 변수 목록: {env_vars}
    """.format(
        env_vars=", ".join(
            [
                k
                for k in os.environ.keys()
                if "GOOGLE" in k or "GEMINI" in k or "API" in k
            ]
        )
    )
    raise ValueError(error_msg)

# --- 파일 검색 및 기본값 설정 ---
TARGET_FILE_KEYWORDS = ["9월", "스터디", "이용호"]
DEFAULT_CONTENT = "한국에 대해 알려줘"
