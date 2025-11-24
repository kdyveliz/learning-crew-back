import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import List, Optional, Dict, Any

# 로컬 모듈 임포트
import app_config
import file_utils
import db_utils
from gemini_service import GeminiService
from analysis_service import AnalysisService


# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- FastAPI 앱 및 CORS 설정 ---
app = FastAPI()

origins = [
    # Local development
    "http://127.0.0.1",
    "http://127.0.0.1:5500",
    "http://localhost",
    "http://localhost:5500",
    # Vercel frontend deployment
    "https://learning-crew-front-gq5rktzzp-pyhs-projects-a23019b7.vercel.app",
    # Wildcard for any other origins (development)
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 서비스 초기화 ---
gemini_service = GeminiService()
analysis_service = AnalysisService(gemini_service)
SYSTEM_PROMPT = "ERROR: PROMPT NOT LOADED"


# --- FastAPI 이벤트 핸들러 (DB 초기화) ---
@app.on_event("startup")
def startup_event():
    global SYSTEM_PROMPT
    logger.info("서버 시작... 시스템 프롬프트 로드 및 DB 초기화 중...")
    try:
        SYSTEM_PROMPT = file_utils.load_system_prompt(app_config.SYSTEM_PROMPT_PATH)
        logger.info("시스템 프롬프트 로드 완료")
    except Exception as e:
        logger.critical(f"서버 시작 실패: 시스템 프롬프트 로드 중 오류 발생 - {e}")
        SYSTEM_PROMPT = "ERROR: PROMPT NOT LOADED"

    db_utils.init_db()


@app.get("/")
def read_root():
    return {"message": "Gemini 분석 API 서버"}


# --- (수정) 파일 업로드 API ---
@app.post("/upload-and-analyze")
async def upload_and_analyze(
    plan_files: List[UploadFile] = File(...), report_files: List[UploadFile] = File(...)
):
    plans_map, reports_map = {}, {}
    all_keys = set()

    # 매칭 키 추출 로직도 서비스로 위임
    for file in plan_files:
        if key := analysis_service.get_matching_key(file.filename):
            plans_map[key] = file
            all_keys.add(key)

    for file in report_files:
        if key := analysis_service.get_matching_key(file.filename):
            reports_map[key] = file
            all_keys.add(key)

    tasks = []
    for key in all_keys:
        tasks.append(
            analysis_service.process_single_pair(
                key,
                plans_map.get(key),
                reports_map.get(key),
                SYSTEM_PROMPT,
            )
        )

    # 모든 작업 비동기 실행
    processing_results = await asyncio.gather(*tasks)

    summary = {
        "total_plans": len(plan_files),
        "total_reports": len(report_files),
        "processed_count": len(processing_results),
        "unmatchable_plans": [
            f.filename
            for f in plan_files
            if analysis_service.get_matching_key(f.filename) not in all_keys
        ],
        "unmatchable_reports": [
            f.filename
            for f in report_files
            if analysis_service.get_matching_key(f.filename) not in all_keys
        ],
    }
    return {"summary": summary, "results": processing_results}


# --- 게시판 목록 API ---
@app.get("/results")
async def get_all_results(
    campus: Optional[str] = Query(None),
    class_name: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
):
    try:
        results = db_utils.get_all_results(campus, class_name, start_date, end_date, q)
        return results
    except Exception as e:
        logger.error(f"결과 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# --- 참석자 목록 API ---
@app.get("/attendance")
async def get_attendance(
    campus: Optional[str] = Query(None),
    class_name: Optional[str] = Query(None),
):
    try:
        results = db_utils.get_all_attendance(campus, class_name)
        return results
    except Exception as e:
        logger.error(f"참석자 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# --- 필터 옵션 API ---
@app.get("/filter-options")
async def get_filter_options():
    try:
        options = db_utils.get_filter_options()
        return options
    except Exception as e:
        logger.error(f"필터 옵션 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# --- 세부 내용 API ---
@app.get("/results/{result_id}")
async def get_result_detail(result_id: int):
    try:
        return db_utils.get_result_detail(result_id)
    except Exception as e:
        logger.error(f"결과 세부 조회 실패 (ID: {result_id}): {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# --- 결과 수정 API ---
@app.put("/results/{result_id}")
async def update_result(result_id: int, analysis_data: Dict[str, Any]):
    try:
        success = db_utils.update_analysis_result(result_id, analysis_data)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Result ID {result_id} not found or update failed",
            )
        return {"message": "Update successful", "id": result_id}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"결과 수정 실패 (ID: {result_id}): {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# --- DB 전체 내용 확인 API (관리자/디버깅용) ---
@app.get("/db/inspect")
async def inspect_database():
    """
    DB에 저장된 모든 레코드를 전체 컬럼과 함께 반환합니다.
    관리자 또는 디버깅 목적으로 사용합니다.
    """
    try:
        records = db_utils.get_all_db_records()
        return {"total_records": len(records), "records": records}
    except Exception as e:
        logger.error(f"DB 전체 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# --- DB 전체 내용 삭제 API (관리자용, 위험!) ---
@app.delete("/db/clear")
async def clear_database(confirm: bool = Query(False)):
    """
    DB에 저장된 모든 레코드를 삭제합니다.
    안전을 위해 confirm=true 파라미터가 필요합니다.

    사용 예: DELETE /db/clear?confirm=true
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="DB 삭제를 위해서는 confirm=true 파라미터가 필요합니다.",
        )

    try:
        deleted_count = db_utils.delete_all_records()
        return {
            "success": True,
            "message": f"{deleted_count}개의 레코드가 삭제되었습니다.",
            "deleted_count": deleted_count,
        }
    except Exception as e:
        logger.error(f"DB 전체 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True, log_level="info")
