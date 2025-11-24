# db_utils.py
import sqlite3
import logging
import json
from typing import Optional

# 로컬 모듈 임포트
# app_config에서 DB 경로를 관리하는 경우 여기에 포함시키거나, server.py에서와 같이 직접 정의합니다.
# 여기서는 server.py에서 정의한 DATABASE_URL을 재정의합니다.
DATABASE_URL = "analysis_results.db"

logger = logging.getLogger(__name__)


# --- DB 설정 및 초기화 (server.py에서 이동) ---
def init_db():
    """DB 테이블을 확인하고, 새 컬럼(campus, class_name, author_name)을 추가합니다."""
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        total_score INTEGER,
        photo_count INTEGER,
        analysis_json TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    )

    cursor.execute("PRAGMA table_info(analysis_results)")
    columns = [row[1] for row in cursor.fetchall()]

    if "campus" not in columns:
        cursor.execute("ALTER TABLE analysis_results ADD COLUMN campus TEXT")
        logger.info("DB 스키마 변경: 'campus' 컬럼 추가")
    if "class_name" not in columns:
        cursor.execute("ALTER TABLE analysis_results ADD COLUMN class_name TEXT")
        logger.info("DB 스키마 변경: 'class_name' 컬럼 추가")
    if "author_name" not in columns:
        cursor.execute("ALTER TABLE analysis_results ADD COLUMN author_name TEXT")
        logger.info("DB 스키마 변경: 'author_name' 컬럼 추가")

    conn.commit()
    conn.close()
    logger.info("데이터베이스 테이블 확인/업데이트 완료.")

    # 참석자 테이블 초기화
    init_attendance_db()


# --- DB 저장 함수 (server.py에서 이동) ---
def save_result_to_db(
    filename: str,
    total_score: int,
    photo_count: int,
    analysis_json: str,
    campus: Optional[str],
    class_name: Optional[str],
    author_name: Optional[str],
):
    """분석 결과를 DB에 저장 (신규 컬럼 포함)"""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO analysis_results 
            (filename, total_score, photo_count, analysis_json, campus, class_name, author_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                total_score,
                photo_count,
                analysis_json,
                campus,
                class_name,
                author_name,
            ),
        )
        conn.commit()
        conn.close()
        logger.info(
            f"[{filename}] 결과를 DB에 저장했습니다. (정보: {campus}, {class_name}, {author_name})"
        )
    except Exception as e:
        logger.error(f"[{filename}] DB 저장 실패: {e}")


# --- 결과 목록 조회 함수 (server.py에서 이동) ---
def get_all_results(
    campus: Optional[str],
    class_name: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    q: Optional[str],
) -> list[dict]:
    """DB에 저장된 분석 결과 목록을 (필터링하여) 반환합니다."""
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT id, filename, total_score, created_at, campus, class_name, author_name FROM analysis_results"
    conditions = []
    params = []

    if campus:
        conditions.append("campus = ?")
        params.append(campus)
    if class_name:
        conditions.append("class_name = ?")
        params.append(class_name)
    if start_date:
        conditions.append("DATE(created_at) >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("DATE(created_at) <= ?")
        params.append(end_date)
    if q:
        conditions.append("(author_name LIKE ? OR filename LIKE ?)")
        params.append(f"%{q}%")
        params.append(f"%{q}%")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY created_at DESC"

    cursor.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


# --- 필터 옵션 조회 함수 (server.py에서 이동) ---
def get_filter_options() -> dict:
    """필터링 드롭다운에 사용할 캠퍼스 및 반 목록을 반환합니다."""
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DISTINCT campus FROM analysis_results WHERE campus IS NOT NULL ORDER BY campus"
    )
    campuses = [row["campus"] for row in cursor.fetchall()]

    cursor.execute(
        "SELECT DISTINCT class_name FROM analysis_results WHERE class_name IS NOT NULL ORDER BY class_name"
    )
    class_names = [row["class_name"] for row in cursor.fetchall()]

    conn.close()
    return {"campuses": campuses, "class_names": class_names}


def get_result_detail(result_id: int) -> dict:
    """특정 분석 결과의 상세 내용(JSON 데이터)을 반환합니다."""
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """SELECT 
            filename, 
            analysis_json, 
            campus, 
            class_name, 
            author_name 
        FROM analysis_results 
        WHERE id = ?""",
        (result_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        try:
            analysis_data = json.loads(row["analysis_json"])
        except json.JSONDecodeError:
            analysis_data = {"error": "저장된 JSON 데이터 파싱 실패"}
            logger.error(f"DB 저장된 JSON 파싱 실패 (ID: {result_id})")

        return {
            "filename": row["filename"],
            "campus": row["campus"],
            "class_name": row["class_name"],
            "author_name": row["author_name"],
            "analysis_data": analysis_data,
        }
    else:
        # 이 함수를 server.py에서 호출할 때 HTTPException으로 래핑됩니다.
        raise FileNotFoundError(f"결과 ID {result_id}를 찾을 수 없습니다.")


def update_analysis_result(result_id: int, analysis_data: dict) -> bool:
    """분석 결과를 업데이트합니다."""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()

        # 중첩된 analysis_data가 있다면 평탄화
        if "analysis_data" in analysis_data:
            analysis_data = analysis_data["analysis_data"]

        # 필요한 정보 추출
        total_score = analysis_data.get("total", 0)
        photo_count = analysis_data.get("photo_count_detected", 0)
        analysis_json = json.dumps(analysis_data, ensure_ascii=False)

        cursor.execute(
            """
            UPDATE analysis_results
            SET total_score = ?, photo_count = ?, analysis_json = ?
            WHERE id = ?
            """,
            (total_score, photo_count, analysis_json, result_id),
        )

        if cursor.rowcount == 0:
            logger.warning(f"업데이트 실패: ID {result_id}를 찾을 수 없습니다.")
            conn.close()
            return False

        conn.commit()
        conn.close()
        logger.info(f"ID {result_id} 분석 결과 업데이트 완료")
        return True
    except Exception as e:
        logger.error(f"ID {result_id} 업데이트 중 오류 발생: {e}")
        return False


# --- DB 전체 내용 조회 함수 (관리자/디버깅용) ---
def get_all_db_records() -> list[dict]:
    """DB에 저장된 모든 레코드를 전체 컬럼과 함께 반환합니다. (관리자/디버깅용)"""
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT 
            id, 
            filename, 
            total_score, 
            photo_count, 
            analysis_json, 
            created_at, 
            campus, 
            class_name, 
            author_name 
        FROM analysis_results 
        ORDER BY created_at DESC
    """
    )

    results = []
    for row in cursor.fetchall():
        record = dict(row)
        # JSON 파싱 시도
        try:
            record["analysis_data"] = json.loads(record["analysis_json"])
        except (json.JSONDecodeError, TypeError):
            record["analysis_data"] = None
        # 원본 JSON 문자열은 제거 (데이터 중복 방지)
        del record["analysis_json"]
        results.append(record)

    conn.close()
    return results


# --- DB 전체 내용 삭제 함수 (관리자용) ---
def delete_all_records() -> int:
    """DB에 저장된 모든 레코드를 삭제합니다. (관리자용, 주의!)"""
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    # 삭제 전 레코드 수 확인
    cursor.execute("SELECT COUNT(*) FROM analysis_results")
    count_before = cursor.fetchone()[0]

    # 모든 레코드 삭제
    cursor.execute("DELETE FROM analysis_results")
    conn.commit()

    # 삭제 후 레코드 수 확인
    cursor.execute("SELECT COUNT(*) FROM analysis_results")
    count_after = cursor.fetchone()[0]

    conn.close()

    deleted_count = count_before - count_after
    logger.warning(
        f"⚠️ DB 전체 삭제 실행: {deleted_count}개 레코드 삭제됨 (남은 레코드: {count_after}개)"
    )

    return deleted_count


# --- 참석자 명단 관리 함수 ---


def init_attendance_db():
    """참석자 명단 테이블을 생성합니다."""
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        campus TEXT,
        class_name TEXT,
        name TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    )
    conn.commit()
    conn.close()
    logger.info("참석자 명단 테이블(attendance) 확인/생성 완료.")


def save_attendance_entry(campus: str, class_name: str, name: str):
    """참석자 정보를 DB에 저장합니다."""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO attendance (campus, class_name, name)
            VALUES (?, ?, ?)
            """,
            (campus, class_name, name),
        )
        conn.commit()
        conn.close()
        logger.info(f"참석자 저장 완료: {campus} {class_name} {name}")
    except Exception as e:
        logger.error(f"참석자 저장 실패: {e}")


def get_all_attendance(
    campus: Optional[str] = None, class_name: Optional[str] = None
) -> list[dict]:
    """모든 참석자 명단을 조회합니다 (필터링 지원)."""
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM attendance"
    conditions = []
    params = []

    if campus:
        conditions.append("campus = ?")
        params.append(campus)
    if class_name:
        conditions.append("class_name = ?")
        params.append(class_name)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY created_at DESC"

    cursor.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results
