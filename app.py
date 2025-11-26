"""
FastAPI 백엔드 서버 (리팩토링 후)

라우팅과 요청/응답 처리만 담당합니다.
모든 비즈니스 로직과 DB 접근은 Service 계층에 위임합니다.

사용법:
    python app.py

API 엔드포인트:
    GET  /                              - main.html 제공
    GET  /search                        - search.html 제공
    POST /api/search                    - 검색 쿼리 처리
    GET  /api/agenda/{id}               - 안건 상세 조회
    GET  /api/agenda/{id}/formatted-detail - 포맷된 안건 상세
    GET  /api/top-agendas               - Top 5 안건 조회
    GET  /api/hot-issues                - 핫이슈 top 5 조회
    GET  /api/cost-summary              - API 비용 요약
    GET  /details                       - details.html 제공
    GET  /health                        - 헬스 체크
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from pathlib import Path

# Repository
from repositories.agenda_repository import AgendaRepository
from repositories.chroma_repository import ChromaRepository

# Service
from services.agenda_service import AgendaService
from services.agenda_search_service import AgendaSearchService

# Search Pipeline
from search.query_analyzer import QueryAnalyzer
from search.simple_query_analyzer import SimpleQueryAnalyzer
from search.metadata_validator import MetadataValidator

# Utils
from utils.cost_tracker import CostTracker

# Chatbot
from chatbot.router import router as chatbot_router

# ============================================================
# Pydantic Models
# ============================================================

class SearchRequest(BaseModel):
    """검색 요청 모델"""
    query: str
    n_results: Optional[int] = 5


class SearchResult(BaseModel):
    """검색 결과 모델 (안건 단위)"""
    agenda_id: str
    title: str
    ai_summary: str
    key_issues: Optional[List[str]] = None
    main_speaker: str
    all_speakers: str
    speaker_count: int
    meeting_date: str
    meeting_title: str
    status: str
    similarity: float
    chunk_count: int
    meeting_url: str


class SearchResponse(BaseModel):
    """검색 응답 모델"""
    query: str
    total_results: int
    results: List[SearchResult]


class HotIssue(BaseModel):
    """핫이슈 모델"""
    rank: int
    title: str
    proposer: str
    status: str


class TopAgenda(BaseModel):
    """Top 안건 모델"""
    agenda_id: str
    title: str
    meeting_title: str
    meeting_date: str
    ai_summary: Optional[str] = None
    chunk_count: int
    main_speaker: str
    status: str


# ============================================================
# FastAPI App 초기화
# ============================================================

app = FastAPI(title="SeoulLog API")

# Chatbot 라우터 추가
app.include_router(chatbot_router, prefix="/api", tags=["Chatbot"])

# HTML 파일 경로
HTML_DIR = Path("frontend")

# 비용 추적기 초기화 (전역)
cost_tracker = CostTracker()

# ============================================================
# 의존성 초기화 (Repository → Service)
# ============================================================

print("="*80)
print("SeoulLog 백엔드 서버 초기화")
print("="*80)

# Repository 초기화
print("\n📦 Repository 계층 초기화...")
chroma_repo = ChromaRepository()
agenda_repo = AgendaRepository()
print("✅ ChromaRepository, AgendaRepository 초기화 완료")

# 쿼리 분석기 초기화
print("\n🔍 쿼리 분석기 초기화...")
try:
    analyzer = QueryAnalyzer()
    print("✅ QueryAnalyzer (OpenAI) 초기화 성공")
except Exception as e:
    print(f"⚠️ QueryAnalyzer (OpenAI) 초기화 실패: {e}")
    print("   → SimpleQueryAnalyzer (규칙 기반) 사용")
    analyzer = SimpleQueryAnalyzer()

# 메타데이터 검증기 초기화
print("\n🔎 메타데이터 검증기 초기화...")
try:
    validator = MetadataValidator(
        collection_name="seoul_council_meetings",
        persist_directory="./data/chroma_db"
    )
    print("✅ MetadataValidator 초기화 성공")
except Exception as e:
    print(f"⚠️ MetadataValidator 초기화 실패: {e}")
    validator = None

# Service 초기화 (의존성 주입)
print("\n⚙️ Service 계층 초기화...")
search_service = AgendaSearchService(
    chroma_repo=chroma_repo,
    agenda_repo=agenda_repo,
    analyzer=analyzer,
    validator=validator,
    cost_tracker=cost_tracker
)
agenda_service = AgendaService(agenda_repo=agenda_repo)
print("✅ AgendaSearchService, AgendaService 초기화 완료")

print("\n" + "="*80)
print("✅ 서버 초기화 완료!")
print("="*80 + "\n")


# ============================================================
# 라우트 정의
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """
    메인 페이지 (main.html) 반환
    """
    main_html_path = HTML_DIR / "main.html"

    if not main_html_path.exists():
        raise HTTPException(status_code=404, detail="main.html not found")

    with open(main_html_path, 'r', encoding='utf-8') as f:
        return f.read()


@app.get("/search", response_class=HTMLResponse)
async def get_search_page():
    """
    검색 결과 페이지 (search.html) 반환
    """
    search_html_path = HTML_DIR / "search.html"

    if not search_html_path.exists():
        raise HTTPException(status_code=404, detail="search.html not found")

    with open(search_html_path, 'r', encoding='utf-8') as f:
        return f.read()


@app.get("/chat", response_class=HTMLResponse)
async def get_chat_page():
    """
    챗봇 페이지 (chatbot.html) 반환
    """
    chat_html_path = HTML_DIR / "chatbot.html"

    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail="chatbot.html not found")

    with open(chat_html_path, 'r', encoding='utf-8') as f:
        return f.read()


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    안건 단위 검색

    Service 계층에 완전히 위임합니다.

    Args:
        request: 검색 요청 (query, n_results)

    Returns:
        안건 단위 검색 결과 리스트
    """
    try:
        # Service 호출만
        results = await search_service.search(
            query=request.query,
            n_results=request.n_results or 5
        )

        return SearchResponse(
            query=request.query,
            total_results=len(results),
            results=results
        )

    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hot-issues", response_model=List[HotIssue])
async def get_hot_issues():
    """
    핫이슈 top 5 조회

    현재는 임시 데이터를 반환합니다.
    TODO: 실제로는 ChromaDB에서 인기 안건을 조회해야 함

    Returns:
        핫이슈 리스트
    """
    hot_issues = [
        HotIssue(
            rank=1,
            title="청년안심주택 공급 확대 조례안",
            proposer="김서울 의원",
            status="심사 중"
        ),
        HotIssue(
            rank=2,
            title="역세권 청년주택 관련 개정안",
            proposer="박시민 의원",
            status="통과"
        ),
        HotIssue(
            rank=3,
            title="서울시 청년주거 기본 조례 일부개정조례안",
            proposer="이나라 의원",
            status="계류"
        ),
        HotIssue(
            rank=4,
            title="공공자전거 '따릉이' 운영 효율화 방안",
            proposer="최교통 의원",
            status="심사 중"
        ),
        HotIssue(
            rank=5,
            title="반려동물 친화도시 조성을 위한 조례안",
            proposer="김애견 의원",
            status="통과"
        )
    ]

    return hot_issues


@app.get("/api/top-agendas", response_model=List[TopAgenda])
async def get_top_agendas():
    """
    Top 5 안건 조회 (논의가 활발했던 최신 안건)

    Service 계층에 완전히 위임합니다.

    Returns:
        Top 5 안건 리스트
    """
    try:
        agendas = await agenda_service.get_top_agendas(limit=5)
        return agendas

    except Exception as e:
        print(f"❌ Top 안건 조회 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cost-summary")
async def get_cost_summary():
    """
    누적 API 비용 요약 조회

    Returns:
        비용 요약 딕셔너리
    """
    summary = cost_tracker.get_summary()

    # 상세 정보 추가
    detailed_summary = {
        **summary,
        "session_info": {
            "total_searches": cost_tracker.costs_breakdown.get('embedding', {}).get('calls', 0),
            "total_queries_analyzed": cost_tracker.costs_breakdown.get('chat', {}).get('calls', 0)
        }
    }

    return detailed_summary


@app.get("/details", response_class=HTMLResponse)
async def get_details_page():
    """
    안건 상세 페이지 (details.html) 반환
    """
    details_html_path = HTML_DIR / "details.html"

    if not details_html_path.exists():
        raise HTTPException(status_code=404, detail="details.html not found")

    with open(details_html_path, 'r', encoding='utf-8') as f:
        return f.read()


@app.get("/api/agenda/{agenda_id}")
async def get_agenda_detail(agenda_id: str):
    """
    안건 상세 정보 조회

    Service 계층에 완전히 위임합니다.

    Args:
        agenda_id: 안건 ID (예: meeting_20251117_195534_agenda_001)

    Returns:
        안건 상세 정보 (제목, 발언자, 전체 텍스트 등)
    """
    try:
        detail = await agenda_service.get_agenda_detail(agenda_id)
        return detail

    except ValueError as e:
        # 안건을 찾을 수 없는 경우
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"❌ 안건 상세 조회 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agenda/{agenda_id}/formatted-detail")
async def get_formatted_agenda_detail(agenda_id: str):
    """
    안건 상세 페이지용 포맷된 텍스트 생성

    Service 계층에 완전히 위임합니다.

    Returns:
        {
            "agenda_title": "...",
            "summary": "...",  # 3-6줄 요약
            "attachments": [{"title": "...", "summary": "..."}],
            "combined_text": "..."
        }
    """
    try:
        detail = await agenda_service.get_formatted_detail(agenda_id)
        return detail

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"❌ 포맷된 안건 상세 조회 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    헬스 체크 엔드포인트
    """
    return {"status": "healthy"}


# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":
    import socket

    # 로컬 IP 주소 가져오기
    def get_local_ip():
        try:
            # 외부 연결을 시도해서 로컬 IP 확인
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return "IP를 가져올 수 없음"

    local_ip = get_local_ip()

    print("=" * 80)
    print("SeoulLog 백엔드 서버 시작")
    print("=" * 80)
    print()
    print("🌐 로컬 접속: http://localhost:8000")
    print(f"📱 모바일 접속 (같은 WiFi): http://{local_ip}:8000")
    print()
    print("📄 메인 페이지: /")
    print("🔍 검색 API: /api/search")
    print("🔥 핫이슈 API: /api/hot-issues")
    print("📊 Top 안건 API: /api/top-agendas")
    print("💰 비용 요약 API: /api/cost-summary")
    print()
    print("💡 검색 1회당 비용: 약 0.03~0.05원 (QueryAnalyzer 사용 시)")
    print("   - Embedding: ~0.001원")
    print("   - QueryAnalyzer: ~0.04원")
    print()
    print("서버를 종료하려면 Ctrl+C를 누르세요.")
    print("=" * 80)
    print()

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("🛑 서버 종료 중...")
        print("=" * 80)

        # 전체 세션 비용 출력
        if cost_tracker.total_cost > 0:
            cost_tracker.print_summary()
        else:
            print("\n💰 이번 세션에서는 검색이 없었습니다.")
            print("=" * 80 + "\n")

        print("👋 SeoulLog 서버가 종료되었습니다.\n")
