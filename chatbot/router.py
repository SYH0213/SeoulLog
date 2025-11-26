from fastapi import APIRouter
from pydantic import BaseModel
from chatbot.query_rewriter import rewrite_query
from chatbot.retriever import Retriever  # Retriever 클래스를 직접 임포트
from chatbot.generator import generate_answer

# --- Retriever 인스턴스 전역 생성 ---
# 서버가 시작될 때 한 번만 실행되어 Retriever 객체를 생성하고 재사용합니다.
retriever_instance = Retriever()
# ------------------------------------

# 요청 본문을 위한 Pydantic 모델
class ChatRequest(BaseModel):
    message: str
    session_id: str
    history: list[tuple[str, str]] = []

# 챗봇 API 라우터 생성
router = APIRouter()

@router.post("/chat")
async def handle_chat(request: ChatRequest):
    """
    RAG 챗봇의 전체 파이프라인을 실행하는 엔드포인트입니다.
    (쿼리 재구성 -> 문서 검색 -> 답변 생성)
    """
    # 1. 프론트엔드에서 직접 받은 대화 기록을 사용
    history = request.history
    
    # 2. 쿼리 재구성
    rewritten_question = rewrite_query(request.message, history)

    # 3. 문서 검색 (미리 생성된 retriever_instance 사용)
    retrieved_docs = retriever_instance.retrieve(rewritten_question, n_results=3)

    # 4. 답변 생성
    final_answer = generate_answer(rewritten_question, retrieved_docs)

    # 최종 답변 및 중간 결과 반환 (디버깅용)
    return {
        "response": final_answer,
        "debug_info": {
            "original_question": request.message,
            "rewritten_question": rewritten_question,
            "retrieved_documents": retrieved_docs
        }
    }
