# Gemini 작업 기록

### 2025년 11월 24일

1.  **백엔드 서버 오류 디버깅**
    *   `backend_server.py` 실행 시 발생한 `chromadb.errors.NotFoundError` 해결
        *   원인: `seoul_council_meetings` 컬렉션이 존재하지 않음.
        *   해결: `database/insert_to_chromadb.py` 스크립트를 실행하여 컬렉션 생성 및 데이터 삽입.
        *   추가 조치: 스크립트 실행 중 `UnicodeEncodeError`가 발생하여, 문자 인코딩 문제(이모지)를 수정함.
    *   `/api/search` API의 `500 Internal Server Error` 해결
        *   원인: OpenAI API 키의 사용량(quota) 초과.
        *   해결: 사용자에게 원인을 안내하고, 사용자가 직접 OpenAI 계정 문제를 해결함.

2.  **챗봇 기능 통합**
    *   `backend_server.py`에 챗봇 API 라우터(`chatbot/router.py`)를 연동하여 `/api/chat` 엔드포인트를 활성화함.
    *   `/chat` 경로로 접속 시 챗봇 UI(`chatbot.html`)가 표시되도록 `backend_server.py`에 엔드포인트를 추가함.

3.  **프론트엔드 UI 개선**
    *   **플로팅 버튼 추가:** 모든 페이지(`main.html`, `search.html`, `details.html`)에 챗봇 페이지로 이동하는 플로팅 버튼을 추가함.
    *   **정렬 드롭다운 구현:** `search.html`의 '관련도순' 버튼을 실제 드롭다운 메뉴로 변경하고, '최신순', '진행도순' 옵션을 추가함.
    *   **챗봇 UI 통일성 확보:**
        *   `chatbot.html` 파일을 `frontend` 폴더로 이동하고 서버 코드의 경로를 수정함.
        *   `chatbot.html`의 색상 팔레트, 헤더 디자인 등을 다른 페이지와 일치하도록 수정하여 전체적인 UI 통일성을 높임.

### 향후 계획: LangGraph 기반 챗봇 에이전트 구현

*   **목표:** 사용자의 입력 유형(단순 키워드 vs. 질문)을 구분하여, 각기 다른 로직을 실행하는 지능형 챗봇 에이전트 개발.
*   **핵심 기술:** LangGraph
*   **구현 방안:**
    1.  **입력 분류 (Intent Classification):** 사용자의 입력을 받아 '키워드' 또는 '질문' 유형으로 분류하는 노드(Node)를 생성.
    2.  **조건부 라우팅 (Conditional Routing):**
        *   '키워드'로 분류될 경우: 키워드에 최적화된 필터링 검색을 수행하는 경로로 분기.
        *   '질문'으로 분류될 경우: 기존의 RAG 파이프라인(쿼리 재구성 -> 검색 -> 답변 생성)을 수행하는 경로로 분기.
    3.  **그래프 구성:** 위의 분류 및 라우팅 로직을 LangGraph의 상태(State), 노드(Node), 엣지(Edge)를 사용하여 하나의 워크플로우로 통합.