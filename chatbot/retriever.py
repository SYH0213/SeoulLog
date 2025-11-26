from utils.search_chromadb import MeetingSearcher
from typing import List, Dict

class Retriever:
    """
    ChromaDB에서 관련 문서를 검색하는 클래스
    """
    def __init__(self, collection_name: str = "seoul_council_meetings", persist_directory: str = "./data/chroma_db"):
        """
        Retriever 초기화
        """
        try:
            self.searcher = MeetingSearcher(
                collection_name=collection_name,
                persist_directory=persist_directory
            )
            print("✅ Retriever 초기화 완료")
        except Exception as e:
            print(f"❌ Retriever 초기화 실패: {e}")
            self.searcher = None

    def retrieve_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        주어진 쿼리로 문서를 검색하고, 텍스트 내용을 반환합니다.

        Args:
            query: 검색할 쿼리 (재작성된 질문)
            n_results: 반환할 결과 수

        Returns:
            검색된 문서의 내용(text)과 메타데이터를 담은 딕셔너리 리스트
        """
        if not self.searcher:
            return []

        print(f"🔍 문서 검색 중... (query: {query})")
        
        # MeetingSearcher를 사용하여 간단한 텍스트 기반 검색 수행
        search_results = self.searcher.search(query=query, n_results=n_results)

        if not search_results or 'results' not in search_results:
            print("   -> 검색 결과 없음")
            return []

        documents = []
        for result in search_results['results']:
            documents.append({
                "text": result.get("text", ""),
                "similarity": result.get("similarity", 0.0),
                "source": result.get("agenda", "N/A")  # BUGFIX: agenda_id -> agenda
            })
        
        print(f"   -> {len(documents)}개 문서 검색 완료")
        return documents

def retrieve_documents(query: str, n_results: int = 5) -> List[Dict]:
    """
    Retriever 인스턴스를 생성하고 문서를 검색하는 헬퍼 함수
    """
    retriever = Retriever()
    return retriever.retrieve_documents(query, n_results)

if __name__ == '__main__':
    # 테스트용 코드
    test_query = "서울시 AI 정책"
    print(f"\n--- Retriever 테스트 (query: '{test_query}') ---")
    retrieved_docs = retrieve_documents(test_query, n_results=3)

    if retrieved_docs:
        for i, doc in enumerate(retrieved_docs):
            print(f"\n[문서 {i+1}] 유사도: {doc['similarity']:.4f}, 출처: {doc['source']}")
            print(f"내용: {doc['text'][:150]}...")
    else:
        print("검색된 문서가 없습니다.")
