"""
JSON 메타데이터를 ChromaDB에 비동기 병렬 삽입하는 스크립트

사용법:
    python insert_to_chromadb_async.py

특징:
    - 10개 파일 동시 처리 (비동기 병렬)
    - OpenAI Embedding 비동기 호출
    - 5-8배 빠른 처리 속도
"""

import json
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict
import os
import sys
import asyncio
from openai import AsyncOpenAI

# 프로젝트 루트를 Python path에 추가
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from utils.cost_tracker import CostTracker

load_dotenv()


class AsyncOpenAIEmbeddingFunction:
    """
    비동기 OpenAI Embedding Function (비용 추적 포함)
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "text-embedding-3-small",
        cost_tracker: CostTracker = None
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.cost_tracker = cost_tracker

    async def __call__(self, input: List[str]) -> List[List[float]]:
        """
        텍스트 리스트를 embedding으로 변환 (비동기)

        Args:
            input: 텍스트 리스트

        Returns:
            Embeddings (리스트의 리스트)
        """
        response = await self.client.embeddings.create(
            input=input,
            model=self.model_name
        )

        # 비용 추적
        if self.cost_tracker and hasattr(response, 'usage'):
            tokens = response.usage.total_tokens
            # Embedding 전용 메서드 사용
            self.cost_tracker.add_embedding_cost_tokens(
                tokens=tokens,
                model=self.model_name
            )

        embeddings = [item.embedding for item in response.data]
        return embeddings


def load_json_metadata(json_path: str) -> Dict:
    """JSON 파일에서 메타데이터 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


async def insert_to_chromadb_async(
    json_path: str,
    collection_name: str = "seoul_council_meetings",
    persist_directory: str = "./data/chroma_db",
    semaphore: asyncio.Semaphore = None,
    cost_tracker: CostTracker = None
) -> Dict:
    """
    JSON 메타데이터를 ChromaDB에 비동기 삽입

    Args:
        json_path: JSON 파일 경로
        collection_name: ChromaDB 컬렉션 이름
        persist_directory: ChromaDB 저장 경로
        semaphore: 동시 실행 제한
        cost_tracker: 비용 추적 객체

    Returns:
        삽입 결과 딕셔너리
    """
    if semaphore:
        async with semaphore:
            return await _insert_single_json(json_path, collection_name, persist_directory, cost_tracker)
    else:
        return await _insert_single_json(json_path, collection_name, persist_directory, cost_tracker)


async def _insert_single_json(
    json_path: str,
    collection_name: str,
    persist_directory: str,
    cost_tracker: CostTracker = None
) -> Dict:
    """단일 JSON 파일 삽입 (내부 함수)"""

    try:
        # 1. JSON 로드
        data = load_json_metadata(json_path)
        meeting_info = data['meeting_info']
        chunks = data['chunks']
        meeting_id = Path(json_path).stem

        print(f"📂 처리 중: {meeting_info['title'][:50]}... ({len(chunks)}개 청크)")

        # 2. ChromaDB 클라이언트 생성
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # 2-1. 비동기 Embedding 함수 생성 (비용 추적 포함)
        async_openai_ef = AsyncOpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
            cost_tracker=cost_tracker
        )

        # 3. 컬렉션 가져오기 또는 생성
        try:
            collection = client.get_collection(
                name=collection_name,
                embedding_function=None  # 비동기 함수는 나중에 사용
            )
        except:
            # 컬렉션이 없으면 생성
            collection = client.create_collection(
                name=collection_name,
                metadata={
                    "description": "서울시의회 회의록 청크",
                    "hnsw:space": "cosine"
                }
            )

        # 4. 안건별 인덱싱
        agenda_to_id = {}
        agenda_counter = 0

        for chunk in chunks:
            agenda = chunk['agenda'] if chunk['agenda'] else "기타발언"
            if agenda not in agenda_to_id:
                agenda_to_id[agenda] = f"{meeting_id}_agenda_{agenda_counter:03d}"
                agenda_counter += 1

        # 5. 배치 처리 준비
        documents = []
        metadatas = []
        ids = []

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{meeting_id}_chunk_{idx:04d}"
            document = chunk['text']
            agenda = chunk['agenda'] if chunk['agenda'] else "기타발언"
            agenda_id = agenda_to_id[agenda]

            metadata = {
                "meeting_title": meeting_info['title'],
                "meeting_date": meeting_info['date'],
                "meeting_url": meeting_info.get('url', ''),
                "speaker": chunk['speaker'],
                "agenda": chunk['agenda'] if chunk['agenda'] else "기타발언",
                "agenda_id": agenda_id,
                "chunk_index": idx
            }

            documents.append(document)
            metadatas.append(metadata)
            ids.append(chunk_id)

        # 6. 비동기 Embedding 생성 (배치)
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]

            # 비동기 Embedding 생성
            embeddings = await async_openai_ef(batch_docs)
            all_embeddings.extend(embeddings)

            # 진행 상황 출력 (간결하게)
            if i + batch_size < len(documents):
                print(f"   ⚡ {i+batch_size}/{len(documents)} 임베딩 완료")

        # 7. ChromaDB에 삽입 (Embedding 포함)
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_embeddings = all_embeddings[i:i+batch_size]

            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids,
                embeddings=batch_embeddings
            )

        print(f"   ✅ 완료: {meeting_info['title'][:50]}... ({len(chunks)}개 청크)")

        return {
            "status": "success",
            "file": json_path,
            "chunks": len(chunks),
            "agendas": len(agenda_to_id)
        }

    except Exception as e:
        print(f"   ❌ 실패: {Path(json_path).name} - {str(e)[:100]}")
        return {
            "status": "failed",
            "file": json_path,
            "error": str(e)
        }


async def insert_all_jsons_async(
    json_dir: str = "data/result_txt",
    collection_name: str = "seoul_council_meetings",
    persist_directory: str = "./data/chroma_db",
    max_concurrent: int = 10
):
    """
    result_txt 폴더의 모든 JSON 파일을 ChromaDB에 비동기 병렬 삽입

    Args:
        json_dir: JSON 파일들이 있는 디렉토리
        collection_name: ChromaDB 컬렉션 이름
        persist_directory: ChromaDB 저장 경로
        max_concurrent: 최대 동시 처리 개수 (기본 10개)

    Returns:
        CostTracker: 비용 추적 객체
    """
    json_path = Path(json_dir)
    json_files = list(json_path.glob("*.json"))

    print(f"📁 {len(json_files)}개 JSON 파일 발견")
    print(f"⚡ {max_concurrent}개 파일씩 동시 처리")
    print("="*80)

    # 비용 추적기 생성
    cost_tracker = CostTracker()

    # Semaphore로 동시 처리 개수 제한
    semaphore = asyncio.Semaphore(max_concurrent)

    # 모든 파일 병렬 처리
    tasks = [
        insert_to_chromadb_async(
            json_path=str(json_file),
            collection_name=collection_name,
            persist_directory=persist_directory,
            semaphore=semaphore,
            cost_tracker=cost_tracker
        )
        for json_file in json_files
    ]

    # 완료된 순서대로 결과 수집
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 결과 집계
    success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    failed_count = len(results) - success_count

    print("="*80)
    print(f"\n📊 최종 결과:")
    print(f"   ✅ 성공: {success_count}개")
    print(f"   ❌ 실패: {failed_count}개")

    if failed_count > 0:
        print(f"\n실패한 파일:")
        for r in results:
            if isinstance(r, dict) and r.get("status") == "failed":
                print(f"   - {Path(r['file']).name}: {r['error'][:50]}")

    print(f"\n🎉 ChromaDB 삽입 완료!")

    # 비용 요약 출력
    print("="*80)
    print("💰 Step 2 비용 요약 (OpenAI Embedding)")
    print("="*80)
    cost_tracker.print_summary()
    print()

    return cost_tracker


def insert_all_jsons_sync():
    """동기 래퍼 함수 (기존 코드와 호환성 유지)

    Returns:
        CostTracker: 비용 추적 객체
    """
    return asyncio.run(insert_all_jsons_async())


if __name__ == "__main__":
    print("=" * 80)
    print("🚀 비동기 병렬 ChromaDB 삽입")
    print("=" * 80)
    print()

    # 비동기 병렬 삽입 실행
    asyncio.run(insert_all_jsons_async())
