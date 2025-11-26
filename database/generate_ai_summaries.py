"""
AI 요약 생성 스크립트 (비동기 병렬 처리)

이미 생성된 SQLite DB의 agenda_chunks를 읽어서 AI 요약을 생성하고
agendas 테이블의 ai_summary, key_issues를 업데이트합니다.

사용법:
    python database/generate_ai_summaries.py

특징:
    - 비동기 병렬 처리 (10개 안건 동시 처리)
    - 속도 약 10배 향상
"""

import json
import sqlite3
import os
import asyncio
import threading
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 프로젝트 루트를 Python path에 추가
sys.path.append(str(Path(__file__).parent.parent))

from utils.cost_tracker import CostTracker

load_dotenv()

# Gemini 설정
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    print("✅ Gemini 2.5 Flash 초기화 성공")
else:
    client = None
    print("⚠️ GOOGLE_API_KEY 없음 - AI 요약 생성 불가")
    exit(1)

# SQLite DB 경로
SQLITE_DB_PATH = "data/sqlite_DB/agendas.db"

# 전역 카운터 (스레드 안전)
lock = threading.Lock()
completed_count = 0
failed_count = 0


def chunk_text(text, chunk_size=2000):
    """텍스트를 일정 크기로 청킹 (글자 수 기준)"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks


async def summarize_text_chunk_async(text_chunk, agenda_title, chunk_index, cost_tracker=None):
    """텍스트 청크 하나를 요약 (비동기)"""
    if not client or not text_chunk.strip():
        return None

    try:
        prompt = f"""안건 '{agenda_title}'의 일부 내용입니다:

{text_chunk}

위 내용을 간결하게 요약하세요. 핵심 내용을 중심으로 요약문만 반환하세요."""

        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        summary = response.text.strip()

        # 비용 추적
        if cost_tracker and hasattr(response, 'usage_metadata'):
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            cost_tracker.add_gemini_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model="gemini-2.5-flash"
            )

        # 비동기 대기 (다른 작업 가능)
        await asyncio.sleep(1)

        return summary
    except Exception as e:
        print(f"  ⚠️ 청크 요약 실패 (청크 {chunk_index}): {e}")
        await asyncio.sleep(2)
        return None


async def summarize_agenda_async(chunk_summaries, agenda_title, cost_tracker=None):
    """청크 요약들을 합쳐서 최종 요약 (비동기)"""
    if not client or not chunk_summaries:
        return None

    try:
        combined = "\n\n".join([s for s in chunk_summaries if s])

        if not combined.strip():
            return None

        prompt = f"""안건 '{agenda_title}'에 대한 요약들입니다:

{combined}

위 내용을 통합하여 150자 이내로 최종 요약하세요.
- 안건의 핵심 목적
- 주요 논의 내용
- 결론 또는 결과

요약문만 반환하세요."""

        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        summary = response.text.strip()

        # 비용 추적
        if cost_tracker and hasattr(response, 'usage_metadata'):
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            cost_tracker.add_gemini_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model="gemini-2.5-flash"
            )

        await asyncio.sleep(1)

        # 200자 넘으면 자르기 (LLM이 150자로 생성하므로 보통 200자 이하)
        if len(summary) > 200:
            summary = summary[:200]

        return summary
    except Exception as e:
        print(f"  ⚠️ 최종 요약 실패: {e}")
        await asyncio.sleep(2)
        return None


async def extract_key_issues_async(chunk_summaries, agenda_title, cost_tracker=None):
    """핵심 의제 추출 (비동기)"""
    if not client or not chunk_summaries:
        return None

    try:
        combined = "\n\n".join([s for s in chunk_summaries if s])

        if not combined.strip():
            return None

        prompt = f"""안건 '{agenda_title}'에 대한 요약들입니다:

{combined}

이 안건의 핵심 의제를 추출하세요.
- 개수는 안건의 복잡도에 따라 자유롭게 결정하세요 (단, 너무 많으면 안 됩니다)
- 각 의제는 한 줄로 간결하게 작성하세요
- JSON 배열 형식으로만 반환하세요

예시: ["의제1", "의제2", "의제3"]"""

        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        text = response.text.strip()

        # 비용 추적
        if cost_tracker and hasattr(response, 'usage_metadata'):
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            cost_tracker.add_gemini_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model="gemini-2.5-flash"
            )

        await asyncio.sleep(1)

        # 1. 마크다운 코드블록 제거 (```json ... ``` 또는 ``` ... ```)
        text = text.strip()
        if text.startswith('```'):
            # 첫 번째 줄과 마지막 줄 제거
            lines = text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]  # 첫 줄 제거 (```json 또는 ```)
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]  # 마지막 줄 제거 (```)
            text = '\n'.join(lines).strip()

        # 2. JSON 파싱 시도
        if text.startswith('[') and text.endswith(']'):
            try:
                issues = json.loads(text)
                # 각 의제에서 따옴표, 쉼표 등 정제 (개수 제한 제거)
                cleaned_issues = []
                for issue in issues:
                    # 양쪽 공백, 따옴표, 쉼표, 대괄호 제거
                    cleaned = issue.strip().strip('"').strip("'").strip(',').strip()
                    if cleaned:
                        cleaned_issues.append(cleaned)
                return cleaned_issues
            except:
                pass

        # 3. JSON 파싱 실패 시 수동 파싱 (개수 제한 제거)
        lines = []
        for line in text.split('\n'):
            if line.strip():
                # 양쪽 공백, 하이픈, 따옴표, 쉼표, 대괄호 제거
                cleaned = line.strip().strip('- ').strip('"').strip("'").strip(',').strip('[').strip(']').strip()
                if cleaned:
                    lines.append(cleaned)
        return lines
    except Exception as e:
        print(f"  ⚠️ 핵심 의제 추출 실패: {e}")
        await asyncio.sleep(2)
        return None


async def process_single_agenda(agenda_id, agenda_title, combined_text, total, idx, cost_tracker=None):
    """단일 안건 처리 (비동기)"""
    global completed_count, failed_count

    try:
        if not combined_text or not combined_text.strip():
            print(f"[{idx}/{total}] ⚠️ {agenda_title[:50]}... - 텍스트 없음")
            with lock:
                failed_count += 1
            return None

        # 1단계: 청킹
        text_chunks = chunk_text(combined_text, chunk_size=2000)

        # 2단계: 각 청크 요약 (병렬)
        chunk_summary_tasks = [
            summarize_text_chunk_async(chunk, agenda_title, i+1, cost_tracker)
            for i, chunk in enumerate(text_chunks)
        ]
        chunk_summaries = await asyncio.gather(*chunk_summary_tasks)
        chunk_summaries = [s for s in chunk_summaries if s]

        if not chunk_summaries:
            print(f"[{idx}/{total}] ❌ {agenda_title[:50]}... - 청크 요약 실패")
            with lock:
                failed_count += 1
            return None

        # 3단계: 최종 요약 + 핵심 의제 (병렬)
        ai_summary_task = summarize_agenda_async(chunk_summaries, agenda_title, cost_tracker)
        key_issues_task = extract_key_issues_async(chunk_summaries, agenda_title, cost_tracker)

        ai_summary, key_issues = await asyncio.gather(ai_summary_task, key_issues_task)

        with lock:
            completed_count += 1

        print(f"[{idx}/{total}] ✅ {agenda_title[:50]}...")
        if ai_summary:
            print(f"   📝 {ai_summary[:80]}...")
        if key_issues:
            print(f"   🔍 {len(key_issues)}개 의제")

        return (agenda_id, ai_summary, key_issues)

    except Exception as e:
        print(f"[{idx}/{total}] ❌ {agenda_title[:50]}... - 오류: {e}")
        with lock:
            failed_count += 1
        return None


async def generate_ai_summaries_async():
    """비동기 병렬로 AI 요약 생성 (3개씩)

    Returns:
        CostTracker: 비용 추적 객체
    """
    global completed_count, failed_count

    if not client:
        print("\n⚠️ Gemini API 없음 - AI 요약 건너뜀")
        return None

    # DB 연결
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    # 모든 안건 조회
    cursor.execute('SELECT agenda_id, agenda_title, combined_text FROM agendas')
    agendas = cursor.fetchall()

    print("\n" + "=" * 80)
    print(f"🤖 AI 요약 생성 (비동기 병렬 처리 - 10개씩)")
    print("=" * 80)
    print(f"총 안건 수: {len(agendas)}개\n")

    # 초기화
    completed_count = 0
    failed_count = 0
    cost_tracker = CostTracker()

    # 10개씩 병렬 처리
    semaphore = asyncio.Semaphore(10)

    async def process_with_semaphore(agenda, idx):
        async with semaphore:
            return await process_single_agenda(
                agenda[0], agenda[1], agenda[2], len(agendas), idx, cost_tracker
            )

    tasks = [process_with_semaphore(agenda, idx) for idx, agenda in enumerate(agendas, 1)]
    results = await asyncio.gather(*tasks)

    # DB 업데이트
    print("\n💾 DB 업데이트 중...")
    for result in results:
        if result:
            agenda_id, ai_summary, key_issues = result
            cursor.execute('''
                UPDATE agendas
                SET ai_summary = ?, key_issues = ?
                WHERE agenda_id = ?
            ''', (
                ai_summary,
                json.dumps(key_issues, ensure_ascii=False) if key_issues else None,
                agenda_id
            ))

    conn.commit()
    conn.close()

    print("\n" + "=" * 80)
    print("📊 최종 결과")
    print("=" * 80)
    print(f"✅ 성공: {completed_count}개")
    print(f"❌ 실패: {failed_count}개")
    print("=" * 80)

    # 비용 요약 출력
    print("\n" + "=" * 80)
    print("💰 Step 4 비용 요약 (Gemini 2.5 Flash)")
    print("=" * 80)
    cost_tracker.print_summary()
    print()

    return cost_tracker


def generate_ai_summaries():
    """동기 래퍼 함수

    Returns:
        CostTracker: 비용 추적 객체
    """
    # Windows에서 nested asyncio.run() 지원
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    return asyncio.run(generate_ai_summaries_async())


if __name__ == "__main__":
    print("=" * 80)
    print("AI 요약 생성 스크립트")
    print("=" * 80)
    print()

    # AI 요약 생성
    generate_ai_summaries()

    print("\n✅ 모든 작업 완료!")
