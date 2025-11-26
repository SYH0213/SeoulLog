"""
하이브리드 파싱: 1단계 Gemini + 2단계 순수 코드

사용법:
    python extract_metadata_hybrid.py

특징:
    - 1단계: Gemini 2.5 Pro로 전체 txt 분석 → 안건 라인 매핑
    - 2단계: 순수 Python 코드로 발언 추출 (빠르고 안정적)
    - 비용 절감: Gemini 2.5 Flash 단계 제거
    - 속도 향상: 10배 이상 빠름
    - 안정성: JSON 파싱 오류 없음
"""

import requests
from bs4 import BeautifulSoup, NavigableString
import json
import os
import re
import logging
from datetime import datetime
from pathlib import Path
from google import genai
from google.genai import types
from typing import List, Dict, Optional

# 로그 설정
def setup_logging():
    """로그 설정"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"parsing_hybrid_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # AFC (Automatic Function Calling) 로그 숨기기
    logging.getLogger('google.genai').setLevel(logging.WARNING)
    logging.getLogger('google.ai.generativelanguage').setLevel(logging.WARNING)

    return logging.getLogger(__name__)

logger = setup_logging()


def extract_text_with_links(element):
    """HTML 요소에서 텍스트와 링크를 순서대로 추출"""
    result = []

    for child in element.children:
        if isinstance(child, NavigableString):
            text = str(child)
            if text:
                result.append({"type": "text", "content": text})
        elif child.name == 'a':
            link_text = child.get_text()
            href = child.get('href', '')

            if href.startswith('/'):
                full_url = f"https://ms.smc.seoul.kr{href}"
            elif href.startswith('http'):
                full_url = href
            else:
                full_url = href

            result.append({"type": "link", "text": link_text, "url": full_url})
        elif child.name == 'br':
            result.append({"type": "text", "content": "\n"})
        elif child.name == 'hr':
            result.append({"type": "separator", "content": "---"})
        else:
            result.extend(extract_text_with_links(child))

    return result


def extract_reference_materials(content_list):
    """
    (참고) 섹션에서 참고자료 링크 추출

    Returns:
        [{"title": "문서명", "url": "다운로드 URL"}]
    """
    attachments = []
    in_reference = False
    pending_links = []

    for i, item in enumerate(content_list):
        # (참고) 시작 감지
        if item.get("type") == "text" and "(참고)" in item.get("content", ""):
            in_reference = True
            pending_links = []
            continue

        # (회의록 끝에 실음) 감지 시 종료
        if in_reference and item.get("type") == "text" and "회의록 끝에 실음" in item.get("content", ""):
            in_reference = False
            # pending_links를 attachments에 추가
            attachments.extend(pending_links)
            pending_links = []
            continue

        # (참고) 구간 내의 링크 수집
        if in_reference and item.get("type") == "link":
            # PDF/HWP 다운로드 링크만 추출
            url = item.get("url", "")
            if "appendixDownload" in url or url.endswith(('.pdf', '.hwp', '.docx')):
                pending_links.append({
                    "title": item.get("text", "").strip(),
                    "url": url
                })

    return attachments


def crawl_url(url: str) -> Dict:
    """URL 크롤링하여 txt 형식으로 반환"""
    logger.info(f"크롤링 시작: {url}")
    print(f"🌐 크롤링 시작: {url}\n")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.encoding = 'utf-8'

    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code}")

    soup = BeautifulSoup(response.text, 'html.parser')
    canvas = soup.find('div', id='canvas')

    if not canvas:
        raise Exception("메인 컨텐츠를 찾을 수 없습니다")

    title = soup.title.string if soup.title else "제목 없음"
    title = title.strip()

    # 텍스트 + 링크 추출
    extracted = extract_text_with_links(canvas)

    # 첨부 문서 추출 (참고 섹션에서)
    attachments = extract_reference_materials(extracted)

    # txt 변환
    lines = []
    for item in extracted:
        if item['type'] == 'text':
            text = item['content'].strip()
            if text:
                lines.append(text)
        elif item['type'] == 'separator':
            lines.append('---')

    txt_content = '\n'.join(lines)

    # 참고자료 제거
    if '(회의록 끝에 실음)' in txt_content:
        txt_content = txt_content.split('(회의록 끝에 실음)')[0]
    if '(참고)' in txt_content:
        txt_content = txt_content.split('(참고)')[0]

    logger.info(f"크롤링 완료: {len(txt_content)} bytes, 첨부 {len(attachments)}개")
    print(f"✅ 크롤링 완료: {len(txt_content):,} bytes\n")
    print(f"📎 첨부 문서: {len(attachments)}개\n")

    return {
        "title": title,
        "url": url,
        "content": txt_content,
        "attachments": attachments
    }


def extract_agenda_mapping(
    txt_content: str,
    title: str,
    url: str,
    api_key: str,
    attachments: List[Dict] = None,
    model: str = "gemini-2.5-pro",
    max_retries: int = 3
) -> Dict:
    """
    1단계: Gemini로 안건 라인 매핑 추출

    Args:
        txt_content: 회의록 텍스트
        title: 회의록 제목
        url: 회의록 URL
        api_key: Google API Key
        attachments: 첨부 문서 목록 [{"title": "...", "url": "..."}]
        model: 사용할 모델
        max_retries: 최대 재시도 횟수 (기본: 3)

    Returns:
        {
            "meeting_info": {
                "title": "...",
                "url": "...",
                "date": "YYYY.MM.DD"
            },
            "agenda_mapping": [
                {
                    "agenda_title": "안건 제목",
                    "line_start": 1,
                    "line_end": 50,
                    "speakers": ["발언자1", "발언자2"],
                    "attachments": [{"title": "...", "url": "..."}]
                },
                ...
            ]
        }
    """
    logger.info("1단계: 안건 라인 매핑 추출 시작")
    print("=" * 80)
    print("1단계: 안건 라인 매핑 추출 (Gemini 2.5 Pro)")
    print("=" * 80)
    print()

    # 재시도 로직
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                print(f"🔄 재시도 {attempt}/{max_retries}...")
                logger.info(f"재시도 {attempt}/{max_retries}")

            result, tokens = _extract_agenda_mapping_once(
                txt_content, title, url, api_key, attachments, model
            )
            return result, tokens

        except json.JSONDecodeError as e:
            if attempt < max_retries:
                print(f"⚠️  JSON 파싱 실패 (시도 {attempt}/{max_retries}): {e}")
                print(f"   재시도합니다...")
                logger.warning(f"JSON 파싱 실패 (시도 {attempt}/{max_retries}): {e}")
                continue
            else:
                print(f"❌ 최대 재시도 횟수 ({max_retries}회) 도달, 실패")
                logger.error(f"최대 재시도 횟수 도달: {e}")
                raise
        except Exception as e:
            # JSON 파싱 외 다른 에러는 재시도 없이 바로 raise
            raise


def _extract_agenda_mapping_once(
    txt_content: str,
    title: str,
    url: str,
    api_key: str,
    attachments: List[Dict] = None,
    model: str = "gemini-2.5-pro"
) -> Dict:
    """
    1단계: Gemini로 안건 라인 매핑 추출 (단일 시도)

    재시도 로직 없이 한 번만 실행하는 내부 함수
    """

    client = genai.Client(api_key=api_key)

    # 라인 번호 추가
    lines = txt_content.split('\n')
    numbered_text = ""
    for i, line in enumerate(lines, 1):
        numbered_text += f"{i:4d} | {line}\n"

    # 첨부 문서 정보를 텍스트로 변환 (Gemini에게 전달)
    attachments_text = ""
    if attachments:
        attachments_text = "\n\n첨부 문서 목록:\n"
        for idx, att in enumerate(attachments, 1):
            attachments_text += f"{idx}. {att['title']} (URL: {att['url']})\n"

    prompt = f"""다음은 서울시의회 회의록입니다. 이 회의록을 분석하여 안건별 라인 번호 매핑을 추출해주세요.

회의록 제목: {title}
회의록 URL: {url}
{attachments_text}

회의록 내용 (라인 번호 포함):
{numbered_text}

작업:
1. meeting_info 추출:
   - meeting_url: 위에 제공된 "회의록 URL"을 그대로 사용
   - date: 제목에서 날짜 추출 (YYYY.MM.DD 형식)

2. agenda_mapping 추출:

**안건 식별 규칙:**

1단계: "심사된안건" 또는 "의사일정" 섹션에서 안건 목록 추출
   - "1. 2. 3. ..." 형태의 안건 목록을 모두 찾으세요
   - agenda_title은 번호를 제외한 순수 안건명만 사용
   - 예: "1. 기획조정실 현안 업무보고" → "기획조정실 현안 업무보고"

2단계: 각 안건의 논의 구간 지정 (일괄상정 안건 처리 규칙)
   - "---" 구분선, "○위원장" 또는 "○의장" 발언, 안건명 언급을 참고하여 line_start, line_end 지정

   **본회의 일괄상정 안건 처리:**
   - 회의 제목에 "본회의"가 포함되고 여러 안건이 일괄 상정된 경우
   - 각 안건별로 "의사일정 제X항은 가결되었음" 또는 "의사일정 제X항을 표결하겠습니다" 형태의 개별 표결이 있는지 확인
   - 개별 표결이 있으면 → **각 안건을 반드시 분리**
     * 각 안건의 line_start: 해당 안건 표결 시작 라인 (예: "의사일정 제24항 ... 표결하겠습니다")
     * 각 안건의 line_end: 해당 안건 표결 종료 라인 (예: "가결되었음을 선포합니다" 또는 다음 안건 시작 직전)
     * 각 안건의 status: 개별 표결 결과 반영

   **위원회 일괄상정 안건 처리 (본회의 외):**
   - 여러 안건이 "일괄 상정" 또는 "일괄해서 상정"된 경우 본문을 분석하여 판단:

   조건 A (통합 처리): 다음 조건을 **모두** 만족하면 통합
     ✓ 일괄 제안설명 (각 안건별 개별 설명 없음)
     ✓ 일괄 검토보고 (각 안건별 개별 검토 없음)
     ✓ 일괄 질의응답 (안건 구분 없이 전체에 대한 질의)
     ✓ 일괄 표결 (각 안건별 개별 표결 없음)
     → agenda_title을 쉼표로 연결: "안건1,안건2,안건3,안건4"
     → line_start/line_end는 전체 논의 구간 하나로 지정
     → status는 전체 안건의 공통 처리 결과

   조건 B (분리 처리): 다음 조건 중 **하나라도** 해당하면 분리
     ✓ 각 안건별로 개별 제안설명이 구분되어 있음
     ✓ 각 안건별로 개별 검토보고가 구분되어 있음
     ✓ 각 안건에 대한 개별 질의응답이 "---" 구분선으로 나뉘어 있음
     ✓ 각 안건별로 개별 표결이 진행됨
     → 각 안건을 별도 agenda로 분리
     → 각각의 line_start/line_end 개별 지정
     → 각각의 status 개별 기록

   **판단 예시:**
   예시 1 (통합): "의사일정 제4항~제8항을 일괄 상정" → 위원장이 5개 안건을 한번에 설명 → 전문위원이 일괄 검토보고 → 질의응답 → "제4항부터 제8항까지 원안대로 의결" (한번에 표결)
   → agenda_title: "안건4,안건5,안건6,안건7,안건8"

   예시 2 (분리): "의사일정 제24항~제33항을 일괄 상정" → 심사보고는 생략 → 하지만 "제24항을 표결하겠습니다" → "제25항을 표결하겠습니다" → ... (각각 개별 투표)
   → 10개의 개별 agenda로 분리

**회의록 구조 이해:**
- 회의록은 "---" 구분선으로 섹션이 나뉘어져 있음
- 각 "---" 구분선 사이에 실제 발언("○위원장", "○위원" 등)이 포함된 구간은 하나의 안건으로 추출
- 첨부 문서(download_url)는 해당 안건 논의 직후에 "(참고)" 형태로 제공되며, 바로 앞 안건의 attachments에 매핑
- line_start는 실제 발언이 시작되는 첫 라인, line_end는 발언이 끝나는 마지막 라인 ("---" 구분선이나 "(참고)" 섹션 직전)

**안건 외 섹션:**
- 안건 목록에 없어도 실제 회의 내용(개의, 질의응답, 5분자유발언, 산회 등)은 모두 포함
- 섹션 성격에 맞는 적절한 제목 사용

**첨부 문서 매칭:**
- 회의록 본문에서 마크다운 링크 형식의 첨부 문서를 찾아서 매칭
  * 형식: [문서명](https://ms.smc.seoul.kr/record/appendixDownload.do?key=...)
  * (참고) 섹션의 링크는 바로 직전 안건에 속함
- 각 안건에 해당하는 첨부 문서를 attachments 필드에 배열로 추가
- 안건과 관련 없는 첨부 문서는 포함하지 않음
- 첨부 문서가 없는 안건은 attachments를 빈 배열로 설정
- **중요**: URL이 있으면 반드시 포함하고, URL이 없으면 null로 설정하지 말고 해당 항목을 제외

**중요:**
- 모든 안건을 빠짐없이 포함 (단 하나도 누락 금지)
- "(회의록 끝에 실음)" 또는 "(참고)" 섹션은 제외
- speakers: 해당 구간의 발언자 목록 (○ 다음 이름, 발언 순서대로)

**안건 타입 분류:**
각 안건에 agenda_type 필드를 추가하여 다음 중 하나로 분류:
- "legislation": 조례안, 규칙안 등 입법 안건
- "report": 업무보고, 현안보고 등 보고 안건
- "budget": 예산안, 결산 관련 안건
- "consent": 동의안, 승인안, 의견청취 안건
- "procedural": 개의, 개회, 폐회, 산회 등 절차적 안건
- "personnel": 위원장 선거, 위원 선임 등 인사 안건
- "discussion": 질의응답, 5분자유발언 등 토론
- "other": 기타

**안건 상태 추출:**
각 안건의 처리 상태를 파악하여 status 필드에 기록:
- 회의록에서 해당 안건이 어떻게 처리되었는지 찾아서 기록
- 가능한 상태 값:
  * "원안가결": "원안가결", "가결되었음", "의결되었음" 등
  * "수정가결": "수정가결", "일부 수정하여 가결" 등
  * "부결": "부결되었음"
  * "본회의 상정": "본회의에 부의", "본회의 상정" 등
  * "위원회 심사중": "위원회에 상정", "심사 중" 등
  * "접수": 상태가 명시되지 않은 경우 기본값
- 회의록에 명시된 표현을 그대로 사용 (예: "원안가결되었음을 선포합니다" → "원안가결")
- 상태가 불명확하면 "접수" 사용

JSON 출력 형식:
{{{{
  "meeting_info": {{{{
    "title": "{title}",
    "meeting_url": "{url}",
    "date": "YYYY.MM.DD"
  }}}},
  "agenda_mapping": [
    {{{{
      "agenda_title": "안건명",
      "agenda_type": "legislation",
      "status": "원안가결",
      "line_start": 1,
      "line_end": 50,
      "speakers": ["발언자1", "발언자2"],
      "attachments": [
        {{{{"title": "문서명", "download_url": "첨부파일 다운로드 URL"}}}}
      ]
    }}}}
  ]
}}}}

규칙:
- 순수 JSON만 출력
- agenda_mapping은 시간 순서대로 배열
- line_start, line_end는 실제 라인 번호 사용
- attachments는 해당 안건에 속한 첨부 문서만 포함 (없으면 빈 배열)
- agenda_type은 반드시 위 8가지 중 하나로 분류
- status는 회의록에서 추출한 실제 처리 상태 (없으면 "접수")
"""

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            )
        )
    except Exception as e:
        error_msg = f"Gemini API 호출 실패: {e}"
        print(f"❌ {error_msg}")
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        raise

    # finish_reason 확인 (디버깅)
    if response.candidates and len(response.candidates) > 0:
        finish_reason = getattr(response.candidates[0], 'finish_reason', None)
        if finish_reason:
            print(f"🔍 Finish Reason: {finish_reason}")
            if finish_reason not in ['STOP', 1]:  # STOP = 정상 완료
                print(f"⚠️  응답이 비정상적으로 종료됨: {finish_reason}")

    response_text = None
    try:
        response_text = response.text
    except Exception as e:
        print(f"❌ response.text 추출 실패: {e}")
        logger.error(f"response.text 추출 실패: {e}")

    if not response_text:
        print("⚠️  response.text가 비어있음, candidates에서 추출 시도...")
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    if len(candidate.content.parts) > 0:
                        part = candidate.content.parts[0]
                        if hasattr(part, 'text'):
                            response_text = part.text
                            print(f"✅ candidates에서 추출 성공 (길이: {len(response_text)})")

    if not response_text:
        error_msg = "Gemini 응답이 비어있습니다"
        print(f"❌ {error_msg}")
        print(f"   Response: {response}")
        if hasattr(response, 'prompt_feedback'):
            print(f"   Prompt Feedback: {response.prompt_feedback}")
        logger.error(error_msg)
        raise Exception(error_msg)

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 에러: {e}")
        print(f"   Response text (처음 500자): {response_text[:500]}")

        # 전체 응답을 파일로 저장
        error_log_dir = Path("logs/gemini_errors")
        error_log_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = error_log_dir / f"json_error_{timestamp}.txt"
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"=== JSON 파싱 에러 ===\n")
            f.write(f"회의록: {title}\n")
            f.write(f"URL: {url}\n")
            f.write(f"에러: {e}\n")
            f.write(f"모델: {model}\n")
            f.write(f"\n=== Gemini 응답 전문 ===\n")
            f.write(response_text)
        print(f"   전체 응답 저장: {error_file}")

        logger.error(f"JSON 파싱 에러: {e}")
        raise

    # 토큰 정보
    tokens = {"input": 0, "output": 0}
    if hasattr(response, 'usage_metadata'):
        tokens["input"] = getattr(response.usage_metadata, 'prompt_token_count', 0)
        tokens["output"] = getattr(response.usage_metadata, 'candidates_token_count', 0)

    logger.info(f"1단계 완료: {len(result['agenda_mapping'])}개 안건, 토큰={tokens['input']}+{tokens['output']}")
    print(f"✅ 안건 매핑 추출 완료: {len(result['agenda_mapping'])}개")
    print(f"📊 토큰: input={tokens['input']:,}, output={tokens['output']:,}")
    print()

    return result, tokens


def parse_speaker_line(line: str) -> tuple:
    """
    발언자 라인 파싱

    예: "○의장 최호정  안녕하세요." → ("의장 최호정", "안녕하세요.")
    예: "○위원장 [서상열](url)  안녕하세요." → ("위원장 서상열", "안녕하세요.")
    """
    # 마크다운 링크 제거 함수
    def remove_markdown_links(text: str) -> str:
        # [텍스트](url) → 텍스트
        return re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # ○ 다음 공백 제거하고 파싱
    match = re.match(r'^○\s*(.+?)\s{2,}(.+)$', line)
    if match:
        speaker = remove_markdown_links(match.group(1).strip())
        text = match.group(2).strip()
        return speaker, text

    # 발언자만 있는 경우 (다음 줄부터 내용)
    match = re.match(r'^○\s*(.+)$', line)
    if match:
        speaker = remove_markdown_links(match.group(1).strip())
        return speaker, ""

    return None, None


def split_long_text(text: str, max_length: int = 500) -> List[str]:
    """
    긴 텍스트를 문장 단위로 분할

    Args:
        text: 분할할 텍스트
        max_length: 최대 길이

    Returns:
        분할된 텍스트 리스트
    """
    if len(text) <= max_length:
        return [text]

    # 문장 단위로 분할
    sentences = re.split(r'([.?!])\s+', text)

    chunks = []
    current_chunk = ""

    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        if i + 1 < len(sentences):
            sentence += sentences[i + 1]  # 마침표 추가

        if len(current_chunk) + len(sentence) > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def parse_section_pure(section_text: str, agenda_title: str, speakers: List[str], previous_speaker: str = None) -> List[Dict]:
    """
    순수 코드로 섹션 파싱

    Args:
        section_text: 회의록 텍스트
        agenda_title: 안건명
        speakers: 발언자 목록 (1단계에서 제공)
        previous_speaker: 이전 구간의 마지막 발언자 (발언자 없을 때 사용)

    Returns:
        chunks 리스트
    """
    chunks = []
    lines = section_text.split('\n')

    current_speaker = previous_speaker  # 이전 발언자로 초기화
    current_text_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 구조적 요소 필터링
        if line.startswith('---'):
            continue
        if '(참고)' in line:
            continue
        if '(회의록 끝에 실음)' in line:
            continue
        # 안건 번호 라인 필터링 (예: "1. 안건명 [](url)" 형태)
        if re.match(r'^\d+\.\s+.+\[\]\(https?://', line):
            continue
        # 시간 표기 필터링 (예: (16신 22분), (11시 23분), (14시 16분))
        if re.match(r'^\([0-9]{1,2}[시신]\s*[0-9]{0,2}\s*분?\)$', line.strip()):
            continue

        # ○로 시작하는 발언자 라인인지 확인
        if line.startswith('○'):
            # 이전 발언 저장
            if current_speaker and current_text_lines:
                full_text = ' '.join(current_text_lines).strip()

                # 500자 넘으면 분할
                text_chunks = split_long_text(full_text, max_length=500)

                for text_chunk in text_chunks:
                    chunks.append({
                        "speaker": current_speaker,
                        "agenda": agenda_title,
                        "text": text_chunk
                    })

            # 새 발언자 시작
            speaker, first_text = parse_speaker_line(line)

            if speaker:
                current_speaker = speaker
                current_text_lines = [first_text] if first_text else []
        else:
            # 발언 내용 계속
            if current_speaker:
                current_text_lines.append(line)

    # 마지막 발언 저장
    if current_speaker and current_text_lines:
        full_text = ' '.join(current_text_lines).strip()
        text_chunks = split_long_text(full_text, max_length=500)

        for text_chunk in text_chunks:
            chunks.append({
                "speaker": current_speaker,
                "agenda": agenda_title,
                "text": text_chunk
            })

    return chunks


def parse_with_pure_code(txt_content: str, agenda_mapping: List[Dict]) -> List[Dict]:
    """
    2단계: 순수 코드로 발언 추출

    Args:
        txt_content: 회의록 본문 (헤더 이미 제거됨)
        agenda_mapping: 1단계 결과 (안건 매핑)

    Returns:
        모든 chunks
    """
    logger.info("2단계: 순수 코드로 발언 추출 시작")
    print("=" * 80)
    print("2단계: 순수 코드로 발언 추출")
    print("=" * 80)
    print()

    # txt_content는 이미 헤더가 제거된 상태로 전달됨
    lines = txt_content.split('\n')

    all_chunks = []
    last_speaker = None  # 이전 발언자 추적

    for idx, agenda in enumerate(agenda_mapping, 1):
        agenda_title = agenda['agenda_title']
        line_start = agenda['line_start'] - 1  # 0-indexed
        line_end = agenda['line_end']
        speakers = agenda.get('speakers', [])

        # 라인 범위 추출
        section_lines = lines[line_start:line_end]
        section_text = '\n'.join(section_lines)

        # 파싱 (이전 발언자 전달)
        chunks = parse_section_pure(section_text, agenda_title, speakers, last_speaker)

        # 청크가 있으면 마지막 발언자 업데이트
        if chunks:
            last_speaker = chunks[-1]['speaker']

        msg = f"[{idx}/{len(agenda_mapping)}] {len(chunks)}개 발언 추출: {agenda_title[:50]}..."
        logger.info(msg)
        print(f"  ✓ {msg}")

        all_chunks.extend(chunks)

    logger.info(f"2단계 완료: 총 {len(all_chunks)}개 발언 추출")
    print()
    print(f"✅ 총 {len(all_chunks)}개 발언 추출 완료!")
    print()

    return all_chunks


def extract_metadata_from_url(
    url: str,
    api_key: str,
    stage1_model: str = "gemini-2.5-pro",
    verbose: bool = True
) -> Dict:
    """
    URL에서 직접 크롤링 + 하이브리드 파싱 (첨부 문서 포함)

    Args:
        url: 회의록 URL
        api_key: Google API Key
        stage1_model: 1단계 모델 (기본: gemini-2.5-pro)
        verbose: 상세 출력 여부 (기본: True)

    Returns:
        {
            "meeting_info": {...},
            "chunks": [...],
            "usage": {...}
        }
    """
    logger.info(f"URL 기반 파싱 시작: {url}")

    if verbose:
        print("=" * 100)
        print("하이브리드 파싱: URL 크롤링 + 1단계 Gemini + 2단계 순수 코드")
        print("=" * 100)
        print()

    # URL 크롤링
    crawled_data = crawl_url(url)
    txt_content = crawled_data['content']
    title = crawled_data['title']
    attachments = crawled_data.get('attachments', [])

    if verbose:
        print(f"📄 제목: {title}")
        print(f"📏 크기: {len(txt_content):,} bytes")
        print(f"📎 첨부: {len(attachments)}개")
        print()

    # 1단계: 안건 매핑 추출 (Gemini)
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    if not verbose:
        sys.stdout = StringIO()

    stage1_result, tokens = extract_agenda_mapping(txt_content, title, url, api_key, attachments, stage1_model)

    if not verbose:
        sys.stdout = old_stdout

    # 2단계: 발언 추출 (순수 코드)
    if not verbose:
        sys.stdout = StringIO()

    # txt_content를 full format으로 재구성 (헤더 포함)
    full_content = f"제목: {title}\nURL: {url}\n{'=' * 80}\n\n{txt_content}"
    chunks = parse_with_pure_code(full_content, stage1_result['agenda_mapping'])

    if not verbose:
        sys.stdout = old_stdout

    # 최종 결과 (agenda_mapping 포함)
    final_result = {
        "meeting_info": stage1_result['meeting_info'],
        "agenda_mapping": stage1_result['agenda_mapping'],  # ⭐ 추가: 첨부 문서 정보
        "chunks": chunks,
        "usage": {
            "stage1_model": stage1_model,
            "stage2_method": "pure_python_code",
            "stage1_tokens": tokens,
            "total_chunks": len(chunks)
        }
    }

    logger.info(f"URL 기반 파싱 완료: {len(chunks)}개 발언, {len(attachments)}개 첨부")

    if verbose:
        print("=" * 100)
        print("✅ URL 기반 파싱 완료!")
        print("=" * 100)
        print(f"총 발언 수: {len(chunks)}개")
        print(f"총 첨부 수: {len(attachments)}개")
        print(f"Stage 1 토큰: {tokens['input']:,} + {tokens['output']:,}")
        print(f"Stage 2 방식: 순수 Python 코드 (비용 0원)")
        print()

    return final_result


def extract_metadata_hybrid(
    txt_path: str,
    api_key: str,
    stage1_model: str = "gemini-2.5-pro",
    verbose: bool = True
) -> Dict:
    """
    하이브리드 파싱: 1단계 Gemini + 2단계 순수 코드 (txt 파일 기반)

    Note: txt 파일에는 첨부 문서 정보가 없으므로 attachments는 빈 배열입니다.
          첨부 문서가 필요한 경우 extract_metadata_from_url()을 사용하세요.

    Args:
        txt_path: txt 파일 경로
        api_key: Google API Key
        stage1_model: 1단계 모델 (기본: gemini-2.5-pro)
        verbose: 상세 출력 여부 (기본: True)

    Returns:
        {
            "meeting_info": {...},
            "chunks": [...],
            "usage": {...}
        }
    """
    logger.info(f"하이브리드 파싱 시작: {txt_path}")

    if verbose:
        print("=" * 100)
        print("하이브리드 파싱: 1단계 Gemini + 2단계 순수 코드")
        print("=" * 100)
        print()

    # txt/md 파일 읽기
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 헤더에서 메타데이터 추출 (TXT와 MD 둘 다 지원)
    lines_raw = content.split('\n')

    # 제목 추출
    if lines_raw and lines_raw[0].startswith('# '):
        # MD 형식: # 제목
        title = lines_raw[0].replace('# ', '').strip()
    elif lines_raw and lines_raw[0].startswith('제목: '):
        # TXT 형식: 제목: xxx
        title = lines_raw[0].replace('제목: ', '').strip()
    else:
        title = ""

    # URL 추출
    url = ""
    for line in lines_raw[:10]:  # 첫 10줄에서 URL 찾기
        if line.startswith('**URL**:'):
            # MD 형식: **URL**: https://...
            url = line.replace('**URL**:', '').strip()
            break
        elif line.startswith('URL: '):
            # TXT 형식: URL: https://...
            url = line.replace('URL: ', '').strip()
            break

    # 본문만 추출 (구분선 이후 또는 크롤링 시간 이후)
    separator_index = content.find('=' * 80)
    if separator_index != -1:
        txt_content = content[separator_index + 80:].strip()
    else:
        # MD 파일의 경우 크롤링 시간 이후부터
        crawl_time_index = content.find('**크롤링 시간**:')
        if crawl_time_index != -1:
            # 다음 줄부터 시작
            remaining = content[crawl_time_index:]
            next_line = remaining.find('\n')
            if next_line != -1:
                txt_content = remaining[next_line + 1:].strip()
            else:
                txt_content = content
        else:
            txt_content = content

    if verbose:
        print(f"📄 파일: {txt_path}")
        print(f"📝 제목: {title}")
        print(f"📏 크기: {len(txt_content):,} bytes")
        print()

    # 1단계: 안건 매핑 추출 (Gemini)
    # 임시로 print 억제
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    if not verbose:
        sys.stdout = StringIO()

    # txt에서 attachments는 추출할 수 없으므로 빈 리스트 전달
    stage1_result, tokens = extract_agenda_mapping(txt_content, title, url, api_key, [], stage1_model)

    if not verbose:
        sys.stdout = old_stdout

    # 1단계 결과 저장 (디버깅용)
    stage1_dir = Path("data/result_txt_gemini")
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage1_filename = Path(txt_path).stem + "_stage1.json"
    stage1_path = stage1_dir / stage1_filename
    with open(stage1_path, 'w', encoding='utf-8') as f:
        json.dump(stage1_result, f, ensure_ascii=False, indent=2)
    logger.info(f"1단계 결과 저장: {stage1_path}")

    # 2단계: 발언 추출 (순수 코드)
    if not verbose:
        sys.stdout = StringIO()

    chunks = parse_with_pure_code(txt_content, stage1_result['agenda_mapping'])

    if not verbose:
        sys.stdout = old_stdout

    # 최종 결과 (agenda_mapping 포함 - attachments는 빈 배열)
    final_result = {
        "meeting_info": stage1_result['meeting_info'],
        "agenda_mapping": stage1_result['agenda_mapping'],  # ⭐ 추가: 안건 매핑 (attachments 빈 배열)
        "chunks": chunks,
        "usage": {
            "stage1_model": stage1_model,
            "stage2_method": "pure_python_code",
            "stage1_tokens": tokens,
            "total_chunks": len(chunks)
        }
    }

    logger.info(f"하이브리드 파싱 완료: {len(chunks)}개 발언")

    if verbose:
        print("=" * 100)
        print("✅ 하이브리드 파싱 완료!")
        print("=" * 100)
        print(f"총 발언 수: {len(chunks)}개")
        print(f"Stage 1 토큰: {tokens['input']:,} + {tokens['output']:,}")
        print(f"Stage 2 방식: 순수 Python 코드 (비용 0원)")
        print()

    return final_result


# ============================================================================
# Gemini 2.5 Flash 전용 함수 (개선된 프롬프트 적용)
# ============================================================================

def extract_agenda_mapping_flash(
    txt_content: str,
    title: str,
    url: str,
    api_key: str,
    attachments: List[Dict] = None,
    model: str = "gemini-2.5-flash",
    max_retries: int = 3
) -> Dict:
    """
    1단계: Gemini Flash로 안건 라인 매핑 추출 (개선된 프롬프트)

    Args:
        txt_content: 회의록 텍스트
        title: 회의록 제목
        url: 회의록 URL
        api_key: Google API Key
        attachments: 첨부 문서 목록 [{"title": "...", "url": "..."}]
        model: 사용할 모델 (기본: gemini-2.5-flash)
        max_retries: 최대 재시도 횟수 (기본: 3)

    Returns:
        {
            "meeting_info": {
                "title": "...",
                "url": "...",
                "date": "YYYY.MM.DD"
            },
            "agenda_mapping": [
                {
                    "agenda_title": "안건 제목",
                    "line_start": 1,
                    "line_end": 50,
                    "speakers": ["발언자1", "발언자2"],
                    "attachments": [{"title": "...", "url": "..."}]
                },
                ...
            ]
        }
    """
    logger.info("1단계: 안건 라인 매핑 추출 시작 (Flash)")
    print("=" * 80)
    print("1단계: 안건 라인 매핑 추출 (Gemini 2.5 Flash - 개선된 프롬프트)")
    print("=" * 80)
    print()

    # 재시도 로직
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                print(f"🔄 재시도 {attempt}/{max_retries}...")
                logger.info(f"재시도 {attempt}/{max_retries}")

            result, tokens = _extract_agenda_mapping_flash_once(
                txt_content, title, url, api_key, attachments, model
            )
            return result, tokens

        except json.JSONDecodeError as e:
            if attempt < max_retries:
                print(f"⚠️  JSON 파싱 실패 (시도 {attempt}/{max_retries}): {e}")
                print(f"   재시도합니다...")
                logger.warning(f"JSON 파싱 실패 (시도 {attempt}/{max_retries}): {e}")
                continue
            else:
                print(f"❌ 최대 재시도 횟수 ({max_retries}회) 도달, 실패")
                logger.error(f"최대 재시도 횟수 도달: {e}")
                raise
        except Exception as e:
            # JSON 파싱 외 다른 에러는 재시도 없이 바로 raise
            raise


def _extract_agenda_mapping_flash_once(
    txt_content: str,
    title: str,
    url: str,
    api_key: str,
    attachments: List[Dict] = None,
    model: str = "gemini-2.5-flash"
) -> Dict:
    """
    1단계: Gemini Flash로 안건 라인 매핑 추출 (단일 시도)

    재시도 로직 없이 한 번만 실행하는 내부 함수
    """

    client = genai.Client(api_key=api_key)

    # 라인 번호 추가
    lines = txt_content.split('\n')
    numbered_text = ""
    for i, line in enumerate(lines, 1):
        numbered_text += f"{i:4d} | {line}\n"

    # 첨부 문서 정보를 텍스트로 변환 (Gemini에게 전달)
    attachments_text = ""
    if attachments:
        attachments_text = "\n\n첨부 문서 목록:\n"
        for idx, att in enumerate(attachments, 1):
            attachments_text += f"{idx}. {att['title']} (URL: {att['url']})\n"

    prompt = f"""다음은 서울시의회 회의록입니다. 이 회의록을 분석하여 안건별 라인 번호 매핑을 추출해주세요.

회의록 제목: {title}
회의록 URL: {url}
{attachments_text}

회의록 내용 (라인 번호 포함):
{numbered_text}

작업:
1. meeting_info 추출:
   - meeting_url: 위에 제공된 "회의록 URL"을 그대로 사용
   - date: 제목에서 날짜 추출 (YYYY.MM.DD 형식)

2. agenda_mapping 추출:

**안건 식별 규칙:**

1단계: "심사된안건" 또는 "의사일정" 섹션에서 안건 목록 추출
   - "1. 2. 3. ..." 형태의 안건 목록을 모두 찾으세요
   - agenda_title은 번호를 제외한 순수 안건명만 사용
   - 예: "1. 기획조정실 현안 업무보고" → "기획조정실 현안 업무보고"

2단계: 각 안건의 논의 구간 지정 (일괄상정 안건 처리 규칙)
   - "---" 구분선, "○위원장" 또는 "○의장" 발언, 안건명 언급을 참고하여 line_start, line_end 지정

   **본회의 일괄상정 안건 처리:**
   - 회의 제목에 "본회의"가 포함되고 여러 안건이 일괄 상정된 경우
   - 각 안건별로 "의사일정 제X항은 가결되었음" 또는 "의사일정 제X항을 표결하겠습니다" 형태의 개별 표결이 있는지 확인
   - 개별 표결이 있으면 → **각 안건을 반드시 분리**
     * 각 안건의 line_start: 해당 안건 표결 시작 라인 (예: "의사일정 제24항 ... 표결하겠습니다")
     * 각 안건의 line_end: 해당 안건 표결 종료 라인 (예: "가결되었음을 선포합니다" 또는 다음 안건 시작 직전)
     * 각 안건의 status: 개별 표결 결과 반영

   **위원회 일괄상정 안건 처리 (본회의 외):**
   - 여러 안건이 "일괄 상정" 또는 "일괄해서 상정"된 경우 본문을 분석하여 판단:

   조건 A (통합 처리): 다음 조건을 **모두** 만족하면 통합
     ✓ 일괄 제안설명 (각 안건별 개별 설명 없음)
     ✓ 일괄 검토보고 (각 안건별 개별 검토 없음)
     ✓ 일괄 질의응답 (안건 구분 없이 전체에 대한 질의)
     ✓ 일괄 표결 (각 안건별 개별 표결 없음)
     → agenda_title을 쉼표로 연결: "안건1,안건2,안건3,안건4"
     → line_start/line_end는 전체 논의 구간 하나로 지정
     → status는 전체 안건의 공통 처리 결과

   조건 B (분리 처리): 다음 조건 중 **하나라도** 해당하면 분리
     ✓ 각 안건별로 개별 제안설명이 구분되어 있음
     ✓ 각 안건별로 개별 검토보고가 구분되어 있음
     ✓ 각 안건에 대한 개별 질의응답이 "---" 구분선으로 나뉘어 있음
     ✓ 각 안건별로 개별 표결이 진행됨
     → 각 안건을 별도 agenda로 분리
     → 각각의 line_start/line_end 개별 지정
     → 각각의 status 개별 기록

   **판단 예시:**
   예시 1 (통합): "의사일정 제4항~제8항을 일괄 상정" → 위원장이 5개 안건을 한번에 설명 → 전문위원이 일괄 검토보고 → 질의응답 → "제4항부터 제8항까지 원안대로 의결" (한번에 표결)
   → agenda_title: "안건4,안건5,안건6,안건7,안건8"

   예시 2 (분리): "의사일정 제24항~제33항을 일괄 상정" → 심사보고는 생략 → 하지만 "제24항을 표결하겠습니다" → "제25항을 표결하겠습니다" → ... (각각 개별 투표)
   → 10개의 개별 agenda로 분리

**회의록 구조 이해:**
- 회의록은 "---" 구분선으로 섹션이 나뉘어져 있음
- 각 "---" 구분선 사이에 실제 발언("○위원장", "○위원" 등)이 포함된 구간은 하나의 안건으로 추출
- 첨부 문서(download_url)는 해당 안건 논의 직후에 "(참고)" 형태로 제공되며, 바로 앞 안건의 attachments에 매핑
- line_start는 실제 발언이 시작되는 첫 라인 (예: ○위원장 김혜영  의사일정 제1항...), line_end는 발언이 끝나는 마지막 라인 ("---" 구분선이나 "(참고)" 섹션 직전)

**안건 외 섹션:**
- 안건 목록에 없어도 실제 회의 내용(개의, 질의응답, 5분자유발언, 산회 등)은 모두 포함
- 섹션 성격에 맞는 적절한 제목 사용

**첨부 문서 매칭:**
- 회의록 본문에서 마크다운 링크 형식의 첨부 문서를 찾아서 매칭
  * 형식: [문서명](https://ms.smc.seoul.kr/record/appendixDownload.do?key=...)
  * (참고) 섹션의 링크는 바로 직전 안건에 속함
- 각 안건에 해당하는 첨부 문서를 attachments 필드에 배열로 추가
- 안건과 관련 없는 첨부 문서는 포함하지 않음
- 첨부 문서가 없는 안건은 attachments를 빈 배열로 설정
- **중요**: URL이 있으면 반드시 포함하고, URL이 없으면 null로 설정하지 말고 해당 항목을 제외

**중요:**
- 모든 안건을 빠짐없이 포함 (단 하나도 누락 금지)
- "(회의록 끝에 실음)" 또는 "(참고)" 섹션은 제외
- speakers: 해당 구간의 발언자 목록 (○ 다음 이름, 발언 순서대로)

**안건 타입 분류:**
각 안건에 agenda_type 필드를 추가하여 다음 중 하나로 분류:
- "legislation": 조례안, 규칙안 등 입법 안건
- "report": 업무보고, 현안보고 등 보고 안건
- "budget": 예산안, 결산 관련 안건
- "consent": 동의안, 승인안, 의견청취 안건
- "procedural": 개의, 개회, 폐회, 산회 등 절차적 안건
- "personnel": 위원장 선거, 위원 선임 등 인사 안건
- "discussion": 질의응답, 5분자유발언 등 토론
- "other": 기타

**안건 상태 추출:**
각 안건의 처리 상태를 파악하여 status 필드에 기록:
- 회의록에서 해당 안건이 어떻게 처리되었는지 찾아서 기록
- 가능한 상태 값:
  * "원안가결": "원안가결", "가결되었음", "의결되었음" 등
  * "수정가결": "수정가결", "일부 수정하여 가결" 등
  * "부결": "부결되었음"
  * "본회의 상정": "본회의에 부의", "본회의 상정" 등
  * "위원회 심사중": "위원회에 상정", "심사 중" 등
  * "접수": 상태가 명시되지 않은 경우 기본값
- 회의록에 명시된 표현을 그대로 사용 (예: "원안가결되었음을 선포합니다" → "원안가결")
- 상태가 불명확하면 "접수" 사용

JSON 출력 형식:
{{{{
  "meeting_info": {{{{
    "title": "{title}",
    "meeting_url": "{url}",
    "date": "YYYY.MM.DD"
  }}}},
  "agenda_mapping": [
    {{{{
      "agenda_title": "안건명",
      "agenda_type": "legislation",
      "status": "원안가결",
      "line_start": 1,
      "line_end": 50,
      "speakers": ["발언자1", "발언자2"],
      "attachments": [
        {{{{"title": "문서명", "download_url": "첨부파일 다운로드 URL"}}}}
      ]
    }}}}
  ]
}}}}

규칙:
- 순수 JSON만 출력
- agenda_mapping은 시간 순서대로 배열
- line_start, line_end는 실제 라인 번호 사용
- attachments는 해당 안건에 속한 첨부 문서만 포함 (없으면 빈 배열)
- agenda_type은 반드시 위 8가지 중 하나로 분류
- status는 회의록에서 추출한 실제 처리 상태 (없으면 "접수")
"""

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            )
        )
    except Exception as e:
        error_msg = f"Gemini API 호출 실패: {e}"
        print(f"❌ {error_msg}")
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        raise

    # finish_reason 확인 (디버깅)
    if response.candidates and len(response.candidates) > 0:
        finish_reason = getattr(response.candidates[0], 'finish_reason', None)
        if finish_reason:
            print(f"🔍 Finish Reason: {finish_reason}")
            if finish_reason not in ['STOP', 1]:  # STOP = 정상 완료
                print(f"⚠️  응답이 비정상적으로 종료됨: {finish_reason}")

    response_text = None
    try:
        response_text = response.text
    except Exception as e:
        print(f"❌ response.text 추출 실패: {e}")
        logger.error(f"response.text 추출 실패: {e}")

    if not response_text:
        print("⚠️  response.text가 비어있음, candidates에서 추출 시도...")
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    if len(candidate.content.parts) > 0:
                        part = candidate.content.parts[0]
                        if hasattr(part, 'text'):
                            response_text = part.text
                            print(f"✅ candidates에서 추출 성공 (길이: {len(response_text)})")

    if not response_text:
        error_msg = "Gemini 응답이 비어있습니다"
        print(f"❌ {error_msg}")
        print(f"   Response: {response}")
        if hasattr(response, 'prompt_feedback'):
            print(f"   Prompt Feedback: {response.prompt_feedback}")
        logger.error(error_msg)
        raise Exception(error_msg)

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 에러: {e}")
        print(f"   Response text (처음 500자): {response_text[:500]}")

        # 전체 응답을 파일로 저장
        error_log_dir = Path("logs/gemini_errors")
        error_log_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = error_log_dir / f"json_error_{timestamp}.txt"
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"=== JSON 파싱 에러 ===\n")
            f.write(f"회의록: {title}\n")
            f.write(f"URL: {url}\n")
            f.write(f"에러: {e}\n")
            f.write(f"모델: {model}\n")
            f.write(f"\n=== Gemini 응답 전문 ===\n")
            f.write(response_text)
        print(f"   전체 응답 저장: {error_file}")

        logger.error(f"JSON 파싱 에러: {e}")
        raise

    # 토큰 정보
    tokens = {"input": 0, "output": 0}
    if hasattr(response, 'usage_metadata'):
        tokens["input"] = getattr(response.usage_metadata, 'prompt_token_count', 0)
        tokens["output"] = getattr(response.usage_metadata, 'candidates_token_count', 0)

    logger.info(f"1단계 완료 (Flash): {len(result['agenda_mapping'])}개 안건, 토큰={tokens['input']}+{tokens['output']}")
    print(f"✅ 안건 매핑 추출 완료 (Flash): {len(result['agenda_mapping'])}개")
    print(f"📊 토큰: input={tokens['input']:,}, output={tokens['output']:,}")
    print()

    return result, tokens


def extract_metadata_hybrid_flash(
    txt_path: str,
    api_key: str,
    verbose: bool = True
) -> Dict:
    """
    하이브리드 파싱 (Flash 전용): 1단계 Gemini Flash + 2단계 순수 코드 (txt/md 파일 기반)

    개선된 프롬프트를 사용하여 gemini-2.5-flash의 파싱 정확도를 향상시킵니다.

    Note: txt 파일에는 첨부 문서 정보가 없으므로 attachments는 빈 배열입니다.
          첨부 문서가 필요한 경우 extract_metadata_from_url()을 사용하세요.

    Args:
        txt_path: txt/md 파일 경로
        api_key: Google API Key
        verbose: 상세 출력 여부 (기본: True)

    Returns:
        {
            "meeting_info": {...},
            "agenda_mapping": [...],
            "chunks": [...],
            "usage": {...}
        }
    """
    logger.info(f"하이브리드 파싱 시작 (Flash): {txt_path}")

    if verbose:
        print("=" * 100)
        print("하이브리드 파싱 (Flash): 1단계 Gemini Flash + 2단계 순수 코드")
        print("=" * 100)
        print()

    # txt/md 파일 읽기
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 헤더에서 메타데이터 추출 (TXT와 MD 둘 다 지원)
    lines_raw = content.split('\n')

    # 제목 추출
    if lines_raw and lines_raw[0].startswith('# '):
        # MD 형식: # 제목
        title = lines_raw[0].replace('# ', '').strip()
    elif lines_raw and lines_raw[0].startswith('제목: '):
        # TXT 형식: 제목: xxx
        title = lines_raw[0].replace('제목: ', '').strip()
    else:
        title = ""

    # URL 추출
    url = ""
    for line in lines_raw[:10]:  # 첫 10줄에서 URL 찾기
        if line.startswith('**URL**:'):
            # MD 형식: **URL**: https://...
            url = line.replace('**URL**:', '').strip()
            break
        elif line.startswith('URL: '):
            # TXT 형식: URL: https://...
            url = line.replace('URL: ', '').strip()
            break

    # 본문만 추출 (구분선 이후 또는 크롤링 시간 이후)
    separator_index = content.find('=' * 80)
    if separator_index != -1:
        txt_content = content[separator_index + 80:].strip()
    else:
        # MD 파일의 경우 크롤링 시간 이후부터
        crawl_time_index = content.find('**크롤링 시간**:')
        if crawl_time_index != -1:
            # 다음 줄부터 시작
            remaining = content[crawl_time_index:]
            next_line = remaining.find('\n')
            if next_line != -1:
                txt_content = remaining[next_line + 1:].strip()
            else:
                txt_content = content
        else:
            txt_content = content

    if verbose:
        print(f"📄 파일: {txt_path}")
        print(f"📝 제목: {title}")
        print(f"📏 크기: {len(txt_content):,} bytes")
        print()

    # 1단계: 안건 매핑 추출 (Gemini Flash - 개선된 프롬프트) with 재시도
    import sys
    from io import StringIO
    old_stdout = sys.stdout

    max_retries = 3
    stage1_result = None
    tokens = {"input": 0, "output": 0}

    for attempt in range(max_retries + 1):
        if not verbose:
            sys.stdout = StringIO()

        # txt에서 attachments는 추출할 수 없으므로 빈 리스트 전달
        stage1_result, tokens = extract_agenda_mapping_flash(txt_content, title, url, api_key, [])

        if not verbose:
            sys.stdout = old_stdout

        # 중복 line range 검증
        has_overlap = False
        overlap_details = []
        agendas = stage1_result.get('agenda_mapping', [])

        for i in range(len(agendas)):
            for j in range(i + 1, len(agendas)):
                agenda1 = agendas[i]
                agenda2 = agendas[j]

                start1 = agenda1.get('line_start', 0)
                end1 = agenda1.get('line_end', 0)
                start2 = agenda2.get('line_start', 0)
                end2 = agenda2.get('line_end', 0)

                # 겹침 계산 (5줄 이상 겹치면 문제)
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > 5:
                    has_overlap = True
                    overlap_details.append({
                        'agenda1_title': agenda1.get('agenda_title', '(제목없음)')[:40],
                        'range1': f"{start1}-{end1}",
                        'agenda2_title': agenda2.get('agenda_title', '(제목없음)')[:40],
                        'range2': f"{start2}-{end2}",
                        'overlap': overlap
                    })

        if not has_overlap:
            if verbose and attempt > 0:
                print(f"✅ 재시도 성공 (시도 {attempt + 1}): 중복 line range 해결됨")
            break
        else:
            if attempt < max_retries:
                if verbose:
                    print(f"⚠️  시도 {attempt + 1}: 중복 line range 발견 ({len(overlap_details)}개)")
                    for detail in overlap_details[:3]:
                        print(f"   - {detail['agenda1_title']} ({detail['range1']}) ↔ {detail['agenda2_title']} ({detail['range2']}) | 겹침: {detail['overlap']}줄")
                    print(f"🔄 Stage 1 재시도 중...")
                logger.warning(f"중복 line range 발견, 재시도 {attempt + 1}/{max_retries}")
            else:
                if verbose:
                    print(f"❌ 최대 재시도 횟수 초과: 중복 line range가 여전히 존재함 ({len(overlap_details)}개)")
                    print(f"   계속 진행하지만 결과에 문제가 있을 수 있습니다.")
                logger.error(f"중복 line range 해결 실패 (최대 재시도 초과)")

    if not verbose:
        sys.stdout = old_stdout

    # 1단계 결과 저장 (디버깅용)
    stage1_dir = Path("data/result_txt_gemini_flash")
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage1_filename = Path(txt_path).stem + "_stage1_flash.json"
    stage1_path = stage1_dir / stage1_filename
    with open(stage1_path, 'w', encoding='utf-8') as f:
        json.dump(stage1_result, f, ensure_ascii=False, indent=2)
    logger.info(f"1단계 결과 저장 (Flash): {stage1_path}")

    # 2단계: 발언 추출 (순수 코드)
    if not verbose:
        sys.stdout = StringIO()

    chunks = parse_with_pure_code(txt_content, stage1_result['agenda_mapping'])

    if not verbose:
        sys.stdout = old_stdout

    # 최종 결과 (agenda_mapping 포함 - attachments는 빈 배열)
    final_result = {
        "meeting_info": stage1_result['meeting_info'],
        "agenda_mapping": stage1_result['agenda_mapping'],  # ⭐ 추가: 안건 매핑 (Flash)
        "chunks": chunks,
        "usage": {
            "stage1_model": "gemini-2.5-flash",
            "stage2_method": "pure_python_code",
            "stage1_tokens": tokens,
            "total_chunks": len(chunks)
        }
    }

    logger.info(f"하이브리드 파싱 완료 (Flash): {len(chunks)}개 발언")

    if verbose:
        print("=" * 100)
        print("✅ 하이브리드 파싱 완료 (Flash)!")
        print("=" * 100)
        print(f"총 발언 수: {len(chunks)}개")
        print(f"Stage 1 토큰: {tokens['input']:,} + {tokens['output']:,}")
        print(f"Stage 2 방식: 순수 Python 코드 (비용 0원)")
        print()

    return final_result


def main():
    """메인 함수"""
    from dotenv import load_dotenv
    import random
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY가 설정되지 않았습니다.")
        return

    # result 폴더의 모든 txt 파일 찾기
    result_dir = Path("result")
    all_txt_files = list(result_dir.glob("*/meeting_*.txt"))

    if len(all_txt_files) == 0:
        print("❌ result 폴더에 txt 파일이 없습니다.")
        return

    # 랜덤으로 3개 선택
    random.seed()  # 매번 다른 결과
    selected_files = random.sample(all_txt_files, min(3, len(all_txt_files)))

    print("=" * 100)
    print("🎲 랜덤으로 선택된 파일 3개 (수정된 파서 테스트)")
    print("=" * 100)
    for i, file in enumerate(selected_files, 1):
        print(f"{i}. {file.parent.name}")
    print()

    success_count = 0
    fail_count = 0

    for idx, txt_path in enumerate(selected_files, 1):
        print("=" * 100)
        print(f"[{idx}/3] {txt_path.parent.name}")
        print("=" * 100)
        print()

        try:
            result = extract_metadata_hybrid(
                txt_path=str(txt_path),
                api_key=api_key,
                stage1_model="gemini-2.5-pro",
                verbose=True
            )

            # 제목에서 안전한 파일명 생성
            title = result['meeting_info']['title']
            safe_title = title.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')

            # 결과 저장 (test_{title}.json)
            output_path = Path("test_results") / f"test_{safe_title}.json"
            output_path.parent.mkdir(exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"💾 결과 저장: {output_path}")
            print()

            # 통계
            speakers = set(chunk['speaker'] for chunk in result['chunks'])
            agendas = set(chunk['agenda'] for chunk in result['chunks'])

            print("📊 통계")
            print(f"  - 총 발언 수: {len(result['chunks'])}개")
            print(f"  - 발언자: {len(speakers)}명")
            print(f"  - 안건: {len(agendas)}개")
            print()

            success_count += 1

        except Exception as e:
            logger.error(f"파싱 실패: {txt_path} - {e}")
            print(f"❌ 실패: {e}")
            print()
            fail_count += 1

    # 최종 결과
    print("=" * 100)
    print("📊 최종 결과")
    print("=" * 100)
    print(f"✅ 성공: {success_count}개")
    print(f"❌ 실패: {fail_count}개")
    print()


if __name__ == "__main__":
    main()
