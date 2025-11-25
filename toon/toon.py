import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import io

# --- 설정 영역 ---
# 실행 전 시스템 환경변수에 'GOOGLE_API_KEY'를 설정하거나, 아래에 직접 입력하세요.
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 요청하신 모델명 설정
MODEL_NAME = "gemini-3-pro-image-preview"

# 해치 캐릭터 및 만화 스타일 일관성을 위한 가이드 (모든 이미지 프롬프트에 추가됨)
HAECHI_STYLE_GUIDE = """
A four-panel comic strip based on the character 'Haechi' from Seoul City branding.
Character consistency is crucial: Haechi is a cute, yellow mythical creature with curly patterns on its head and body, small white feathered wings, big friendly eyes, and cute small fangs.
The art style is clean, cheerful, friendly cartoon, consistent across all panels.
"""

# ----------------

class HaechiComicGenerator:
    def __init__(self, api_key, model_name):
        if not api_key:
            raise ValueError("API Key가 설정되지 않았습니다.")
        genai.configure(api_key=api_key)
        # 설정된 모델의 기능(텍스트, 이미지 등)을 확인하고 가져옵니다.
        try:
             self.model = genai.GenerativeModel(model_name)
             print(f"Model '{model_name}' initialized successfully.")
        except Exception as e:
             print(f"Error initializing model '{model_name}'. Please check access rights.")
             raise e

    def _get_storyboard_from_text(self, topic, full_text):
        """
        1단계: 기획자 AI (LLM) - 입력된 텍스트를 분석하여 4컷 만화 스토리보드(JSON) 생성
        """
        print("Creating storyboard from text...")

        # 기획자 AI를 위한 시스템 프롬프트
        storyteller_prompt = f"""
        You are a professional comic scriptwriter for Seoul City's mascot, 'Haechi'.
        Your goal is to take complex official texts regarding Seoul City Council and simplify them into a fun, easy-to-understand 4-panel comic strip for citizens, starring Haechi.

        Haechi's persona: Bright, friendly, helpful, confident, and loves Seoul.

        Input Topic: {topic}
        Input Full Text: """ + full_text + """

        Task:
        1. Analyze the input text to understand the core message, problem, solution, and expected effect.
        2. Create a 4-panel narrative structure based on 'Ki-Seung-Jeon-Gyeol' (Introduction-Development-Turn-Conclusion).
        3. Panel 1 (Intro): Present the initial problem or citizen's confusion. Haechi appears.
        4. Panel 2 (Development): Haechi explains the core concept simply using visual metaphors.
        5. Panel 3 (Turn): Highlight the positive impact or necessity of the measure.
        6. Panel 4 (Conclusion): A happy resolution, Haechi gives a confident closing remark regarding Seoul's safety/development.

        Output Format:
        You MUST return strictly a valid JSON object containing a list of 4 panel objects. No introductory or concluding text.
        JSON Structure:
        {
            "comic_panels": [
                {
                    "panel_number": 1,
                    "dialogue": "Dialogue for characters in Korean. Keep it concise.",
                    "image_description_en": "A highly detailed visual description of the scene in English for an image generator AI. Include character actions, background details, and specific visual metaphors."
                },
                # ... repeat for panels 2, 3, 4
            ]
        }
        """

        response = self.model.generate_content(storyteller_prompt)
        
        try:
            # 마크다운 코드 블록 제거 및 JSON 파싱
            cleaned_text = response.text.strip().replace('```json', '').replace('```', '')
            storyboard_json = json.loads(cleaned_text)
            return storyboard_json['comic_panels']
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {response.text}")
            raise e

    def _generate_image_for_panel(self, panel_data):
        """
        2단계: 화가 AI (Image Gen) - 스토리보드의 묘사를 바탕으로 실제 이미지 생성
        (수정됨: 이미지 추출 로직 개선 및 안전한 에러 처리)
        """
        panel_num = panel_data['panel_number']
        print(f"Generating image for Panel {panel_num}...")

        # 일관성 유지를 위한 스타일 가이드와 개별 컷 묘사 결합
        full_image_prompt = f"""
        {HAECHI_STYLE_GUIDE}
        Scene description for this panel: {panel_data['image_description_en']}
        The image must include speech bubbles with the following Korean text: "{panel_data['dialogue']}"
        Ensure the composition looks like a single panel from a comic strip.
        """

        try:
            # 이미지 생성 요청
            response = self.model.generate_content(full_image_prompt)
            
            # --- [수정된 부분] 이미지 데이터 안전하게 추출하기 ---
            img_data = None
            # 응답의 각 부분을 순회하며 이미지 데이터가 있는지 확인
            if response.parts:
                for part in response.parts:
                    # 1. 일반적인 'image' 속성 확인
                    if hasattr(part, 'image'):
                        img_data = part.image
                        break
                    # 2. 'inline_data' 형태로 들어오는 경우 확인 (최신 모델 대비)
                    elif hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                         img_data = part.inline_data.data
                         break
            
            if img_data:
                 # 이미지 데이터가 성공적으로 추출된 경우
                 img = Image.open(io.BytesIO(img_data))
                 return img
            else:
                 # 이미지가 없거나 추출에 실패한 경우
                 print(f"Warning: Image generation requested for panel {panel_num}, but no valid image data extracted.")
                 # 안전을 위해 response.text를 직접 출력하지 않고 넘어갑니다.
                 # 필요시 디버깅을 위해 아래 주석을 해제하여 raw parts를 확인할 수 있습니다.
                 # print(f"Debug - Response parts info: {response.parts}")
                 return None
            # ----------------------------------------------------

        except Exception as e:
            # response.text 접근으로 인한 오류 등 모든 예외를 잡아냅니다.
            print(f"Error generating image for Panel {panel_num}: {e}")
            return None

    def create_comic(self, topic, full_text):
        """
        전체 프로세스 실행 함수
        """
        # 1. 텍스트 분석 및 스토리보드 생성
        storyboard = self._get_storyboard_from_text(topic, full_text)
        print("Storyboard created successfully.")

        generated_images = []
        # 2. 각 패널별로 순차적으로 이미지 생성
        for panel_data in storyboard:
            img = self._generate_image_for_panel(panel_data)
            if img:
                generated_images.append(img)
            else:
                print("Failed to generate an image, stopping process.")
                break
        
        return generated_images

# =========================================
# 실행 테스트 (우리가 사용했던 예시 데이터)
# =========================================
if __name__ == "__main__":
    # 테스트용 안건 제목과 내용 (이전 대화 내용 활용)
    test_topic = "서울특별시 주취자 보호 지원에 관한 조례 일부개정조례안"
    test_full_text = """
    (회의록 요약 가정)
    이번 조례안은 주취자 보호에 있어 관계 기관 간의 역할을 명확히 하기 위함입니다. 
    현재는 단순 주취자도 119 구급대가 이송하는 경우가 많아, 실제 응급환자 이송에 공백이 발생하고 있습니다.
    개정안은 응급 상황이 아닌 단순 주취자의 경우 경찰 등 다른 관계 기관이 보호 조치를 하도록 역할을 분담하여,
    119 구급대는 긴급한 생명을 구하는 본연의 임무에 집중할 수 있도록 하려는 것입니다.
    이를 통해 시민의 안전 공백을 최소화할 것으로 기대합니다.
    """

    try:
        # 생성기 초기화
        generator = HaechiComicGenerator(api_key=GOOGLE_API_KEY, model_name=MODEL_NAME)
        
        print(f"--- '{test_topic}' 만화 생성 시작 ---")
        comic_images = generator.create_comic(test_topic, test_full_text)

        if len(comic_images) == 4:
            print("\n--- 만화 생성 완료! ---")
            # 결과 이미지 저장 또는 표시
            for i, img in enumerate(comic_images):
                file_name = f"haechi_comic_panel_{i+1}.png"
                img.save(file_name)
                print(f"Saved: {file_name}")
            
            # (선택사항) 결과 이미지를 바로 화면에 띄우기
            # comic_images[0].show() 
        else:
            print("\n--- 만화 생성 실패 (일부 이미지 누락) ---")

    except Exception as e:
        print(f"\nAn error occurred: {e}")