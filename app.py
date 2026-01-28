import streamlit as st
import io
import json
from google.oauth2 import service_account
from google.cloud import speech
import google.generativeai as genai
from googleapiclient.discovery import build
from datetime import datetime

# ==========================================
# ⚙️ 설정 (클라우드 보안 금고 + 오류 방지 처리)
# ==========================================
try:
    # 1. Secrets에서 정보 가져오기
    gcp_info = dict(st.secrets["gcp_service_account"])
    
    # 🚨 중요: private_key의 줄바꿈 문자(\n)가 깨지는 문제 자동 해결
    if "private_key" in gcp_info:
        gcp_info["private_key"] = gcp_info["private_key"].replace("\\n", "\n")

    GOOGLE_API_KEY = st.secrets["general"]["GOOGLE_API_KEY"]
    SHARED_DRIVE_ID = st.secrets["general"]["SHARED_DRIVE_ID"]
    AI_MODEL_NAME = 'gemini-2.0-flash'

except Exception as e:
    st.error(f"🚨 보안 설정(Secrets) 로드 실패: {e}")
    st.stop()

# ==========================================
# 🛠️ 기능 함수들
# ==========================================
def step1_transcribe(uploaded_file):
    # 파일을 메모리에 읽음
    content = uploaded_file.read()
    
    creds = service_account.Credentials.from_service_account_info(gcp_info)
    client = speech.SpeechClient(credentials=creds)

    audio = speech.RecognitionAudio(content=content)
    
    # ★ 수정됨: encoding과 sample_rate_hertz를 지워서 구글이 '자동 감지'하게 만듦
    config = speech.RecognitionConfig(
        language_code="ko-KR",
        enable_automatic_punctuation=True,
        # mp3나 스마트폰 녹음 파일 호환성을 위해 자동 감지 모드로 변경
        diarization_config=speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=2,
            max_speaker_count=5,
        ),
    )
    
    # 10MB 이상의 파일은 에러가 날 수 있으므로 예외 처리 추가
    try:
        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=600)
    except Exception as e:
        st.error(f"❌ 녹음 파일 분석 중 오류 발생: {e}")
        st.warning("💡 힌트: 파일 용량이 10MB(약 10분)보다 크면 구글 정책상 처리가 안 될 수 있습니다. 짧은 파일로 테스트해보세요.")
        st.stop()

    transcript_text = ""
    # 결과가 비어있는 경우(말이 없는 경우) 처리
    if not response.results:
        return "대화 내용이 감지되지 않았습니다."

    result = response.results[-1]
    
    # alternatives가 있는지 확인
    if not result.alternatives:
        return "대화 내용을 텍스트로 변환할 수 없습니다."
        
    words_info = result.alternatives[0].words

    current_speaker = None
    sentence_buffer = []

    for word_info in words_info:
        speaker_tag = word_info.speaker_tag
        if current_speaker != speaker_tag:
            if current_speaker is not None:
                line = f"[화자 {current_speaker}]: {' '.join(sentence_buffer)}"
                transcript_text += line + "\n"
            current_speaker = speaker_tag
            sentence_buffer = []
        sentence_buffer.append(word_info.word)
    
    if sentence_buffer:
        line = f"[화자 {current_speaker}]: {' '.join(sentence_buffer)}"
        transcript_text += line + "\n"

    return transcript_text

def step2_summarize(transcript):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(AI_MODEL_NAME)
    prompt = f"""
    당신은 KCH Global의 유능한 회의록 서기입니다.
    아래 녹취록을 바탕으로 보고서 형식으로 요약해주세요.
    
    [녹취록]
    {transcript}
    
    [작성 양식]
    # 📅 회의 요약 보고서
    ## 1. 핵심 안건
    ## 2. 주요 논의 사항
    ## 3. 결정 사항
    ## 4. 향후 계획 (담당자 지정)
    """
    response = model.generate_content(prompt)
    return response.text

def step3_save(summary, transcript):
    SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/documents']
    creds = service_account.Credentials.from_service_account_info(gcp_info, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=creds)
    docs_service = build('docs', 'v1', credentials=creds)

    today = datetime.now().strftime("%Y-%m-%d %H시%M분")
    file_name = f"[AI회의록] {today} 회의 결과"

    file_metadata = {
        'name': file_name,
        'mimeType': 'application/vnd.google-apps.document',
        'parents': [SHARED_DRIVE_ID]
    }
    file = drive_service.files().create(body=file_metadata, fields='id', supportsAllDrives=True).execute()
    doc_id = file.get('id')

    full_content = summary + "\n\n" + "-"*30 + "\n[참고: 대화 원본]\n" + transcript
    requests = [{'insertText': {'location': {'index': 1}, 'text': full_content}}]
    docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': requests}).execute()
    return file_name

# ==========================================
# 🖥️ 화면 구성
# ==========================================
st.set_page_config(page_title="KCH Global AI 회의록", page_icon="🎙️")
st.title("🎙️ KCH Global AI 회의록 생성기")
st.markdown("언제 어디서나 녹음 파일만 올리세요. **AI가 자동으로 처리합니다.**")

# 파일 업로더 (용량 제한 안내 추가)
uploaded_file = st.file_uploader("녹음 파일 업로드 (MP3, WAV 권장)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    if st.button("🚀 회의록 만들기 시작"):
        with st.status("클라우드 서버에서 작업 중... (3~5분 소요)", expanded=True) as status:
            
            st.write("🎧 1단계: 받아쓰기 중...")
            transcript = step1_transcribe(uploaded_file)
            st.write("✅ 받아쓰기 완료!")
            
            st.write("🧠 2단계: 요약 중...")
            summary = step2_summarize(transcript)
            st.write("✅ 요약 완료!")
            
            st.write("💾 3단계: 저장 중...")
            file_name = step3_save(summary, transcript)
            
            status.update(label="🎉 완료!", state="complete", expanded=False)

        st.success(f"'{file_name}' 저장 완료!")
        st.subheader("미리보기")
        st.markdown(summary)
