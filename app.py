import streamlit as st
import io
import os
from google.oauth2 import service_account
from google.cloud import speech
from google.cloud import storage
import google.generativeai as genai
from googleapiclient.discovery import build
from datetime import datetime
import uuid
from pydub import AudioSegment

# ==========================================
# ⚙️ 설정
# ==========================================
try:
    gcp_info = dict(st.secrets["gcp_service_account"])
    if "private_key" in gcp_info:
        gcp_info["private_key"] = gcp_info["private_key"].replace("\\n", "\n")

    GOOGLE_API_KEY = st.secrets["general"]["GOOGLE_API_KEY"]
    SHARED_DRIVE_ID = st.secrets["general"]["SHARED_DRIVE_ID"]
    BUCKET_NAME = st.secrets["general"]["BUCKET_NAME"]
    AI_MODEL_NAME = 'gemini-2.0-flash'

except Exception as e:
    st.error(f"🚨 설정 로드 실패: {e}")
    st.stop()

# ==========================================
# 🛠️ 기능 함수들
# ==========================================

# 0. 오디오 포맷 변환 (무엇이든 WAV로!)
def convert_to_wav(uploaded_file):
    # 파일 확장자 확인
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    # Pydub로 오디오 읽기
    audio = AudioSegment.from_file(uploaded_file, format=file_ext)
    
    # WAV로 변환 (모노, 16000Hz - 구글 STT 최적화)
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    # 메모리 버퍼에 저장
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0) # 버퍼 포인터 초기화
    
    return buffer

# 1. 파일을 클라우드 창고(Bucket)로 올리는 함수
def upload_to_bucket(blob_name, file_obj):
    creds = service_account.Credentials.from_service_account_info(gcp_info)
    storage_client = storage.Client(credentials=creds, project=gcp_info["project_id"])
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(file_obj, content_type="audio/wav")
    return f"gs://{BUCKET_NAME}/{blob_name}"

# 2. 창고에 있는 파일을 받아쓰기 하는 함수
def step1_transcribe_gcs(gcs_uri):
    creds = service_account.Credentials.from_service_account_info(gcp_info)
    client = speech.SpeechClient(credentials=creds)

    audio = speech.RecognitionAudio(uri=gcs_uri)
    
    # WAV(Linear16)에 16000Hz로 맞춤 설정 (오류 원천 차단)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR",
        enable_automatic_punctuation=True,
        diarization_config=speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=2,
            max_speaker_count=5,
        ),
    )
    
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=1800)

    transcript_text = ""
    if not response.results:
        return "대화 내용이 감지되지 않았습니다."

    result = response.results[-1]
    
    if not result.alternatives:
        return "분석 결과 없음"

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
st.title("🎙️ KCH Global AI 회의록 생성기 (Enterprise)")
st.markdown("아이폰(m4a), 갤럭시(m4a), 녹음기(mp3) 등 **모든 파일을 지원합니다.**")

# 팁: Streamlit Cloud에서 ffmpeg 설치되기를 기다려야 함
if "ffmpeg_checked" not in st.session_state:
    st.session_state.ffmpeg_checked = True

uploaded_file = st.file_uploader("녹음 파일 업로드", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    if st.button("🚀 대용량 회의록 만들기 시작"):
        with st.status("AI 시스템 가동 중...", expanded=True) as status:
            
            # 0. 변환
            st.write("🔄 1단계: 오디오 파일을 최적화(WAV) 변환 중...")
            try:
                wav_buffer = convert_to_wav(uploaded_file)
                st.write("✅ 변환 완료!")
            except Exception as e:
                st.error(f"변환 실패 (ffmpeg가 아직 설치 중일 수 있습니다. 1분 뒤 다시 시도하세요): {e}")
                st.stop()

            # 1. 업로드
            st.write("☁️ 2단계: 클라우드 창고로 전송 중...")
            # 확장자를 .wav로 변경해서 저장
            unique_filename = f"{uuid.uuid4()}.wav"
            gcs_uri = upload_to_bucket(unique_filename, wav_buffer)
            st.write(f"✅ 전송 완료! ({gcs_uri})")

            # 2. 받아쓰기
            st.write("🎧 3단계: AI가 내용을 듣고 받아쓰는 중... (시간이 좀 걸립니다)")
            try:
                transcript = step1_transcribe_gcs(gcs_uri)
                if transcript.startswith("대화 내용이") or transcript.startswith("분석 결과"):
                     st.warning("⚠️ 대화 내용이 명확하게 들리지 않거나 너무 짧습니다.")
                     st.stop()
                st.write("✅ 받아쓰기 완료!")
            except Exception as e:
                st.error(f"받아쓰기 실패: {e}")
                st.stop()
            
            # 3. 요약
            st.write("🧠 4단계: 핵심 내용 요약 중...")
            summary = step2_summarize(transcript)
            st.write("✅ 요약 완료!")
            
            # 4. 저장
            st.write("💾 5단계: 드라이브 저장 중...")
            file_name = step3_save(summary, transcript)
            
            status.update(label="🎉 작업 완료!", state="complete", expanded=False)

        st.success(f"'{file_name}' 저장 완료!")
        st.subheader("📝 요약 미리보기")
        st.markdown(summary)
