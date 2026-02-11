import base64
import difflib
import html
import io
import json
import os
import smtplib
import subprocess
import tempfile
import uuid
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.request import urlopen

import google.generativeai as genai
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from google.cloud import speech
from google.cloud import storage
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload


LOGO_URL = "https://i.postimg.cc/ZJVYW4Mj/KCH-LOGOV3.png"
ROSTER_FILE = "명부.xlsx"
ROSTER_SHEET = "Users"
REQ_COLS = ["Name", "Email"]
OPT_COLS = ["Dept", "Title", "Team", "Role", "Lang", "Timezone", "IsCCDefault", "ManagerEmail"]
PROMPTS = {
    "memo": "(a) 메모 -> 회의록",
    "transcript": "(b) 녹취 -> 1p 요약",
    "agenda": "(c) 60분 안건 생성",
    "invite": "(d) 초대메일 설명 문구",
    "followup": "(e) Follow-up 이메일(개인별)",
}


def b(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def uniq(items):
    out, seen = [], set()
    for x in items:
        x = str(x).strip()
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def parse_int(v, default):
    try:
        return int(str(v).strip())
    except Exception:
        return default


def load_settings():
    try:
        gcp = dict(st.secrets["gcp_service_account"])
        if "private_key" in gcp:
            gcp["private_key"] = gcp["private_key"].replace("\\n", "\n")
        general = dict(st.secrets.get("general", {}))
    except Exception as e:
        st.error(f"설정 로드 실패: {e}")
        st.stop()

    for k in ["GOOGLE_API_KEY", "SHARED_DRIVE_ID", "BUCKET_NAME"]:
        if not str(general.get(k, "")).strip():
            st.error(f"필수 secrets 누락: {k}")
            st.stop()

    return {
        "gcp": gcp,
        "api_key": str(general.get("GOOGLE_API_KEY", "")).strip(),
        "bucket": str(general.get("BUCKET_NAME", "")).strip(),
        "shared_drive": str(general.get("SHARED_DRIVE_ID", "")).strip(),
        "model": str(general.get("AI_MODEL_NAME", "gemini-2.0-flash")).strip() or "gemini-2.0-flash",
        "gmail_impersonate": str(general.get("GMAIL_IMPERSONATE_USER", "")).strip(),
        "gmail_name": str(general.get("GMAIL_FROM_NAME", "KCH Global")).strip() or "KCH Global",
        "smtp_host": str(general.get("SMTP_HOST", "")).strip(),
        "smtp_port": parse_int(general.get("SMTP_PORT", 587), 587),
        "smtp_user": str(general.get("SMTP_USER", "")).strip(),
        "smtp_pw": str(general.get("SMTP_PASSWORD", "")),
        "smtp_from": str(general.get("SMTP_FROM_EMAIL", "")).strip(),
        "smtp_name": str(general.get("SMTP_FROM_NAME", "KCH Global")).strip() or "KCH Global",
        "smtp_ssl": b(general.get("SMTP_USE_SSL", False)),
        "smtp_tls": b(general.get("SMTP_STARTTLS", True)),
        "roster_folder": str(general.get("ROSTER_DRIVE_FOLDER_ID", "")).strip(),
    }


def creds(info, scopes, subject=None):
    c = service_account.Credentials.from_service_account_info(info, scopes=scopes)
    return c.with_subject(subject) if subject else c


def audio_ext(name="", mime=""):
    if name and "." in name:
        return name.rsplit(".", 1)[-1].lower()
    if mime and "/" in mime:
        x = mime.split("/")[-1].lower()
        return "wav" if x in {"x-wav", "wav"} else x
    return "wav"


def to_wav(raw, ext):
    ext = (ext or "wav").lower()
    if ext == "x-wav":
        ext = "wav"
    if not ext.isalnum():
        ext = "wav"

    with tempfile.TemporaryDirectory(prefix="kch_audio_") as td:
        in_path = os.path.join(td, f"in.{ext}")
        out_path = os.path.join(td, "out.wav")
        with open(in_path, "wb") as f:
            f.write(raw)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            in_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            out_path,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError as e:
            raise RuntimeError("오디오 변환 실패(ffmpeg 미설치 또는 PATH 미설정)") from e

        if proc.returncode != 0 or not os.path.exists(out_path):
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"오디오 변환 실패(ffmpeg): {detail[:500]}")

        with open(out_path, "rb") as f:
            data = f.read()

    buf = io.BytesIO(data)
    buf.seek(0)
    return buf


def upload_wav(wav_buf, blob_name, cfg):
    c = creds(cfg["gcp"], ["https://www.googleapis.com/auth/cloud-platform"])
    cli = storage.Client(credentials=c, project=cfg["gcp"]["project_id"])
    blob = cli.bucket(cfg["bucket"]).blob(blob_name)
    blob.upload_from_file(wav_buf, content_type="audio/wav")
    return f"gs://{cfg['bucket']}/{blob_name}"


def transcribe(uri, cfg):
    c = creds(cfg["gcp"], ["https://www.googleapis.com/auth/cloud-platform"])
    client = speech.SpeechClient(credentials=c)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR",
        enable_automatic_punctuation=True,
        diarization_config=speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=2,
            max_speaker_count=8,
        ),
    )
    op = client.long_running_recognize(config=config, audio=speech.RecognitionAudio(uri=uri))
    res = op.result(timeout=1800)
    if not res.results:
        return ""
    alt = res.results[-1].alternatives
    if not alt:
        return ""
    words = alt[0].words
    if not words:
        return alt[0].transcript.strip()

    lines, cur_sp, cur_words = [], None, []
    for w in words:
        sp = getattr(w, "speaker_tag", 0)
        if sp != cur_sp:
            if cur_sp is not None and cur_words:
                lines.append(f"[화자 {cur_sp}]: {' '.join(cur_words)}")
            cur_sp, cur_words = sp, []
        cur_words.append(w.word)
    if cur_words:
        lines.append(f"[화자 {cur_sp}]: {' '.join(cur_words)}")
    return "\n".join(lines).strip()


def common_meta(meta):
    return f"""[공통 지시]
- 너는 KCH Global의 "회의 운영/회의록" 담당자다.
- 사실만 기반으로 작성하고, 메모/녹취에 없는 내용은 만들지 말 것.
- 불명확하면 (확인 필요)/(결정 보류)/(추가 데이터 필요)로 표기.
- 출력 형식은 지정된 섹션/표를 반드시 따른다.

[회의 메타]
- 회의명: {meta.get('title', '')}
- 일시: {meta.get('datetime', '')}
- 장소/채널: {meta.get('location', '')}
- 진행자: {meta.get('host', '')}
- 서기: {meta.get('note_taker', '')}
- 참석자: {meta.get('participants', '')}
- 참조 링크/자료: {meta.get('refs', '')}
- 보안등급: {meta.get('security', '')}
"""


def build_prompt(kind, meta, p):
    cm = common_meta(meta)
    if kind == "memo":
        return f"""{cm}
[작업]
아래 메모를 임원 공유 가능한 회의록으로 정리해라.

[출력 형식 — 반드시 준수]
# 회의록
## 1) 회의 개요
- 목적:
- 배경:
- 참석자:
- 회의 범위(오늘 다룬 것 / 다루지 않은 것):
## 2) 주요 논의 내용
- 안건 1: {{안건명}}
  - 현황/문제 정의:
  - 핵심 논점(찬반/대안 비교 포함):
  - 근거(메모 데이터/사실):
  - 리스크/우려:
  - 미결 질문(확인 필요):
## 3) 결정 사항 (Decision Log)
- [결정] D1. ___
- [보류] H1. ___
## 4) 액션 아이템 (Action Items)
| No | To-do | 담당자 | 마감일 | 우선순위(H/M/L) | 상태(신규/진행/보류) | 비고 |
|---|------|------|------|----------------|---------------------|-----|

[메모]
<메모>: {p.get('memo_text', '')}
"""
    if kind == "transcript":
        return f"""{cm}
[작업]
아래 녹취(전사)를 1페이지 요약 회의록으로 작성하라.

[출력 형식 — 반드시 준수]
# 1p 요약 회의록
## 핵심 결론 (3~6줄)
- …
## 합의된 내용 / 결정 사항 (최대 7개)
- D1. …
## 핵심 논의 요약 (안건별 2~4줄)
- 안건1: …
## 리스크 / 쟁점 / 확인 필요
- 리스크:
- 쟁점:
- 확인 필요:
## 액션 아이템 (Top 5)
| To-do | 담당자 | 마감일 | 비고 |
|------|------|------|-----|

[녹취]
<녹음본>: {p.get('transcript_text', '')}
"""
    if kind == "agenda":
        return f"""{cm}
[작업]
아래 회의 목적/배경으로 60분 안건과 진행 순서를 제안하라.

[출력 형식 — 반드시 준수]
# 60분 회의 안건(Agenda)
## 회의 목표 (1문장)
- …
## 타임테이블
| 순서 | 안건 | 목적 | 예상시간 | 진행 방식(설명/토론/결정) | 산출물 | Decision Point |
|-----|------|------|---------|--------------------------|--------|----------------|
| 1 | … | … | 5m | … | … | Y/N |
## 사전 준비(Pre-read)
- …
## 회의 진행 룰(권장)
- 시간 초과 시 컷오프 기준:
- 의사결정 기준:
- 주차(Parking lot) 규칙:

[회의 목적/배경]
<회의 목적/배경>: {p.get('purpose', '')}
"""
    if kind == "invite":
        return f"""{cm}
[작업]
아래 회의 정보를 바탕으로 캘린더 초대 설명 문구를 작성하라.

[출력 형식 — 반드시 준수]
[회의 목적]
- …
[주요 안건]
- 1) …
[참여자]
- …
[소요 시간]
- …
[회의 장소/접속]
- …
[사전 준비/자료]
- …
[회의에서 결정할 것]
- …

<회의 정보>: {p.get('meeting_info', '')}
"""
    if kind == "followup":
        return f"""{cm}
[작업]
아래 회의 요약으로 개인별 Follow-up 이메일을 작성하라.

[출력 형식 — 반드시 준수]
Subject: {p.get('subject', '')}

안녕하세요, {p.get('recipient_name', '')}님.
1) 감사합니다
- …
2) 오늘 합의/결정된 내용 요약
- …
3) {p.get('recipient_name', '')}님의 할 일 (우선순위 순)
- [ ] … (마감: …)
4) 전체 액션아이템(참고)
- …
5) 다음 일정
- 다음 회의: …
- 필요 시: …
6) 참고 링크/회의록
- 회의록(Google Doc): {p.get('doc_url', '')}
- 기타: {p.get('refs', '')}

감사합니다.
{p.get('signature', '')}

<회의 요약>: {p.get('summary', '')}
"""
    raise ValueError(f"지원하지 않는 템플릿: {kind}")


def run_gemini(prompt, cfg):
    genai.configure(api_key=cfg["api_key"])
    model = genai.GenerativeModel(cfg["model"])
    res = model.generate_content(prompt)
    text = (getattr(res, "text", "") or "").strip()
    if not text:
        raise RuntimeError("Gemini 응답이 비어 있습니다.")
    return text


def save_doc(text, title, cfg):
    c = creds(
        cfg["gcp"],
        ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/documents"],
    )
    drive = build("drive", "v3", credentials=c, cache_discovery=False)
    docs = build("docs", "v1", credentials=c, cache_discovery=False)
    meta = {
        "name": title,
        "mimeType": "application/vnd.google-apps.document",
        "parents": [cfg["shared_drive"]],
    }
    doc = drive.files().create(body=meta, fields="id", supportsAllDrives=True).execute()
    doc_id = doc["id"]
    docs.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": [{"insertText": {"location": {"index": 1}, "text": text}}]},
    ).execute()
    return f"https://docs.google.com/document/d/{doc_id}/edit"


def roster_empty():
    df = pd.DataFrame(columns=REQ_COLS + OPT_COLS)
    df["IsCCDefault"] = False
    return df


def ext_empty():
    return pd.DataFrame(columns=["Name", "Email"])


def roster_norm(df):
    if df is None:
        return roster_empty()
    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]
    m = {str(c).strip().lower(): c for c in x.columns}
    ren = {}
    for col in REQ_COLS + OPT_COLS:
        src = m.get(col.lower())
        if src is not None:
            ren[src] = col
    x = x.rename(columns=ren)
    miss = [c for c in REQ_COLS if c not in x.columns]
    if miss:
        raise ValueError(f"필수 컬럼 누락: {', '.join(miss)}")
    for c in OPT_COLS:
        if c not in x.columns:
            x[c] = False if c == "IsCCDefault" else ""
    x = x[REQ_COLS + OPT_COLS]
    for c in x.columns:
        if c == "IsCCDefault":
            x[c] = x[c].apply(b)
        else:
            x[c] = x[c].fillna("").astype(str).str.strip()
    x = x[~((x["Name"] == "") & (x["Email"] == ""))].reset_index(drop=True)
    return x


def ext_norm(df):
    if df is None:
        return ext_empty()
    x = df.copy()
    for c in ["Name", "Email"]:
        if c not in x.columns:
            x[c] = ""
        x[c] = x[c].fillna("").astype(str).str.strip()
    x = x[["Name", "Email"]]
    x = x[~((x["Name"] == "") & (x["Email"] == ""))].reset_index(drop=True)
    return x


def roster_load_bytes(raw):
    try:
        df = pd.read_excel(io.BytesIO(raw), sheet_name=ROSTER_SHEET)
    except ValueError as e:
        raise ValueError("Users 시트를 찾을 수 없습니다.") from e
    return roster_norm(df)


def roster_load_default():
    if not os.path.exists(ROSTER_FILE):
        return roster_empty()
    try:
        with open(ROSTER_FILE, "rb") as f:
            return roster_load_bytes(f.read())
    except Exception as e:
        st.warning(f"기본 명부 로드 실패: {e}")
        return roster_empty()


def roster_to_xlsx(df):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name=ROSTER_SHEET)
    out.seek(0)
    return out.getvalue()


def parse_email_csv(text):
    if not text:
        return []
    return uniq(text.replace(";", ",").split(","))


def build_recipients(roster_df, names, ext_df):
    rec = []
    sel = set(names or [])
    rows = roster_df[roster_df["Name"].isin(sel)] if sel else roster_df.iloc[0:0]
    for _, r in rows.iterrows():
        email = str(r.get("Email", "")).strip()
        if not email:
            continue
        rec.append(
            {
                "name": str(r.get("Name", "")).strip() or email,
                "email": email,
                "team": str(r.get("Team", "")).strip(),
                "title": str(r.get("Title", "")).strip(),
                "manager": str(r.get("ManagerEmail", "")).strip(),
                "cc_default": b(r.get("IsCCDefault", False)),
            }
        )
    ext_rows = ext_norm(ext_df)
    for _, r in ext_rows.iterrows():
        email = str(r.get("Email", "")).strip()
        if not email:
            continue
        rec.append({"name": str(r.get("Name", "")).strip() or email, "email": email, "team": "", "title": "", "manager": "", "cc_default": False})

    out, seen = [], set()
    for r in rec:
        k = r["email"].lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def participants_text(recipients):
    items = []
    for r in recipients:
        extra = ", ".join([x for x in [r.get("team", ""), r.get("title", "")] if x])
        items.append(f"{r['name']} ({extra + ', ' if extra else ''}{r['email']})")
    return ", ".join(items)


def cc_for(recipient, manual_cc):
    cc = list(manual_cc)
    if recipient.get("cc_default") and recipient.get("manager"):
        cc.append(recipient["manager"])
    cc = uniq(cc)
    to = recipient.get("email", "").lower()
    return [x for x in cc if x.lower() != to]


def parse_subject_body(text, fallback):
    lines = text.strip().splitlines()
    if lines and lines[0].lower().startswith("subject:"):
        sub = lines[0].split(":", 1)[1].strip() or fallback
        return sub, "\n".join(lines[1:]).strip()
    return fallback, text.strip()


def personal_actions(summary, name):
    out = [ln.strip() for ln in (summary or "").splitlines() if name in ln]
    return out[:5] if out else ["(확인 필요) 개인 액션아이템 지정 필요"]


def followup_fallback(name, title, summary, doc_url, refs, sign):
    acts = "\n".join([f"- [ ] {a}" for a in personal_actions(summary, name)])
    return (
        f"안녕하세요, {name}님.\n\n1) 감사합니다\n- 회의 참석 감사합니다.\n\n"
        f"2) 오늘 합의/결정된 내용 요약\n- {title or '(확인 필요)'}\n\n"
        f"3) {name}님의 할 일 (우선순위 순)\n{acts}\n\n"
        "4) 전체 액션아이템(참고)\n- (확인 필요)\n\n"
        "5) 다음 일정\n- 다음 회의: (확인 필요)\n- 필요 시: 개별 안내\n\n"
        f"6) 참고 링크/회의록\n- 회의록(Google Doc): {doc_url or '(확인 필요)'}\n- 기타: {refs or '(확인 필요)'}\n\n"
        f"감사합니다.\n{sign}"
    )


@st.cache_data(show_spinner=False)
def logo_bytes():
    try:
        with urlopen(LOGO_URL, timeout=10) as r:
            return r.read()
    except Exception:
        return b""


def email_html(body, inline):
    txt = "<br>".join(html.escape(x) for x in body.splitlines())
    img = '<img src="cid:kch-logo" alt="KCH Logo" style="width:220px;max-width:100%;margin-top:16px;" />'
    if not inline:
        img = f'<img src="{LOGO_URL}" alt="KCH Logo" style="width:220px;max-width:100%;margin-top:16px;" />'
    return f'<div style="font-family:Arial,sans-serif;font-size:14px;line-height:1.6;">{txt}<br><br>{img}</div>'


def build_mail(sender_name, sender_email, to, cc, bcc, subject, body, logo):
    msg = MIMEMultipart("related")
    msg["From"] = f"{sender_name} <{sender_email}>"
    msg["To"] = to
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg["Subject"] = subject
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(body, "plain", "utf-8"))
    alt.attach(MIMEText(email_html(body, bool(logo)), "html", "utf-8"))
    msg.attach(alt)
    if logo:
        img = MIMEImage(logo, _subtype="png")
        img.add_header("Content-ID", "<kch-logo>")
        img.add_header("Content-Disposition", "inline", filename="KCH-LOGOV3.png")
        msg.attach(img)
    return msg, uniq([to] + cc + bcc)


def send_gmail(msg, cfg):
    user = cfg["gmail_impersonate"]
    if not user:
        raise ValueError("GMAIL_IMPERSONATE_USER 필요")
    c = creds(cfg["gcp"], ["https://www.googleapis.com/auth/gmail.send"], subject=user)
    gm = build("gmail", "v1", credentials=c, cache_discovery=False)
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    gm.users().messages().send(userId="me", body={"raw": raw}).execute()


def send_smtp(msg, rcpts, cfg):
    if not cfg["smtp_host"]:
        raise ValueError("SMTP_HOST 필요")
    if cfg["smtp_ssl"]:
        s = smtplib.SMTP_SSL(cfg["smtp_host"], cfg["smtp_port"], timeout=30)
    else:
        s = smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"], timeout=30)
    try:
        if not cfg["smtp_ssl"] and cfg["smtp_tls"]:
            s.starttls()
        if cfg["smtp_user"]:
            s.login(cfg["smtp_user"], cfg["smtp_pw"])
        s.sendmail(msg["From"], rcpts, msg.as_string())
    finally:
        s.quit()


def roster_save_drive(df, cfg):
    c = creds(cfg["gcp"], ["https://www.googleapis.com/auth/drive"])
    drive = build("drive", "v3", credentials=c, cache_discovery=False)
    folder = cfg["roster_folder"] or cfg["shared_drive"]
    q = f"name = '{ROSTER_FILE}' and trashed = false"
    if folder:
        q += f" and '{folder}' in parents"
    res = drive.files().list(
        q=q,
        fields="files(id,name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        pageSize=10,
    ).execute()
    files = res.get("files", [])
    media = MediaIoBaseUpload(
        io.BytesIO(roster_to_xlsx(df)),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        resumable=False,
    )
    if files:
        fid = files[0]["id"]
        drive.files().update(fileId=fid, media_body=media, supportsAllDrives=True).execute()
    else:
        body = {"name": ROSTER_FILE}
        if folder:
            body["parents"] = [folder]
        fid = drive.files().create(body=body, media_body=media, supportsAllDrives=True, fields="id").execute()["id"]
    return f"https://drive.google.com/file/d/{fid}/view"


def copy_btn(text, key, label):
    components.html(
        f"""
<button id="btn-{key}" style="background:#1f77b4;color:#fff;border:0;padding:6px 12px;border-radius:6px;cursor:pointer;">{label}</button>
<span id="msg-{key}" style="margin-left:8px;font-size:12px;color:#2c7;"></span>
<script>
const b=document.getElementById("btn-{key}");
const m=document.getElementById("msg-{key}");
b.onclick=async()=>{{try{{await navigator.clipboard.writeText({json.dumps(text)});m.textContent="복사됨";setTimeout(()=>m.textContent="",1200);}}catch(e){{m.textContent="복사 실패";}}}};
</script>
        """,
        height=42,
    )


def init_state():
    defaults = {
        "meta_title": "",
        "meta_dt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "meta_loc": "",
        "meta_host": "",
        "meta_note": "",
        "meta_participants": "",
        "meta_refs": "",
        "meta_sec": "사내공유",
        "selected_names": [],
        "last_transcript": "",
        "last_summary": "",
        "last_doc_url": "",
        "prompt_out": "",
        "prompt_kind": "",
        "email_previews": [],
        "sender_backend": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "roster_df" not in st.session_state:
        st.session_state["roster_df"] = roster_load_default()
    if "ext_df" not in st.session_state:
        st.session_state["ext_df"] = ext_empty()


def meta_from_state():
    return {
        "title": st.session_state.get("meta_title", ""),
        "datetime": st.session_state.get("meta_dt", ""),
        "location": st.session_state.get("meta_loc", ""),
        "host": st.session_state.get("meta_host", ""),
        "note_taker": st.session_state.get("meta_note", ""),
        "participants": st.session_state.get("meta_participants", ""),
        "refs": st.session_state.get("meta_refs", ""),
        "security": st.session_state.get("meta_sec", "사내공유"),
    }


def process_audio(raw_bytes, name, mime, meta, cfg):
    with st.status("오디오 처리 중...", expanded=True) as status:
        st.write("1) WAV 변환")
        wav = to_wav(raw_bytes, audio_ext(name, mime))

        st.write("2) GCS 업로드")
        uri = upload_wav(wav, f"{uuid.uuid4()}.wav", cfg)
        st.write(uri)

        st.write("3) STT (화자 분리)")
        transcript = transcribe(uri, cfg)
        if not transcript:
            raise RuntimeError("대화 내용이 감지되지 않았습니다.")

        st.write("4) Gemini 요약")
        s_prompt = build_prompt("transcript", meta, {"transcript_text": transcript})
        summary = run_gemini(s_prompt, cfg)

        st.write("5) Google Docs 저장")
        title = f"[AI회의록] {datetime.now().strftime('%Y-%m-%d %H시%M분')} {meta.get('title', '')}".strip()
        full = summary + "\n\n" + "-" * 30 + "\n[참고: 대화 원본]\n" + transcript
        doc_url = save_doc(full, title, cfg)
        status.update(label="완료", state="complete", expanded=False)

    return transcript, summary, doc_url


def info_text(meta):
    return (
        f"- 회의명: {meta.get('title', '')}\n"
        f"- 일시: {meta.get('datetime', '')}\n"
        f"- 장소/채널: {meta.get('location', '')}\n"
        f"- 진행자: {meta.get('host', '')}\n"
        f"- 참석자: {meta.get('participants', '')}\n"
        f"- 참조 링크/자료: {meta.get('refs', '')}"
    )


st.set_page_config(page_title="KCH Global AI 회의록", page_icon="🎙️", layout="wide")
cfg = load_settings()
init_state()

st.image(LOGO_URL, width=220)
st.title("🎙️ KCH Global AI 회의록 생성기 v2")
st.caption("오디오 업로드/브라우저 녹음, 프롬프트 라이브러리, 명부 자동완성, 개인별 이메일 발송")

with st.sidebar:
    st.subheader("회의 메타")
    st.text_input("회의명", key="meta_title")
    st.text_input("일시", key="meta_dt")
    st.text_input("장소/채널", key="meta_loc")
    st.text_input("진행자", key="meta_host")
    st.text_input("서기", key="meta_note")
    st.text_area("참석자(메타용)", key="meta_participants", height=90)
    st.text_area("참조 링크/자료", key="meta_refs", height=90)
    st.selectbox("보안등급", ["사내공유", "제한공유", "대외비"], key="meta_sec")

meta = meta_from_state()
t1, t2, t3 = st.tabs(["1) 오디오 회의록", "2) 프롬프트 라이브러리", "3) 명부/이메일"])

with t1:
    u_tab, r_tab = st.tabs(["파일 업로드", "브라우저 녹음"])

    with u_tab:
        up = st.file_uploader("녹음 파일 업로드", type=["mp3", "wav", "m4a"], key="audio_upload")
        if up is not None:
            st.audio(up.getvalue())
        if st.button("업로드 파일로 회의록 생성", key="btn_audio_upload", disabled=up is None):
            try:
                tr, sm, url = process_audio(up.getvalue(), up.name, getattr(up, "type", ""), meta, cfg)
                st.session_state["last_transcript"] = tr
                st.session_state["last_summary"] = sm
                st.session_state["last_doc_url"] = url
                st.success(f"완료: {url}")
            except Exception as e:
                st.error(f"처리 실패: {e}")

    with r_tab:
        if hasattr(st, "audio_input"):
            st.info("녹음 버튼을 누르면 브라우저에서 마이크 허용/차단 팝업이 표시됩니다. 허용한 경우에만 녹음이 가능합니다.")
            st.caption("권한을 차단했다면 주소창의 사이트 권한(자물쇠 아이콘)에서 마이크를 허용으로 바꾼 뒤 페이지를 새로고침하세요.")
            rec = st.audio_input("브라우저에서 녹음", key="audio_record")
            if rec is None:
                st.caption("녹음 파일이 아직 없습니다. 권한 허용 후 녹음을 완료하면 아래 처리 버튼이 활성화됩니다.")
            else:
                st.audio(rec.getvalue())
            if st.button("녹음본으로 회의록 생성", key="btn_audio_record", disabled=rec is None):
                try:
                    tr, sm, url = process_audio(
                        rec.getvalue(),
                        getattr(rec, "name", "browser_record.wav"),
                        getattr(rec, "type", "audio/wav"),
                        meta,
                        cfg,
                    )
                    st.session_state["last_transcript"] = tr
                    st.session_state["last_summary"] = sm
                    st.session_state["last_doc_url"] = url
                    st.success(f"완료: {url}")
                except Exception as e:
                    st.error(f"처리 실패: {e}")
        else:
            st.info("현재 Streamlit 버전은 st.audio_input을 지원하지 않습니다.")

    if st.session_state.get("last_summary"):
        st.markdown("### 최근 생성 결과")
        if st.session_state.get("last_doc_url"):
            st.markdown(f"[Google Docs 열기]({st.session_state['last_doc_url']})")
        st.markdown(st.session_state["last_summary"])
        copy_btn(st.session_state["last_summary"], "copy-last-summary", "요약 복사")
        with st.expander("녹취 보기"):
            st.text_area("녹취", value=st.session_state.get("last_transcript", ""), height=220)

with t2:
    kind = st.selectbox("템플릿 선택", options=list(PROMPTS.keys()), format_func=lambda x: PROMPTS[x], key="prompt_kind_ui")
    payload = {}
    if kind == "memo":
        payload["memo_text"] = st.text_area("메모 원문", height=220, key="pl_memo")
    elif kind == "transcript":
        payload["transcript_text"] = st.text_area("녹취 원문", value=st.session_state.get("last_transcript", ""), height=220, key="pl_transcript")
    elif kind == "agenda":
        payload["purpose"] = st.text_area("회의 목적/배경", height=220, key="pl_purpose")
    elif kind == "invite":
        payload["meeting_info"] = st.text_area("회의 정보", value=info_text(meta), height=220, key="pl_invite_info")
    elif kind == "followup":
        subject = f"[{meta.get('title') or '회의'}] 결과 및 Action Items ({datetime.now().strftime('%Y-%m-%d')})"
        payload["recipient_name"] = st.text_input("수신자 이름", key="pl_rec_name")
        payload["subject"] = st.text_input("제목", value=subject, key="pl_subject")
        payload["doc_url"] = st.text_input("회의록 URL", value=st.session_state.get("last_doc_url", ""), key="pl_doc_url")
        payload["refs"] = st.text_input("참고 링크", value=meta.get("refs", ""), key="pl_refs")
        payload["signature"] = st.text_input("서명", value="KCH Global AI 회의록", key="pl_sign")
        payload["summary"] = st.text_area("회의 요약", value=st.session_state.get("last_summary", ""), height=220, key="pl_summary")

    prompt_text = build_prompt(kind, meta, payload)
    st.markdown("### 조립된 프롬프트")
    st.code(prompt_text, language="markdown")
    copy_btn(prompt_text, "copy-prompt", "프롬프트 복사")

    if st.button("Gemini 실행", key="btn_prompt_run"):
        try:
            out = run_gemini(prompt_text, cfg)
            st.session_state["prompt_out"] = out
            st.session_state["prompt_kind"] = kind
        except Exception as e:
            st.error(f"실행 실패: {e}")

    if st.session_state.get("prompt_out"):
        st.markdown("### 생성 결과")
        st.markdown(st.session_state["prompt_out"])
        copy_btn(st.session_state["prompt_out"], "copy-prompt-out", "결과 복사")
        doc_name = st.text_input(
            "Google Docs 저장 문서명",
            value=f"[AI회의록] {meta.get('title') or '회의'} - {PROMPTS.get(st.session_state.get('prompt_kind', ''), '결과')}",
            key="prompt_doc_name",
        )
        if st.button("결과를 Google Docs 저장", key="btn_prompt_save"):
            try:
                url = save_doc(st.session_state["prompt_out"], doc_name, cfg)
                st.session_state["last_doc_url"] = url
                st.success(f"저장 완료: {url}")
            except Exception as e:
                st.error(f"저장 실패: {e}")

with t3:
    st.subheader("명부 관리")
    r_up = st.file_uploader("명부.xlsx 업로드", type=["xlsx"], key="roster_upload")
    if r_up is not None:
        try:
            st.session_state["roster_df"] = roster_load_bytes(r_up.getvalue())
            st.success("명부 업로드 완료")
        except Exception as e:
            st.error(f"명부 로드 실패: {e}")

    edited = st.data_editor(
        st.session_state["roster_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="roster_editor",
        column_config={"IsCCDefault": st.column_config.CheckboxColumn("IsCCDefault", default=False)},
    )
    try:
        st.session_state["roster_df"] = roster_norm(edited)
    except Exception as e:
        st.error(f"명부 형식 오류: {e}")

    st.download_button(
        "업데이트된 명부.xlsx 다운로드",
        data=roster_to_xlsx(st.session_state["roster_df"]),
        file_name="명부.updated.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="btn_roster_download",
    )
    if st.button("명부를 Google Drive에 저장/갱신", key="btn_roster_drive"):
        try:
            st.success(f"저장 완료: {roster_save_drive(st.session_state['roster_df'], cfg)}")
        except Exception as e:
            st.error(f"드라이브 저장 실패: {e}")

    st.markdown("---")
    st.subheader("참석자 자동완성")
    name_opts = sorted([n for n in st.session_state["roster_df"]["Name"].astype(str).tolist() if n.strip()])
    st.multiselect("참석자(검색 가능)", options=name_opts, key="selected_names")
    q = st.text_input("이름만 입력 (빠른 추가)", key="quick_name")
    matches = difflib.get_close_matches(q.strip(), name_opts, n=5, cutoff=0.45) if q.strip() else []
    pick = st.selectbox("유사도 후보", options=[""] + matches, key="quick_pick")
    if st.button("후보를 참석자에 추가", key="btn_quick_add"):
        if pick:
            cur = list(st.session_state.get("selected_names", []))
            if pick not in cur:
                cur.append(pick)
                st.session_state["selected_names"] = cur
                st.success(f"{pick} 추가됨")
            else:
                st.info("이미 추가됨")
        else:
            st.warning("후보를 선택하세요.")

    st.markdown("#### 외부 참석자")
    ext_edit = st.data_editor(st.session_state["ext_df"], num_rows="dynamic", use_container_width=True, key="ext_editor")
    st.session_state["ext_df"] = ext_norm(ext_edit)

    recipients = build_recipients(st.session_state["roster_df"], st.session_state.get("selected_names", []), st.session_state["ext_df"])
    if recipients:
        st.dataframe(
            pd.DataFrame(
                [
                    {"Name": r["name"], "Email": r["email"], "ManagerEmail": r.get("manager", ""), "IsCCDefault": r.get("cc_default", False)}
                    for r in recipients
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        p_text = participants_text(recipients)
        st.code(p_text)
        if st.button("선택 참석자를 메타에 반영", key="btn_sync_participants"):
            st.session_state["meta_participants"] = p_text
            st.success("사이드바 참석자 메타 갱신 완료")
    else:
        st.info("발송 대상이 없습니다.")

    st.markdown("---")
    st.subheader("이메일 생성 + 개인별 발송")
    g_ok = bool(cfg["gmail_impersonate"])
    s_ok = bool(cfg["smtp_host"])
    st.write(f"Gmail API 사용 가능: {'예' if g_ok else '아니오'}")
    st.write(f"SMTP 사용 가능: {'예' if s_ok else '아니오'}")

    backends = (["Gmail API"] if g_ok else []) + (["SMTP"] if s_ok else [])
    if not backends:
        backends = ["발송 비활성화"]
    backend = st.selectbox("발송 백엔드", options=backends, key="email_backend")

    if st.session_state.get("sender_backend") != backend:
        st.session_state["sender_backend"] = backend
        if backend == "SMTP":
            st.session_state["sender_name"] = cfg.get("smtp_name", "KCH Global")
            st.session_state["sender_email"] = cfg.get("smtp_from") or cfg.get("smtp_user", "")
        else:
            st.session_state["sender_name"] = cfg.get("gmail_name", "KCH Global")
            st.session_state["sender_email"] = cfg.get("gmail_impersonate", "")

    st.text_input("발신자 이름", key="sender_name")
    st.text_input("발신자 이메일", key="sender_email")
    cc_text = st.text_input("CC (콤마/세미콜론)", key="email_cc")
    bcc_text = st.text_input("BCC (콤마/세미콜론)", key="email_bcc")
    cc_manual = parse_email_csv(cc_text)
    bcc_manual = parse_email_csv(bcc_text)

    mode = st.radio("메일 유형", ["회의 초대메일", "회의 종료 Follow-up"], horizontal=True, key="email_mode")
    today = datetime.now().strftime("%Y-%m-%d")
    sub_default = f"[{meta.get('title') or '회의'}] 회의 초대 ({today})" if mode == "회의 초대메일" else f"[{meta.get('title') or '회의'}] 결과 및 Action Items ({today})"
    subject = st.text_input("메일 제목", value=sub_default, key="email_subject")

    if mode == "회의 초대메일":
        invite_info = st.text_area("회의 정보 원문", value=info_text(meta), height=200, key="invite_info")
        summary_text, doc_url, sign = "", "", ""
    else:
        invite_info = ""
        summary_text = st.text_area("회의 요약 원문", value=st.session_state.get("last_summary", ""), height=220, key="followup_summary")
        doc_url = st.text_input("회의록 URL", value=st.session_state.get("last_doc_url", ""), key="followup_url")
        sign = st.text_input("서명", value="KCH Global AI 회의록", key="followup_sign")

    if st.button("미리보기(전체/개인별) 생성", key="btn_preview_email"):
        if not recipients:
            st.warning("발송 대상이 없습니다.")
        elif backend == "발송 비활성화":
            st.warning("이메일 백엔드 설정이 없습니다.")
        else:
            previews = []
            with st.spinner("미리보기 생성 중..."):
                if mode == "회의 초대메일":
                    ptxt = build_prompt("invite", meta, {"meeting_info": invite_info})
                    try:
                        body = run_gemini(ptxt, cfg)
                    except Exception:
                        body = f"안녕하세요.\n\n{invite_info}\n\n감사합니다."
                    for r in recipients:
                        previews.append(
                            {
                                "name": r["name"],
                                "to": r["email"],
                                "cc": cc_for(r, cc_manual),
                                "bcc": bcc_manual,
                                "subject": subject,
                                "body": body,
                            }
                        )
                else:
                    for r in recipients:
                        ptxt = build_prompt(
                            "followup",
                            meta,
                            {
                                "recipient_name": r["name"],
                                "subject": subject,
                                "doc_url": doc_url,
                                "refs": meta.get("refs", ""),
                                "signature": sign,
                                "summary": summary_text,
                            },
                        )
                        try:
                            gen = run_gemini(ptxt, cfg)
                            sub, body = parse_subject_body(gen, subject)
                        except Exception:
                            sub = subject
                            body = followup_fallback(r["name"], meta.get("title", ""), summary_text, doc_url, meta.get("refs", ""), sign)
                        previews.append(
                            {
                                "name": r["name"],
                                "to": r["email"],
                                "cc": cc_for(r, cc_manual),
                                "bcc": bcc_manual,
                                "subject": sub,
                                "body": body,
                            }
                        )
            st.session_state["email_previews"] = previews
            st.success(f"미리보기 생성 완료: {len(previews)}건")

    previews = st.session_state.get("email_previews", [])
    if previews:
        idx = st.selectbox(
            "개인별 미리보기",
            options=list(range(len(previews))),
            format_func=lambda i: f"{previews[i]['name']} <{previews[i]['to']}>",
            key="preview_idx",
        )
        pv = previews[idx]
        st.markdown(f"- To: `{pv['to']}`")
        st.markdown(f"- CC: `{', '.join(pv['cc']) if pv['cc'] else '-'}`")
        st.markdown(f"- BCC: `{', '.join(pv['bcc']) if pv['bcc'] else '-'}`")
        st.markdown(f"- Subject: `{pv['subject']}`")
        st.code(pv["body"])
        copy_btn(pv["body"], "copy-email-body", "본문 복사")

    if st.button("발송", key="btn_send", disabled=(not previews or backend == "발송 비활성화")):
        s_name = st.session_state.get("sender_name", "").strip()
        s_email = st.session_state.get("sender_email", "").strip()
        if not s_name or not s_email:
            st.error("발신자 이름/이메일을 입력하세요.")
        else:
            if backend == "Gmail API" and cfg.get("gmail_impersonate"):
                if s_email.lower() != cfg["gmail_impersonate"].lower():
                    st.info("Gmail API 발송 시 발신자 이메일은 위임 계정으로 고정됩니다.")
                    s_email = cfg["gmail_impersonate"]
            logo = logo_bytes()
            ok, fail = 0, []
            with st.spinner("메일 발송 중..."):
                for pv in previews:
                    try:
                        msg, rcpts = build_mail(
                            s_name,
                            s_email,
                            pv["to"],
                            pv["cc"],
                            pv["bcc"],
                            pv["subject"],
                            pv["body"],
                            logo,
                        )
                        if backend == "Gmail API":
                            send_gmail(msg, cfg)
                        else:
                            send_smtp(msg, rcpts, cfg)
                        ok += 1
                    except Exception as e:
                        fail.append(f"{pv['to']}: {e}")
            if ok:
                st.success(f"발송 성공: {ok}건")
            if fail:
                st.error("발송 실패:\n" + "\n".join(fail))
