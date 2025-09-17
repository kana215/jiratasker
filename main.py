import os
import re
import json
import tempfile
import subprocess
from typing import List, Optional

import streamlit as st
import requests

# внешние SDK
from deepgram import Deepgram
import imageio_ffmpeg as ioffmpeg

# ================= UI / page =================
st.set_page_config(page_title="ИИ-секретарь", page_icon="🤖", layout="wide")
st.title("🤖 ИИ-секретарь для онлайн-встреч")

# ================= helpers =================
def ffmpeg_convert_to_wav(src_path: str, wav_path: str, sr: int = 16000) -> None:
    """Любой аудио/видео → WAV 16k mono через ffmpeg (бинарник берём из imageio-ffmpeg)."""
    ffmpeg_bin = ioffmpeg.get_ffmpeg_exe()  # гарантированно есть на Streamlit Cloud
    cmd = [
        ffmpeg_bin, "-hide_banner", "-y",
        "-i", src_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ac", "1",
        wav_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg conversion failed: " + proc.stderr.decode(errors="ignore")[:800])

def detect_lang_code(text: str) -> str:
    """Грубое определение языка по наличию кириллицы."""
    cyr = sum('а' <= ch.lower() <= 'я' or ch == 'ё' for ch in text)
    lat = sum('a' <= ch.lower() <= 'z' for ch in text)
    return "ru" if cyr > lat else "en"

def split_sentences(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+|[\n\r]+|•\s*| - ", text.strip())
    return [s.strip(" \t-—•") for s in sents if len(s.strip()) > 2]

def expand_compounds(s: str) -> List[str]:
    parts = re.split(r"\b(и|а также|затем|после этого|потом|далее|and then|and)\b", s, flags=re.IGNORECASE)
    out = []
    for p in parts:
        if re.fullmatch(r"(и|а также|затем|после этого|потом|далее|and then|and)", p, flags=re.IGNORECASE):
            continue
        frag = p.strip(" ,.;:—-")
        if frag:
            out.append(frag)
    return out or [s]

# базовый список глаголов-триггеров (RU/EN)
VERB_RE = r"(провести|подготовить|отправить|создать|написать|проверить|созвониться|добавить|исправить|закрыть|запланировать|согласовать|обновить|описать|развернуть|подключить|оформить|назначить|организовать|презентовать|ожидать|собрать|дать|выполнить|подтвердить|утвердить|поделиться|скинуть|зафиксировать|напомнить|подвести итоги|review|plan|schedule|deploy|implement|prepare|send|create|write|check|fix|update|investigate|present|follow up)"

def candidate_actions(text: str) -> List[str]:
    """Выделяем кандидатов на задачи: только фразы с глаголами-триггерами."""
    out = []
    for s in split_sentences(text):
        if not re.search(VERB_RE, s, flags=re.IGNORECASE):
            continue
        for sub in expand_compounds(s):
            sub = re.sub(r"(?i)\b(нужно|надо|будет|давайте|давай|предлагаю)\s+", "", sub).strip()
            m = re.search(VERB_RE + r".*", sub, flags=re.IGNORECASE)
            frag = sub[m.start():].strip() if m else sub
            frag = re.split(r"[.;!?]", frag)[0].strip(" ,.;:—-")
            words = frag.split()
            if len(words) > 16:
                frag = " ".join(words[:16])
            if len(frag) >= 3:
                out.append(frag)
    seen, res = set(), []
    for t in out:
        if t not in seen:
            res.append(t); seen.add(t)
    return res

# ============== NLI-фильтр (HuggingFace) ==============
@st.cache_resource(show_spinner=False)
def load_nli_models():
    """
    Загружаем две бесплатные модели (скачаются 1 раз):
    - RU: cointegrated/rubert-base-cased-nli-threeway
    - EN: facebook/bart-large-mnli
    Если в папке models/nli-ru или models/nli-en лежат локальные копии — используем их.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    models = {}

    ru_dir = "models/nli-ru"
    ru_name = "cointegrated/rubert-base-cased-nli-threeway"
    try:
        tok_ru = AutoTokenizer.from_pretrained(ru_dir if os.path.isdir(ru_dir) else ru_name,
                                               local_files_only=os.path.isdir(ru_dir))
        mdl_ru = AutoModelForSequenceClassification.from_pretrained(ru_dir if os.path.isdir(ru_dir) else ru_name,
                                                                    local_files_only=os.path.isdir(ru_dir))
        models["ru"] = (tok_ru, mdl_ru)
    except Exception:
        models["ru"] = None

    en_dir = "models/nli-en"
    en_name = "facebook/bart-large-mnli"
    try:
        tok_en = AutoTokenizer.from_pretrained(en_dir if os.path.isdir(en_dir) else en_name,
                                               local_files_only=os.path.isdir(en_dir))
        mdl_en = AutoModelForSequenceClassification.from_pretrained(en_dir if os.path.isdir(en_dir) else en_name,
                                                                    local_files_only=os.path.isdir(en_dir))
        models["en"] = (tok_en, mdl_en)
    except Exception:
        models["en"] = None

    return models

import torch

def nli_is_task(premise: str, lang: str, models, threshold: float = 0.60) -> bool:
    """Проверяем гипотезу 'это поручение/task' через NLI. True, если entailment >= threshold."""
    pair = models.get("ru" if lang == "ru" else "en")
    if not pair:
        return True  # нет модели → не фильтруем

    tok, mdl = pair
    hypothesis = "Это конкретное поручение, которое нужно выполнить." if lang == "ru" else \
                 "This is an actionable task to be done."
    inputs = tok([premise], [hypothesis], return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = mdl(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    entail_p = float(probs[-1])  # последний — entailment
    return entail_p >= threshold

def extract_tasks_with_nli(text: str) -> List[str]:
    lang = detect_lang_code(text)
    cand = candidate_actions(text)
    if not cand:
        return []
    models = load_nli_models()

    out = []
    bar = st.progress(0, text="Фильтруем кандидатов задач (NLI)…")
    total = len(cand)
    for i, c in enumerate(cand, 1):
        if nli_is_task(c, lang, models):
            c2 = re.sub(r"(?i)^(прошу|нужно|надо|будет|давайте|давай)\s+", "", c).strip(" -—•")
            c2 = re.sub(r"[\.!\s]+$", "", c2)
            if len(c2) > 140:
                c2 = c2[:140]
            if c2:
                out.append(c2)
        bar.progress(i/total)
    # дедуп
    seen, res = set(), []
    for t in out:
        if t and t not in seen:
            res.append(t); seen.add(t)
    return res

def _adf_paragraph(text: str):
    return {"type": "paragraph", "content": [{"type": "text", "text": text}]}

def create_jira_task(base_url: str, email: str, token: str, project_key: str,
                     summary: str, description: Optional[str] = None) -> str:
    """Создаём задачу в Jira Cloud (API v3) c ADF-описанием."""
    url = f"{base_url.rstrip('/')}/rest/api/3/issue"
    auth = (email, token)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    adf_desc = {
        "type": "doc",
        "version": 1,
        "content": [
            _adf_paragraph(description or ""),
            _adf_paragraph("— создано автоматически: AI Secretary (Deepgram → Jira)"),
        ],
    }

    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": (summary or "Task")[:254],
            "issuetype": {"name": "Task"},
            "description": adf_desc,
        }
    }
    r = requests.post(url, auth=auth, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Jira error {r.status_code}: {r.text[:600]}")
    return r.json().get("key", "?")

# ================= UI: загрузка =================
up = st.file_uploader(
    "Загрузите аудио/видео (mp3, wav, ogg, m4a, mp4, mkv)",
    type=["mp3", "wav", "ogg", "m4a", "mp4", "mkv"]
)

# ================= Распознавание =================
if up:
    st.audio(up.getvalue())
    if st.button("🎙️ Распознать и выделить задачи"):
        # сохраняем и конвертим
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as tmp:
            tmp.write(up.getvalue())
            src_path = tmp.name
        wav_path = src_path + ".wav"

        with st.spinner("Конвертация в WAV (16kHz mono)…"):
            ffmpeg_convert_to_wav(src_path, wav_path)

        # Deepgram ключ из Streamlit Secrets (или из env для локалки)
        DG_KEY = st.secrets.get("DEEPGRAM_API_KEY") or os.getenv("DEEPGRAM_API_KEY")
        if not DG_KEY:
            st.error("❌ Нет ключа Deepgram. Добавьте DEEPGRAM_API_KEY в Secrets (Streamlit Cloud) или env.")
            st.stop()

        dg = Deepgram(DG_KEY)

        with st.spinner("Распознаём в Deepgram…"):
            with open(wav_path, "rb") as f:
                source = {"buffer": f, "mimetype": "audio/wav"}
                res = dg.transcription.sync_prerecorded(
                    source,
                    {"punctuate": True, "smart_format": True, "language": "ru"}  # можно "auto" если будет много EN
                )

        transcript = res["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        st.session_state["transcript"] = transcript

        # Извлечение задач через NLI
        st.info("🧠 Извлекаем задачи (NLI-фильтр, бесплатно)…")
        st.session_state["tasks"] = extract_tasks_with_nli(transcript)

# ================= Вывод =================
if "transcript" in st.session_state:
    st.subheader("📄 Распознанный текст")
    st.text_area("Транскрипт", st.session_state["transcript"], height=220)

if "tasks" in st.session_state:
    st.subheader("📌 Извлечённые задачи")
    tasks = st.session_state["tasks"]
    if tasks:
        for i, t in enumerate(tasks, 1):
            st.markdown(f"- **{i}.** {t}")
    else:
        st.info("Не нашёл явных поручений. Сформулируйте фразы с глаголами действия.")

    # ===== Jira =====
    st.divider()
    st.subheader("📤 Отправка задач в Jira")
    with st.form("jira_form"):
        jira_base   = st.text_input("Jira Base URL", placeholder="https://your-org.atlassian.net")
        jira_email  = st.text_input("Jira Email", placeholder="you@example.com")
        jira_token  = st.text_input("Jira API Token", type="password")
        jira_proj   = st.text_input("Project Key", placeholder="TEST")
        send = st.form_submit_button("Отправить в Jira ✅")

    if send:
        if not (jira_base and jira_email and jira_token and jira_proj):
            st.error("Заполните все поля Jira.")
        else:
            results = []
            bar = st.progress(0, text="Создаём задачи в Jira…")
            total = max(1, len(tasks))
            for i, task in enumerate(tasks, 1):
                try:
                    key = create_jira_task(jira_base, jira_email, jira_token, jira_proj,
                                           summary=task, description=task)
                    results.append(("ok", key, task))
                except Exception as e:
                    results.append(("err", str(e), task))
                bar.progress(i/total)
            st.subheader("🧾 Результаты создания задач")
            ok = [r for r in results if r[0] == "ok"]
            err = [r for r in results if r[0] == "err"]
            if ok:
                st.success("Созданы: " + ", ".join([k for _, k, _ in ok]))
            for _, msg, t in err:
                st.error(f"Не создана: „{t}“ → {msg}")
