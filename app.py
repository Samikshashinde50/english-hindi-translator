import os
import html
import json
from io import BytesIO
import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit import IndicProcessor
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS, lang as gtts_lang
from langdetect import detect, LangDetectException


# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="LanguageBridge Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# 2. CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1600px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }
    .app-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 6px;
    }
    .app-logo {
        background: #3157D5;
        color: white;
        width: 54px;
        height: 54px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        flex-shrink: 0;
    }
    .app-title {
        color: var(--text-color);
        font-size: 30px;
        font-weight: 750;
        margin: 0;
        line-height: 1.3;
    }
    .app-subtitle, .panel-description, .status-note {
        color: var(--text-color);
        opacity: 0.72;
        font-size: 13px;
    }
    .app-subtitle { font-size: 14px; margin-top: 3px; }
    .panel-heading {
        font-size: 21px;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 4px;
    }
    .panel-description { margin-bottom: 15px; }
    .translation-result {
        color: var(--text-color);
        background: var(--secondary-background-color);
        border: 1px solid var(--primary-color);
        border-radius: 14px;
        padding: 22px;
        min-height: 145px;
        font-size: 19px;
        line-height: 1.8;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }
    .result-label {
        color: var(--text-color);
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 600;
    }
    div[data-testid="stTextArea"] textarea {
        border-radius: 12px;
        line-height: 1.6;
    }
    @media (max-width: 768px) {
        .block-container { padding: 1rem; }
        .app-title { font-size: 24px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 3. MODEL AND LANGUAGE CONFIGURATION
# ---------------------------------------------------------

MODEL_EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"
MODEL_INDIC_EN = "ai4bharat/indictrans2-indic-en-dist-200M"
MODEL_INDIC_INDIC = "ai4bharat/indictrans2-indic-indic-dist-320M"

LANGUAGES = {
    "Assamese": "asm_Beng",
    "Bengali": "ben_Beng",
    "Bodo": "brx_Deva",
    "Dogri": "doi_Deva",
    "English": "eng_Latn",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Kashmiri": "kas_Arab",
    "Konkani": "gom_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Manipuri": "mni_Beng",
    "Marathi": "mar_Deva",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sindhi": "snd_Arab",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab",
}

VOICE_CODES = {
    "English": "en", "Assamese": "as", "Bengali": "bn",
    "Gujarati": "gu", "Hindi": "hi", "Kannada": "kn",
    "Malayalam": "ml", "Marathi": "mr", "Nepali": "ne",
    "Odia": "or", "Punjabi": "pa", "Tamil": "ta",
    "Telugu": "te", "Urdu": "ur",
}

TTS_CODES = {
    "Assamese": "as", "Bengali": "bn", "English": "en",
    "Gujarati": "gu", "Hindi": "hi", "Kannada": "kn",
    "Malayalam": "ml", "Marathi": "mr", "Nepali": "ne",
    "Odia": "or", "Punjabi": "pa", "Tamil": "ta",
    "Telugu": "te", "Urdu": "ur",
}

DETECT_MAP = {
    "as": "Assamese", "bn": "Bengali", "en": "English",
    "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada",
    "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali",
    "or": "Odia", "pa": "Punjabi", "sa": "Sanskrit",
    "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
}


# ---------------------------------------------------------
# 4. SESSION STATE
# ---------------------------------------------------------

DEFAULTS = {
    "source_text": "",
    "translation_result": "",
    "translation_audio": None,
    "last_spoken_text": "",
    "detected_language": "",
    "last_translation_pair": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_translation():
    st.session_state.translation_result = ""
    st.session_state.translation_audio = None
    st.session_state.last_translation_pair = ""


def clear_workspace():
    st.session_state.source_text = ""
    st.session_state.translation_result = ""
    st.session_state.translation_audio = None
    st.session_state.last_spoken_text = ""
    st.session_state.detected_language = ""
    st.session_state.last_translation_pair = ""


# ---------------------------------------------------------
# 5. HUGGING FACE TOKEN
# ---------------------------------------------------------

def get_hf_token():
    """Read HF_TOKEN from Streamlit Secrets or environment."""
    try:
        token = st.secrets.get("HF_TOKEN")
    except Exception:
        token = None
    return token or os.getenv("HF_TOKEN")


# ---------------------------------------------------------
# 6. LANGUAGE DETECTION
# ---------------------------------------------------------

def detect_source_language(text):
    text = text.strip()
    if not text:
        return None, "Enter text before detecting the language."

    try:
        detected = detect(text)
        language = DETECT_MAP.get(detected)
        if language in LANGUAGES:
            return language, "Language detected automatically."
    except LangDetectException:
        pass

    # Script-based fallback. These are guesses, not guaranteed detection.
    if any("\u0900" <= c <= "\u097F" for c in text):
        return "Hindi", (
            "The text uses Devanagari script. Hindi is only a tentative "
            "guess; Marathi and other languages may use the same script."
        )
    if any("\u0980" <= c <= "\u09FF" for c in text):
        return "Bengali", (
            "The text uses Bengali script. Please confirm the language."
        )
    if any("\u0600" <= c <= "\u06FF" for c in text):
        return "Urdu", (
            "The text uses an Arabic-based script. Please confirm the language."
        )

    return None, "Could not detect the language. Please select it manually."


# ---------------------------------------------------------
# 7. LOAD INDIC TRANS2 MODEL
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False, max_entries=1)
def load_translation_model(model_id, token):
    """
    Cache only one model to reduce memory use.
    IndicTrans2 model repositories may require accepting HF conditions.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_args = {"trust_remote_code": True}
    if token:
        load_args["token"] = token

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **load_args)

        model_args = dict(load_args)
        if device == "cuda":
            model_args["torch_dtype"] = torch.float16

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            **model_args,
        )
        model.to(device)
        model.eval()

        processor = IndicProcessor(inference=True)
        return tokenizer, model, processor, device

    except Exception as exc:
        message = str(exc)
        lowered = message.lower()

        if any(term in lowered for term in (
            "401", "403", "gated repo", "access to model",
            "restricted", "unauthorized"
        )):
            raise RuntimeError(
                "Hugging Face denied access to this model. Open the model "
                "page on Hugging Face, sign in, accept its access conditions "
                "if shown, create a Read token, and add it as HF_TOKEN in "
                "Streamlit Cloud → App settings → Secrets."
            ) from exc

        if "out of memory" in lowered:
            raise RuntimeError(
                "The server ran out of memory while loading the translation "
                "model. The IndicTrans2 model may be too large for the "
                "available Streamlit Cloud memory."
            ) from exc

        raise RuntimeError(
            f"Could not load the translation model: {message}"
        ) from exc


# ---------------------------------------------------------
# 8. TRANSLATION
# ---------------------------------------------------------

def translate_text(text, source_name, target_name):
    text = text.strip()

    if not text:
        raise ValueError("Please enter or speak some text first.")
    if source_name not in LANGUAGES:
        raise ValueError("Please select a valid input language.")
    if target_name not in LANGUAGES:
        raise ValueError("Please select a valid target language.")
    if source_name == target_name:
        raise ValueError("Input and target languages must be different.")

    source_code = LANGUAGES[source_name]
    target_code = LANGUAGES[target_name]

    if source_name == "English":
        model_id = MODEL_EN_INDIC
    elif target_name == "English":
        model_id = MODEL_INDIC_EN
    else:
        model_id = MODEL_INDIC_INDIC

    tokenizer, model, processor, device = load_translation_model(
        model_id, get_hf_token()
    )

    prepared = processor.preprocess_batch(
        [text],
        src_lang=source_code,
        tgt_lang=target_code,
    )

    inputs = tokenizer(
        prepared,
        truncation=True,
        padding="longest",
        max_length=256,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=4,
            num_return_sequences=1,
        )

    decoded = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    results = processor.postprocess_batch(decoded, lang=target_code)

    if not results or not results[0].strip():
        raise RuntimeError("The model returned an empty translation.")

    return results[0].strip(), model_id


# ---------------------------------------------------------
# 9. TEXT-TO-SPEECH
# ---------------------------------------------------------

def create_audio(text, language_name):
    text = text.strip()
    if not text:
        raise ValueError("Translate some text before generating audio.")

    code = TTS_CODES.get(language_name)
    if not code:
        raise ValueError(
            f"Text-to-speech is not configured for {language_name}."
        )

    available_languages = gtts_lang.tts_langs()
    if code not in available_languages:
        raise ValueError(
            f"gTTS does not currently list support for {language_name}."
        )

    audio_buffer = BytesIO()
    speech = gTTS(text=text, lang=code, slow=False)
    speech.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.getvalue()


# ---------------------------------------------------------
# 10. HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="app-header">
        <div class="app-logo">🌐</div>
        <div>
            <div class="app-title">LanguageBridge</div>
            <div class="app-subtitle">Translate naturally. Speak confidently.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")
st.caption(
    "Translate between English and supported Indian languages. "
    "Enter text or use voice input."
)


# ---------------------------------------------------------
# 11. DASHBOARD
# ---------------------------------------------------------

left_col, right_col = st.columns([3, 7], gap="large")


# LEFT: SETTINGS
with left_col:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-heading">⚙️ Language Settings</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="panel-description">Choose the source and target languages.</div>',
            unsafe_allow_html=True,
        )

        source_options = ["Auto-detect"] + list(LANGUAGES.keys())
        source_language = st.selectbox(
            "Input language",
            options=source_options,
            index=0,
            key="source_language_select",
            help="Automatic detection is approximate. Manual selection is more reliable.",
        )
        target_language = st.selectbox(
            "Translate into",
            options=list(LANGUAGES.keys()),
            index=list(LANGUAGES.keys()).index("Hindi"),
            key="target_language_select",
        )

        st.divider()
        st.markdown("**Translation controls**")

        translated_text = st.session_state.translation_result
        if translated_text:
            file_name = (
                f"translation_{target_language.lower().replace(' ', '_')}.txt"
            )
            st.download_button(
                "⬇️ Download translation",
                data=translated_text,
                file_name=file_name,
                mime="text/plain",
                use_container_width=True,
                key="download_translation",
            )
        else:
            st.button(
                "⬇️ Download translation",
                disabled=True,
                use_container_width=True,
                help="Translate your text first.",
            )

        st.button(
            "🗑️ Clear translation",
            on_click=clear_translation,
            use_container_width=True,
        )
        st.button(
            "🧹 Clear workspace",
            on_click=clear_workspace,
            use_container_width=True,
        )

        st.divider()
        st.markdown("**Selected languages**")
        source_display = (
            "Automatic detection"
            if source_language == "Auto-detect"
            else source_language
        )
        st.caption(f"From: {source_display}")
        st.caption(f"To: {target_language}")
        st.markdown(
            '<div class="status-note">Model: AI4Bharat IndicTrans2<br>'
            'Translation quality depends on the language pair and input.</div>',
            unsafe_allow_html=True,
        )


# RIGHT: WORKSPACE
with right_col:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-heading">✍️ Translation Workspace</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="panel-description">Type your text or speak into the microphone.</div>',
            unsafe_allow_html=True,
        )

        voice_language = (
            "en" if source_language == "Auto-detect"
            else VOICE_CODES.get(source_language, "en")
        )

        st.markdown("**Enter or speak your text**")
        voice_text = speech_to_text(
            language=voice_language,
            start_prompt="🎙️ Start speaking",
            stop_prompt="⏹️ Stop recording",
            just_once=True,
            use_container_width=True,
            key="voice_input",
        )

        if (
            isinstance(voice_text, str)
            and voice_text.strip()
            and voice_text != st.session_state.last_spoken_text
        ):
            st.session_state.source_text = voice_text.strip()
            st.session_state.last_spoken_text = voice_text

        st.text_area(
            "Text to translate",
            key="source_text",
            height=175,
            placeholder="Enter your text here...\n\nExample: Hello, how are you?",
            label_visibility="collapsed",
        )

        button_col1, button_col2 = st.columns(2)
        with button_col1:
            translate_clicked = st.button(
                "🌐 Translate",
                type="primary",
                use_container_width=True,
                key="translate_button",
            )
        with button_col2:
            detect_clicked = st.button(
                "🔎 Detect language",
                use_container_width=True,
                key="detect_button",
            )

        if detect_clicked:
            detected_name, detection_note = detect_source_language(
                st.session_state.source_text
            )
            if detected_name:
                st.session_state.detected_language = detected_name
                st.info(f"Likely language: **{detected_name}**. {detection_note}")
            else:
                st.warning(detection_note)

        if translate_clicked:
            text_to_translate = st.session_state.source_text.strip()
            if not text_to_translate:
                st.warning("Please enter or speak some text first.")
            else:
                actual_source = source_language
                if actual_source == "Auto-detect":
                    detected_name, detection_note = detect_source_language(
                        text_to_translate
                    )
                    if not detected_name:
                        st.error(
                            "I could not detect the input language. "
                            "Choose it manually from Input language."
                        )
                        st.stop()
                    actual_source = detected_name
                    st.info(
                        f"Detected input language: **{actual_source}**. "
                        f"{detection_note}"
                    )

                if actual_source == target_language:
                    st.error(
                        "Input and target languages are the same. "
                        "Choose a different target language."
                    )
                else:
                    with st.spinner(
                        f"Translating {actual_source} to {target_language}..."
                    ):
                        try:
                            result, model_id = translate_text(
                                text_to_translate,
                                actual_source,
                                target_language,
                            )
                            st.session_state.translation_result = result
                            st.session_state.translation_audio = None
                            st.session_state.last_translation_pair = (
                                f"{actual_source} → {target_language}"
                            )
                            st.session_state.detected_language = actual_source
                            st.success(
                                f"Translation complete: "
                                f"{actual_source} → {target_language}"
                            )
                        except Exception as exc:
                            st.error(str(exc))

        st.divider()
        st.markdown(
            '<div class="result-label">📝 Translation Result</div>',
            unsafe_allow_html=True,
        )

        translated_text = st.session_state.translation_result
        if translated_text:
            # Escape model output before placing it in HTML.
            safe_result = html.escape(translated_text).replace("\n", "<br>")
            st.markdown(
                f'<div class="translation-result">{safe_result}</div>',
                unsafe_allow_html=True,
            )

            if st.session_state.last_translation_pair:
                st.caption(st.session_state.last_translation_pair)

            encoded_text = json.dumps(translated_text)
            components.html(
                f"""
                <button id="copyTranslationButton"
                    style="padding:10px 16px;border-radius:9px;border:1px solid #777;
                    cursor:pointer;font-size:14px;font-weight:600;
                    background:#3157D5;color:white;">
                    📋 Copy translation
                </button>
                <span id="copyStatus" style="margin-left:10px;font-size:13px;"></span>
                <script>
                    const textToCopy = {encoded_text};
                    const copyButton = document.getElementById("copyTranslationButton");
                    const copyStatus = document.getElementById("copyStatus");
                    copyButton.addEventListener("click", async () => {{
                        try {{
                            await navigator.clipboard.writeText(textToCopy);
                            copyStatus.textContent = "Copied!";
                        }} catch (error) {{
                            copyStatus.textContent =
                                "Copy unavailable. Please select the text.";
                        }}
                    }});
                </script>
                """,
                height=55,
            )

            st.markdown("**🔊 Listen to the translation**")
            if st.button(
                "Generate voice audio",
                use_container_width=False,
                key="generate_audio",
            ):
                with st.spinner("Generating audio..."):
                    try:
                        st.session_state.translation_audio = create_audio(
                            translated_text, target_language
                        )
                    except Exception as exc:
                        st.session_state.translation_audio = None
                        st.warning(str(exc))

            if st.session_state.translation_audio:
                st.audio(
                    st.session_state.translation_audio,
                    format="audio/mp3",
                )
                st.download_button(
                    "⬇️ Download audio",
                    data=st.session_state.translation_audio,
                    file_name="translation_audio.mp3",
                    mime="audio/mpeg",
                    key="download_audio",
                )
        else:
            st.markdown(
                """
                <div class="translation-result">
                    Your translated text will appear here.<br><br>
                    <span style="opacity:0.7;font-size:14px;">
                        Enter your text above and press Translate.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------
# 12. FOOTER
# ---------------------------------------------------------

st.write("")
st.markdown(
    """
    <div style="text-align:center;padding:12px;color:var(--text-color);
                opacity:0.65;font-size:12px;">
        LanguageBridge · Powered by AI4Bharat IndicTrans2<br>
        Automatic language detection and speech support may vary by language.
    </div>
    """,
    unsafe_allow_html=True,
)

