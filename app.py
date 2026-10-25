
import os
import gc
import json
from io import BytesIO

import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS, lang as gtts_lang
from langdetect import detect, LangDetectException

# Support both common IndicTransToolkit import paths.
try:
    from IndicTransToolkit import IndicProcessor
except ImportError:
    from IndicTransToolkit.processor import IndicProcessor


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
# 2. THEME-FRIENDLY CSS
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

    .app-subtitle {
        color: var(--text-color);
        opacity: 0.72;
        font-size: 14px;
        margin-top: 3px;
    }

    .panel-heading {
        font-size: 21px;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 4px;
    }

    .panel-description {
        color: var(--text-color);
        opacity: 0.72;
        font-size: 13px;
        margin-bottom: 15px;
    }

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

    .status-note {
        color: var(--text-color);
        opacity: 0.75;
        font-size: 12px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 600;
    }

    div[data-testid="stTextArea"] textarea {
        border-radius: 12px;
        line-height: 1.6;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }

        .app-title {
            font-size: 24px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 3. MODEL CONFIGURATION
# ---------------------------------------------------------

MODEL_EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"
MODEL_INDIC_EN = "ai4bharat/indictrans2-indic-en-dist-200M"
MODEL_INDIC_INDIC = "ai4bharat/indictrans2-indic-indic-dist-200M"


# Display name -> IndicTrans2 language code.
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

# Speech recognition languages. Automatic detection is not
# reliable for voice transcription, so voice input uses the
# selected input language when available.
VOICE_CODES = {
    "English": "en",
    "Assamese": "as",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Nepali": "ne",
    "Odia": "or",
    "Punjabi": "pa",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
}

# gTTS output language codes. Actual availability is checked
# against the installed gTTS language list before generating audio.
TTS_CODES = {
    "Assamese": "as",
    "Bengali": "bn",
    "English": "en",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Nepali": "ne",
    "Odia": "or",
    "Punjabi": "pa",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
}

# langdetect language codes -> display names.
DETECT_MAP = {
    "as": "Assamese",
    "bn": "Bengali",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
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
    """Clear the result and audio, but keep the input text."""
    st.session_state.translation_result = ""
    st.session_state.translation_audio = None
    st.session_state.last_translation_pair = ""


def clear_workspace():
    """Clear input, result, and generated audio."""
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
    """
    Detect the likely source language.

    Detection is approximate. Some Indian languages share scripts,
    so manually selecting the source language is more reliable.
    """
    text = text.strip()

    if not text:
        return None, "Enter text before detecting the language."

    # Devanagari: Hindi, Marathi, Nepali and several others.
    if any("\u0900" <= char <= "\u097F" for char in text):
        try:
            detected = detect(text)
            language = DETECT_MAP.get(detected)

            if language in LANGUAGES:
                return language, (
                    "Detected from text. Languages using the same "
                    "script can be confused."
                )
        except LangDetectException:
            pass

        return "Hindi", (
            "This text uses Devanagari. Hindi is a tentative guess; "
            "choose the actual language manually if needed."
        )

    # Bengali, Assamese and some Manipuri text share this script.
    if any("\u0980" <= char <= "\u09FF" for char in text):
        try:
            detected = detect(text)
            language = DETECT_MAP.get(detected)

            if language in LANGUAGES:
                return language, (
                    "Detected from text. Bengali-script languages "
                    "can be confused."
                )
        except LangDetectException:
            pass

        return "Bengali", (
            "This text uses the Bengali script. Please confirm "
            "the source language."
        )

    # Arabic script can be used by Urdu, Sindhi and Kashmiri.
    if any("\u0600" <= char <= "\u06FF" for char in text):
        try:
            detected = detect(text)
            language = DETECT_MAP.get(detected)

            if language in LANGUAGES:
                return language, (
                    "Detected from text. Arabic-script languages "
                    "can be confused."
                )
        except LangDetectException:
            pass

        return "Urdu", (
            "This text uses an Arabic-based script. Please confirm "
            "the source language."
        )

    try:
        detected = detect(text)
        language = DETECT_MAP.get(detected)

        if language in LANGUAGES:
            return language, "Language detected automatically."

    except LangDetectException:
        pass

    return None, "Could not detect the language. Please select it manually."


# ---------------------------------------------------------
# 7. LOAD INDIC TRANS MODEL
# ---------------------------------------------------------

@st.cache_resource(max_entries=1, show_spinner=False)
def load_translation_model(model_id, token):
    """
    Load one model at a time to reduce memory usage.

    If a model is gated, the HF token must have permission to access it.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_args = {"trust_remote_code": True}
    if token:
        load_args["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        **load_args,
    )

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


# ---------------------------------------------------------
# 8. TRANSLATION FUNCTION
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
        raise ValueError(
            "Input and target languages are the same. "
            "Please select different languages."
        )

    source_code = LANGUAGES[source_name]
    target_code = LANGUAGES[target_name]

    if source_name == "English":
        model_id = MODEL_EN_INDIC
    elif target_name == "English":
        model_id = MODEL_INDIC_EN
    else:
        model_id = MODEL_INDIC_INDIC

    token = get_hf_token()

    try:
        tokenizer, model, processor, device = load_translation_model(
            model_id,
            token,
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
            return_tensors="pt",
            return_attention_mask=True,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                max_length=256,
                num_beams=4,
                num_return_sequences=1,
            )

        decoded = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        results = processor.postprocess_batch(
            decoded,
            lang=target_code,
        )

        if not results or not results[0].strip():
            raise RuntimeError("The model returned an empty translation.")

        return results[0].strip(), model_id

    except Exception as exc:
        error_text = str(exc)

        if (
            "401" in error_text
            or "403" in error_text
            or "gated repo" in error_text.lower()
            or "access to model" in error_text.lower()
        ):
            raise RuntimeError(
                "Hugging Face denied access to this model. Open its "
                "model page, accept any required access conditions, "
                "and add a Hugging Face Read token as HF_TOKEN in "
                "your Streamlit Secrets."
            ) from exc

        raise RuntimeError(
            f"Translation failed: {error_text}"
        ) from exc


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

    speech = gTTS(
        text=text,
        lang=code,
        slow=False,
    )

    speech.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    return audio_buffer.getvalue()


# ---------------------------------------------------------
# 10. APP HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="app-header">
        <div class="app-logo">🌐</div>
        <div>
            <div class="app-title">LanguageBridge</div>
            <div class="app-subtitle">
                Translate naturally. Speak confidently.
            </div>
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
# 11. TWO-COLUMN DASHBOARD: 30% / 70%
# ---------------------------------------------------------

left_col, right_col = st.columns(
    [3, 7],
    gap="large",
)


# ---------------------------------------------------------
# LEFT PANEL: LANGUAGE SETTINGS + DOWNLOAD + CLEAR
# ---------------------------------------------------------

with left_col:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-heading">⚙️ Language Settings</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="panel-description">'
            'Choose the source and target languages.'
            '</div>',
            unsafe_allow_html=True,
        )

        source_options = ["Auto-detect"] + list(LANGUAGES.keys())

        source_language = st.selectbox(
            "Input language",
            options=source_options,
            index=0,
            key="source_language_select",
            help=(
                "Auto-detection is approximate. Select the language "
                "manually when the detected language is incorrect."
            ),
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
                f"translation_"
                f"{target_language.lower().replace(' ', '_')}.txt"
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
            '<div class="status-note">'
            'Model: AI4Bharat IndicTrans2<br>'
            'Translation quality depends on the language pair and input.'
            '</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# RIGHT PANEL: INPUT + VOICE + RESULT + AUDIO
# ---------------------------------------------------------

with right_col:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-heading">✍️ Translation Workspace</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="panel-description">'
            'Type your text or speak into the microphone.'
            '</div>',
            unsafe_allow_html=True,
        )

        # Select a recognition language for voice input.
        if source_language == "Auto-detect":
            voice_language = "en"
        else:
            voice_language = VOICE_CODES.get(source_language, "en")

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
            voice_text
            and isinstance(voice_text, str)
            and voice_text.strip()
            and voice_text != st.session_state.last_spoken_text
        ):
            st.session_state.source_text = voice_text.strip()
            st.session_state.last_spoken_text = voice_text

        input_text = st.text_area(
            "Text to translate",
            key="source_text",
            height=175,
            placeholder=(
                "Enter your text here...\n\n"
                "Example: Hello, how are you?"
            ),
            label_visibility="collapsed",
        )

        button_col1, button_col2 = st.columns([1, 1])

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

        # Language detection.
        if detect_clicked:
            detected_name, detection_note = detect_source_language(
                st.session_state.source_text
            )

            if detected_name:
                st.session_state.detected_language = detected_name
                st.info(
                    f"Likely language: **{detected_name}**. "
                    f"{detection_note}"
                )
            else:
                st.warning(detection_note)

        # Translation.
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

        # Always render the result panel, including in dark mode.
        st.divider()

        st.markdown(
            '<div class="result-label">📝 Translation Result</div>',
            unsafe_allow_html=True,
        )

        translated_text = st.session_state.translation_result

        if translated_text:
            st.markdown(
                '<div class="translation-result">'
                + translated_text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
                + "</div>",
                unsafe_allow_html=True,
            )

            if st.session_state.last_translation_pair:
                st.caption(st.session_state.last_translation_pair)

            # Copy result to clipboard in the browser.
            # The text is JSON-encoded before being inserted into JavaScript.
            encoded_text = json.dumps(translated_text)

            components.html(
                f"""
                <button id="copyTranslationButton"
                    style="
                        padding: 10px 16px;
                        border-radius: 9px;
                        border: 1px solid #777;
                        cursor: pointer;
                        font-size: 14px;
                        font-weight: 600;
                        background: #3157D5;
                        color: white;
                    ">
                    📋 Copy translation
                </button>
                <span id="copyStatus"
                    style="margin-left: 10px; font-size: 13px;"></span>

                <script>
                    const textToCopy = {encoded_text};
                    const copyButton =
                        document.getElementById("copyTranslationButton");
                    const copyStatus =
                        document.getElementById("copyStatus");

                    copyButton.addEventListener("click", async () => {{
                        try {{
                            await navigator.clipboard.writeText(textToCopy);
                            copyStatus.textContent = "Copied!";
                        }} catch (error) {{
                            copyStatus.textContent =
                                "Copy not available. Please select the text.";
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
                        audio_bytes = create_audio(
                            translated_text,
                            target_language,
                        )
                        st.session_state.translation_audio = audio_bytes

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
                    Your translated text will appear here.
                    <br><br>
                    <span style="opacity: 0.7; font-size: 14px;">
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
    <div style="
        text-align: center;
        padding: 12px;
        color: var(--text-color);
        opacity: 0.65;
        font-size: 12px;
    ">
        LanguageBridge · Powered by AI4Bharat IndicTrans2
        <br>
        Automatic language detection and speech support may vary by language.
    </div>
    """,
    unsafe_allow_html=True,
)
