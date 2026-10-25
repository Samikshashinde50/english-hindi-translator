
import os
import gc
from io import BytesIO
from collections import Counter

import streamlit as st
import streamlit.components.v1 as components
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS, lang as gtts_lang
from langdetect import detect_langs, DetectorFactory


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LanguageBridge | AI Translator",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="collapsed",
)

DetectorFactory.seed = 0

st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        text-align: center;
        padding: 1.2rem 0 1rem 0;
    }

    .hero h1 {
        font-size: 2.2rem;
        margin-bottom: 0.35rem;
    }

    .hero p {
        color: #777777;
        font-size: 1rem;
    }

    .section-heading {
        font-size: 1.2rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    div.stButton > button[kind="primary"] {
        width: 100%;
        min-height: 3rem;
        font-weight: 650;
        border-radius: 0.7rem;
    }

    div[data-testid="stDownloadButton"] button {
        width: 100%;
        border-radius: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🌐 LanguageBridge</h1>
        <p>Your multilingual AI translation assistant</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Translate between English and supported Indian languages, "
    "or between supported Indian languages."
)


# =========================================================
# 2. MODEL CONFIGURATION
# =========================================================

MODEL_IDS = {
    "en_indic": "ai4bharat/indictrans2-en-indic-dist-200M",
    "indic_en": "ai4bharat/indictrans2-indic-en-dist-200M",
    "indic_indic": "ai4bharat/indictrans2-indic-indic-dist-200M",
}

try:
    HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 3. SUPPORTED LANGUAGES
# =========================================================

LANGUAGES = {
    "Assamese": "asm_Beng",
    "Bengali": "ben_Beng",
    "Bodo": "brx_Deva",
    "Dogri": "doi_Deva",
    "English": "eng_Latn",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic script)": "kas_Arab",
    "Kashmiri (Devanagari script)": "kas_Deva",
    "Konkani": "gom_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Manipuri (Bengali script)": "mni_Beng",
    "Manipuri (Meitei script)": "mni_Mtei",
    "Marathi": "mar_Deva",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sindhi (Arabic script)": "snd_Arab",
    "Sindhi (Devanagari script)": "snd_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab",
}

CODE_TO_LANGUAGE = {
    code: name for name, code in LANGUAGES.items()
}


# Browser speech recognition codes.
VOICE_LANGUAGES = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Marathi": "mr-IN",
    "Bengali": "bn-IN",
    "Gujarati": "gu-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Punjabi": "pa-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Urdu": "ur-IN",
}


# Languages with gTTS voice support.
TTS_CODES = {
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


# =========================================================
# 4. LANGUAGE DETECTION
# =========================================================

def count_script_characters(text):
    ranges = {
        "Gujarati": (0x0A80, 0x0AFF),
        "Gurmukhi": (0x0A00, 0x0A7F),
        "Bengali": (0x0980, 0x09FF),
        "Odia": (0x0B00, 0x0B7F),
        "Tamil": (0x0B80, 0x0BFF),
        "Telugu": (0x0C00, 0x0C7F),
        "Kannada": (0x0C80, 0x0CFF),
        "Devanagari": (0x0900, 0x097F),
        "Arabic": (0x0600, 0x06FF),
        "Malayalam": (0x0D00, 0x0D7F),
        "Meitei": (0xABC0, 0xABFF),
        "Ol_Chiki": (0x1C50, 0x1C7F),
    }

    counts = Counter()

    for character in text:
        code_point = ord(character)

        for script, (start, end) in ranges.items():
            if start <= code_point <= end:
                counts[script] += 1
                break

    return counts


def detect_source_language(text):
    """
    Detect a likely source language.

    Script detection cannot reliably distinguish every language.
    Shared-script cases are marked uncertain.
    """

    text = text.strip()

    if not text:
        return None, "Unknown", "Enter text first."

    counts = count_script_characters(text)

    # Latin-script input: identify English when possible.
    if not counts:
        try:
            predictions = detect_langs(text)
            top = predictions[0]

            if top.lang == "en" and top.prob >= 0.60:
                return "eng_Latn", "Likely", ""
        except Exception:
            pass

        return (
            None,
            "Unknown",
            "Language could not be detected reliably. "
            "Select the input language manually. "
            "Romanized Indian text may need manual selection.",
        )

    script, _ = counts.most_common(1)[0]

    direct_script_map = {
        "Gujarati": "guj_Gujr",
        "Gurmukhi": "pan_Guru",
        "Odia": "ory_Orya",
        "Tamil": "tam_Taml",
        "Telugu": "tel_Telu",
        "Kannada": "kan_Knda",
        "Malayalam": "mal_Mlym",
        "Meitei": "mni_Mtei",
        "Ol_Chiki": "sat_Olck",
    }

    if script in direct_script_map:
        return direct_script_map[script], "Likely", ""

    if script == "Devanagari":
        try:
            top = detect_langs(text)[0]

            mapping = {
                "hi": "hin_Deva",
                "mr": "mar_Deva",
                "ne": "npi_Deva",
            }

            if top.lang in mapping and top.prob >= 0.75:
                return (
                    mapping[top.lang],
                    "Likely",
                    "Hindi, Marathi and Nepali can share Devanagari. "
                    "Check the detected language for short text.",
                )
        except Exception:
            pass

        return (
            "hin_Deva",
            "Uncertain",
            "This text uses Devanagari, which is shared by Hindi, "
            "Marathi, Nepali and other languages. Hindi is a tentative "
            "guess. Choose the correct language manually if needed.",
        )

    if script == "Bengali":
        try:
            top = detect_langs(text)[0]

            if top.lang == "as" and top.prob >= 0.75:
                return (
                    "asm_Beng",
                    "Likely",
                    "Please verify the detected language.",
                )
        except Exception:
            pass

        return (
            "ben_Beng",
            "Uncertain",
            "Bengali script may represent Bengali, Assamese or "
            "some Manipuri text. Bengali is a tentative guess.",
        )

    if script == "Arabic":
        try:
            top = detect_langs(text)[0]

            if top.lang == "sd" and top.prob >= 0.75:
                return (
                    "snd_Arab",
                    "Likely",
                    "Please verify the detected language.",
                )
        except Exception:
            pass

        return (
            "urd_Arab",
            "Uncertain",
            "Arabic-derived script may represent Urdu or Sindhi. "
            "Urdu is a tentative guess; select Sindhi manually if needed.",
        )

    return (
        None,
        "Unknown",
        "Could not map the detected script to a supported language. "
        "Please select the source language manually.",
    )


# =========================================================
# 5. MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner=False)
def load_translation_model(model_id, token):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=token,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=token,
    )

    model.to(DEVICE)
    model.eval()

    processor = IndicProcessor(inference=True)

    return tokenizer, model, processor


def get_translation_model(model_id):
    active_model = st.session_state.get("_active_model_id")

    # Clear the old cached model when switching directions.
    if active_model != model_id:
        load_translation_model.clear()
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        st.session_state["_active_model_id"] = model_id

    return load_translation_model(model_id, HF_TOKEN)


# =========================================================
# 6. TRANSLATION
# =========================================================

def translate_text(text, source_code, target_code):
    if source_code == target_code:
        raise ValueError(
            "The input and output languages are the same. "
            "Please choose a different output language."
        )

    if source_code == "eng_Latn":
        model_id = MODEL_IDS["en_indic"]

    elif target_code == "eng_Latn":
        model_id = MODEL_IDS["indic_en"]

    else:
        model_id = MODEL_IDS["indic_indic"]

    tokenizer, model, processor = get_translation_model(model_id)

    prepared_text = processor.preprocess_batch(
        [text.strip()],
        src_lang=source_code,
        tgt_lang=target_code,
    )

    inputs = tokenizer(
        prepared_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            use_cache=True,
        )

    decoded = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    results = processor.postprocess_batch(
        decoded,
        lang=target_code,
    )

    return results[0].strip()


# =========================================================
# 7. CALLBACKS
# =========================================================

def clear_input():
    """Clear input and results without directly changing a widget
    after it has been instantiated in the current run.
    """
    st.session_state["source_text"] = ""
    st.session_state["last_spoken_text"] = ""
    st.session_state.pop("translation_result", None)
    st.session_state.pop("translation_audio", None)


def clear_translation():
    st.session_state.pop("translation_result", None)
    st.session_state.pop("translation_audio", None)


# =========================================================
# 8. LANGUAGE SELECTION
# =========================================================

st.markdown(
    '<div class="section-heading">1. Choose your languages</div>',
    unsafe_allow_html=True,
)

language_names = list(LANGUAGES.keys())

source_options = ["Auto-detect"] + language_names

source_choice = st.selectbox(
    "Input language",
    source_options,
    index=0,
    help=(
        "Auto-detect works best with sufficient text in its native "
        "script. For shared scripts or romanized text, select manually."
    ),
)

target_options = [
    name for name in language_names
    if name != "English"
]

target_language = st.selectbox(
    "Translate into",
    target_options,
    index=target_options.index("Hindi"),
)

st.caption(
    "Translation directions are selected automatically: "
    "English → Indian, Indian → English, or Indian → Indian."
)


# =========================================================
# 9. INPUT TEXT AND VOICE
# =========================================================

st.markdown(
    '<div class="section-heading">2. Enter or speak your text</div>',
    unsafe_allow_html=True,
)

voice_language = st.selectbox(
    "Voice recognition language",
    list(VOICE_LANGUAGES.keys()),
    index=0,
    help="Choose the language you will speak into the microphone.",
)

st.caption(
    "Voice recognition depends on browser support and internet access. "
    "Choose the correct spoken language before recording."
)

spoken_text = speech_to_text(
    language=VOICE_LANGUAGES[voice_language],
    start_prompt="🎤 Start speaking",
    stop_prompt="⏹️ Stop recording",
    just_once=True,
    key="languagebridge_voice_input",
)

if "source_text" not in st.session_state:
    st.session_state["source_text"] = ""

if "last_spoken_text" not in st.session_state:
    st.session_state["last_spoken_text"] = ""

if spoken_text and spoken_text != st.session_state["last_spoken_text"]:
    st.session_state["source_text"] = spoken_text
    st.session_state["last_spoken_text"] = spoken_text

input_text = st.text_area(
    "Input text",
    key="source_text",
    height=170,
    max_chars=5000,
    placeholder=(
        "Type or speak in a supported language..."
    ),
)

col_clear, col_detect = st.columns(2)

with col_clear:
    st.button(
        "🧹 Clear text",
        on_click=clear_input,
        use_container_width=True,
    )

with col_detect:
    if st.button(
        "🔎 Detect language",
        use_container_width=True,
    ):
        detected_code, confidence, note = detect_source_language(
            input_text
        )

        if detected_code:
            detected_name = CODE_TO_LANGUAGE.get(
                detected_code,
                detected_code,
            )

            st.info(
                f"Detected language: {detected_name} "
                f"({confidence.lower()} confidence)"
            )

        if note:
            st.warning(note)


# =========================================================
# 10. TRANSLATE BUTTON
# =========================================================

st.markdown(
    '<div class="section-heading">3. Translate</div>',
    unsafe_allow_html=True,
)

if st.button(
    "🌍 Translate text",
    type="primary",
    use_container_width=True,
):
    text_to_translate = input_text.strip()

    if not text_to_translate:
        st.warning("Please enter or speak some text first.")

    else:
        if source_choice == "Auto-detect":
            source_code, confidence, note = detect_source_language(
                text_to_translate
            )
        else:
            source_code = LANGUAGES[source_choice]
            confidence = "Manually selected"
            note = ""

        if not source_code:
            st.error(
                "Could not identify the input language. "
                "Select the source language manually."
            )

        elif source_code not in set(LANGUAGES.values()):
            st.error("This source language is not supported.")

        else:
            target_code = LANGUAGES[target_language]

            if source_code == target_code:
                st.warning(
                    "Input and output languages are the same. "
                    "Please select another target language."
                )

            else:
                try:
                    with st.spinner(
                        "Loading translation model and translating..."
                    ):
                        translated = translate_text(
                            text_to_translate,
                            source_code,
                            target_code,
                        )

                    st.session_state["translation_result"] = {
                        "source_text": text_to_translate,
                        "source_code": source_code,
                        "source_name": CODE_TO_LANGUAGE[source_code],
                        "target_code": target_code,
                        "target_name": target_language,
                        "translated_text": translated,
                        "detection_note": note,
                        "confidence": confidence,
                    }

                    st.session_state.pop("translation_audio", None)

                except Exception as error:
                    st.error(
                        "Translation failed. Check model access, "
                        "the Hugging Face token, package compatibility, "
                        "and the deployment logs."
                    )

                    with st.expander("Technical error details"):
                        st.code(str(error))


# =========================================================
# 11. DISPLAY RESULT AND COPY/DOWNLOAD
# =========================================================

result = st.session_state.get("translation_result")

if result:
    st.markdown(
        '<div class="section-heading">4. Translation result</div>',
        unsafe_allow_html=True,
    )

    st.success(
        f'{result["source_name"]} → {result["target_name"]}'
    )

    if result["detection_note"]:
        st.info(result["detection_note"])

    translated_text = result["translated_text"]

    # Display the output and a browser clipboard button.
    components.html(
        """
        <div style="font-family: sans-serif;">
            <textarea id="translationOutput"
                readonly
                aria-label="Translated text"
                style="
                    width: 100%;
                    min-height: 120px;
                    box-sizing: border-box;
                    padding: 12px;
                    font-size: 16px;
                    border: 1px solid #999;
                    border-radius: 10px;
                    resize: vertical;
                    background: transparent;
                    color: inherit;
                "></textarea>

            <button id="copyTranslation"
                style="
                    margin-top: 8px;
                    padding: 9px 16px;
                    border-radius: 8px;
                    border: 1px solid #888;
                    cursor: pointer;
                    font-size: 14px;
                ">
                📋 Copy translation
            </button>

            <span id="copyStatus"
                style="margin-left: 10px; font-size: 13px;">
            </span>
        </div>

        <script>
            const translation = __TRANSLATION_TEXT__;
            const output = document.getElementById("translationOutput");
            const copyButton = document.getElementById("copyTranslation");
            const status = document.getElementById("copyStatus");

            output.value = translation;

            copyButton.addEventListener("click", async () => {
                try {
                    await navigator.clipboard.writeText(output.value);
                    status.textContent = "Copied!";
                } catch (error) {
                    output.focus();
                    output.select();
                    status.textContent =
                        "Select the text and copy it manually.";
                }
            });
        </script>
        """.replace(
            "__TRANSLATION_TEXT__",
            __import__("json").dumps(translated_text),
        ),
        height=195,
        scrolling=False,
    )

    st.download_button(
        "⬇️ Download translation",
        data=translated_text,
        file_name="translated_text.txt",
        mime="text/plain",
        use_container_width=True,
    )


    # =====================================================
    # 12. TEXT-TO-SPEECH
    # =====================================================

    st.markdown(
        '<div class="section-heading">5. Listen to the translation</div>',
        unsafe_allow_html=True,
    )

    tts_code = TTS_CODES.get(result["target_name"])

    if tts_code:
        try:
            available_tts = gtts_lang.tts_langs()
        except Exception:
            available_tts = {}

        if tts_code in available_tts:

            if st.button(
                "🔊 Generate voice",
                use_container_width=True,
            ):
                try:
                    with st.spinner("Generating audio..."):
                        audio_buffer = BytesIO()

                        tts = gTTS(
                            text=translated_text,
                            lang=tts_code,
                            slow=False,
                        )

                        tts.write_to_fp(audio_buffer)

                        st.session_state["translation_audio"] = (
                            audio_buffer.getvalue()
                        )

                except Exception as error:
                    st.warning(
                        "Could not generate audio. Check your internet "
                        "connection and try again."
                    )

                    with st.expander("Audio error details"):
                        st.code(str(error))

            audio_data = st.session_state.get("translation_audio")

            if audio_data:
                st.audio(audio_data, format="audio/mp3")

        else:
            st.info(
                "The current text-to-speech service does not list "
                "voice support for this language."
            )

    else:
        st.info(
            "Voice output is not configured for this language. "
            "Text translation and download remain available."
        )

    st.button(
        "🗑️ Clear translation",
        on_click=clear_translation,
        use_container_width=True,
    )


# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.caption(
    "Built with Streamlit, PyTorch, Hugging Face Transformers, "
    "IndicTransToolkit and IndicTrans2."
)

st.caption(
    "Automatic language detection can be uncertain for short text, "
    "romanized input, mixed languages, or languages sharing a script. "
    "Review important translations before using them."
)
