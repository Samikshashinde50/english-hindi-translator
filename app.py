
import os
from io import BytesIO

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS, lang as gtts_lang


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Indian Language Translator",
    page_icon="🌐",
    layout="centered"
)

st.title("🌐 English to Indian Languages Translator")
st.write(
    "Type your text or use your voice to translate "
    "English into Indian languages."
)

# Read Hugging Face token from Streamlit Secrets
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")

# Model IDs
INDIC_MODEL_ID = "ai4bharat/indictrans2-en-indic-dist-200M"
YOUR_MODEL_ID = "samiksha7738/english-hindi-translator"

# Target languages
LANGUAGES = {
    "Hindi": "hin_Deva",
    "Marathi": "mar_Deva",
    "Bengali": "ben_Beng",
    "Gujarati": "guj_Gujr",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Punjabi": "pan_Guru",
    "Odia": "ory_Orya",
    "Assamese": "asm_Beng",
    "Urdu": "urd_Arab",
}

# gTTS language codes (only available languages will be offered)
TTS_CODES = {
    "Hindi": "hi",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Odia": "or",
    "Assamese": "as",
    "Urdu": "ur",
}


# ---------------- LOAD INDIC MODEL ----------------

@st.cache_resource(show_spinner="Loading IndicTrans2 model...")
def load_indic_model():
    tokenizer = AutoTokenizer.from_pretrained(
        INDIC_MODEL_ID,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        INDIC_MODEL_ID,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    model.eval()
    processor = IndicProcessor(inference=True)

    return tokenizer, model, processor


# ---------------- LOAD YOUR HINDI MODEL ----------------

@st.cache_resource(show_spinner="Loading your Hindi model...")
def load_hindi_model():
    tokenizer = AutoTokenizer.from_pretrained(
        YOUR_MODEL_ID,
        token=HF_TOKEN,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        YOUR_MODEL_ID,
        token=HF_TOKEN,
    )

    model.eval()
    return tokenizer, model


# ---------------- TRANSLATION FUNCTIONS ----------------

def translate_with_indic(text, target_code):
    tokenizer, model, processor = load_indic_model()

    prepared_text = processor.preprocess_batch(
        [text.strip()],
        src_lang="eng_Latn",
        tgt_lang=target_code,
    )

    inputs = tokenizer(
        prepared_text,
        truncation=True,
        padding="longest",
        max_length=256,
        return_tensors="pt",
        return_attention_mask=True,
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=4,
            num_return_sequences=1,
        )

    decoded = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    result = processor.postprocess_batch(
        decoded,
        lang=target_code,
    )

    return result[0]


def translate_with_my_model(text):
    tokenizer, model = load_hindi_model()

    inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
        )

    result = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return result


# ---------------- SELECT TRANSLATION MODE ----------------

mode = st.radio(
    "Choose translation model",
    [
        "IndicTrans2 — 12 Indian languages",
        "My fine-tuned English-to-Hindi model",
    ],
)

if mode == "IndicTrans2 — 12 Indian languages":
    selected_language = st.selectbox(
        "Translate English into",
        list(LANGUAGES.keys()),
    )
else:
    selected_language = "Hindi"
    st.info("Your fine-tuned model translates English into Hindi.")


# ---------------- VOICE INPUT ----------------

st.subheader("🎤 Enter or speak English")

st.caption(
    "Click Start speaking, say your sentence in English, "
    "and then stop recording."
)

spoken_text = speech_to_text(
    language="en",
    start_prompt="🎤 Start speaking",
    stop_prompt="⏹️ Stop recording",
    just_once=True,
    key="english_voice_input",
)

# Put recognized speech into the text box before rendering it
if spoken_text:
    st.session_state["source_text"] = spoken_text

if "source_text" not in st.session_state:
    st.session_state["source_text"] = ""

if "translated_text" not in st.session_state:
    st.session_state["translated_text"] = ""

english_text = st.text_area(
    "English text",
    key="source_text",
    height=150,
    placeholder="Type or speak your English sentence here...",
)

# ---------------- TRANSLATE BUTTON ----------------

if st.button("🌍 Translate", type="primary", use_container_width=True):
    if not english_text.strip():
        st.warning("Please enter or speak some English text.")
    else:
        try:
            with st.spinner("Translating..."):
                if mode == "IndicTrans2 — 12 Indian languages":
                    translated = translate_with_indic(
                        english_text,
                        LANGUAGES[selected_language],
                    )
                else:
                    translated = translate_with_my_model(english_text)

                st.session_state["translated_text"] = translated

        except Exception as error:
            st.error(
                "Translation failed. Check your model access, "
                "Hugging Face token, and Streamlit logs."
            )
            st.caption(f"Error details: {error}")


# ---------------- DISPLAY TRANSLATION ----------------

translated_text = st.session_state.get("translated_text", "")

if translated_text:
    st.subheader(f"✅ Translation — {selected_language}")

    st.text_area(
        "Translated text",
        value=translated_text,
        height=150,
        disabled=True,
        key="translation_output",
    )

    st.download_button(
        "⬇️ Download translation",
        data=translated_text,
        file_name="translation.txt",
        mime="text/plain",
    )

    # ---------------- VOICE OUTPUT ----------------

    st.subheader("🔊 Listen to the translation")

    available_tts = gtts_lang.tts_langs()
    tts_code = TTS_CODES.get(selected_language)

    if tts_code and tts_code in available_tts:
        if st.button("🔊 Generate voice"):
            try:
                with st.spinner("Generating audio..."):
                    audio_buffer = BytesIO()

                    tts = gTTS(
                        text=translated_text,
                        lang=tts_code,
                        slow=False,
                    )

                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)

                    st.audio(
                        audio_buffer.getvalue(),
                        format="audio/mp3",
                    )

            except Exception:
                st.warning(
                    "Could not generate audio. Check your internet "
                    "connection or try another language."
                )
    else:
        st.info(
            "Voice output is not available for this language "
            "through the current text-to-speech service."
        )

st.divider()
st.caption("Built with Streamlit, Hugging Face Transformers, and IndicTrans2.")


