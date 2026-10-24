
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit import IndicProcessor

st.set_page_config(
    page_title="BhashaBridge | Indian Language Translator",
    page_icon="🌐",
    layout="centered",
)

INDIC_MODEL_ID = "ai4bharat/indictrans2-en-indic-dist-200M"
YOUR_MODEL_ID = "samiksha7738/english-hindi-translator"

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

st.title("🌐 BhashaBridge")
st.subheader("English to Indian Language Translator")
st.write(
    "Translate English sentences into 12 Indian languages "
    "using neural machine translation."
)

model_choice = st.radio(
    "Choose translation model",
    [
        "IndicTrans2 — 12 Indian languages",
        "My fine-tuned English-to-Hindi model",
    ],
)

@st.cache_resource(show_spinner=False)
def load_indic_model():
    tokenizer = AutoTokenizer.from_pretrained(
        INDIC_MODEL_ID,
        trust_remote_code=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        INDIC_MODEL_ID,
        trust_remote_code=True,
    )
    model.eval()
    processor = IndicProcessor(inference=True)
    return tokenizer, model, processor

@st.cache_resource(show_spinner=False)
def load_hindi_model():
    tokenizer = AutoTokenizer.from_pretrained(YOUR_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(YOUR_MODEL_ID)
    model.eval()
    return tokenizer, model

target_language = "Hindi"

if model_choice.startswith("IndicTrans2"):
    target_language = st.selectbox(
        "Translate into",
        list(LANGUAGES.keys()),
    )
else:
    st.info(
        "This option uses your own fine-tuned model "
        "and supports English-to-Hindi translation only."
    )

english_text = st.text_area(
    "Enter English text",
    placeholder="Example: India is a beautiful country.",
    height=140,
    max_chars=1000,
)

if st.button("Translate", type="primary", use_container_width=True):
    if not english_text.strip():
        st.warning("Please enter some English text.")
    else:
        try:
            if model_choice.startswith("IndicTrans2"):
                with st.spinner("Loading multilingual model..."):
                    tokenizer, model, processor = load_indic_model()

                target_code = LANGUAGES[target_language]
                processed = processor.preprocess_batch(
                    [english_text.strip()],
                    src_lang="eng_Latn",
                    tgt_lang=target_code,
                )

                inputs = tokenizer(
                    processed,
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
                        num_beams=5,
                        num_return_sequences=1,
                    )

                decoded = tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                translated = processor.postprocess_batch(
                    decoded,
                    lang=target_code,
                )[0]

            else:
                with st.spinner("Loading your fine-tuned model..."):
                    tokenizer, model = load_hindi_model()

                inputs = tokenizer(
                    english_text.strip(),
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                )

                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=4,
                    )

                translated = tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                )

            st.subheader(f"{target_language} Translation")
            st.text_area(
                "Translation result",
                value=translated,
                height=140,
            )

        except Exception as error:
            st.error("Translation failed. Check the deployment logs.")
            st.caption(f"Error: {error}")

with st.expander("About this project"):
    st.write("""
    Project: English-to-Indian Language Neural Machine Translation

    Multilingual model: AI4Bharat IndicTrans2

    Fine-tuned model: Your English-to-Hindi MarianMT model

    Framework: Python, PyTorch, Hugging Face Transformers,
    IndicTransToolkit and Streamlit.

    Your earlier BLEU score (10.97) and test loss (3.5023)
    belong to the evaluation of your fine-tuned English-to-Hindi
    model, not to the IndicTrans2 model.
    """)

