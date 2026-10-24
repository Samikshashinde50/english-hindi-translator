
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Hugging Face model repository
MODEL_ID = "samiksha7738/english-hindi-translator"

st.set_page_config(
    page_title="English to Hindi Translator",
    page_icon="🌐",
    layout="centered"
)

@st.cache_resource
def load_translation_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model

st.title("🌐 English to Hindi Translator")
st.write(
    "Translate English sentences into Hindi using "
    "Neural Machine Translation."
)

st.info(
    "Powered by your fine-tuned MarianMT model "
    "and Hugging Face Transformers."
)

try:
    with st.spinner("Loading translation model..."):
        tokenizer, model = load_translation_model()

    st.success("Translation model is ready!")

    english_text = st.text_area(
        "Enter an English sentence:",
        placeholder="Example: India is a beautiful country.",
        height=130,
        max_chars=1000
    )

    if st.button("Translate to Hindi", type="primary",
                 use_container_width=True):
        if not english_text.strip():
            st.warning("Please enter an English sentence.")
        else:
            with st.spinner("Translating..."):
                inputs = tokenizer(
                    english_text.strip(),
                    return_tensors="pt",
                    truncation=True,
                    max_length=128
                )

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=4
                    )

                hindi_text = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

            st.subheader("Hindi Translation")
            st.text_area(
                "Translation result:",
                value=hindi_text,
                height=130
            )

            st.caption(
                "Review the translation for context and accuracy."
            )

except Exception as error:
    st.error(
        "The model could not be loaded. "
        "Please check the deployment logs."
    )
    st.caption(f"Error details: {error}")

with st.expander("About this project"):
    st.write("""
    **Project:** English-to-Hindi Neural Machine Translation

    **Dataset:** IIT Bombay English-Hindi parallel corpus

    **Model:** Fine-tuned Helsinki-NLP MarianMT

    **Framework:** Python and Hugging Face Transformers

    **Evaluation on the current test set:**
    - BLEU Score: 10.97
    - Test Loss: 3.5023
    """)
