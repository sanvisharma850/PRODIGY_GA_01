import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import datetime

@st.cache_resource
def load_model():
    tokenizer= GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()
st.title("GPT-2 Text Generator")
prompt = st.text_input("Enter a prompt", value = "Once upon a time")
max_length = st.slider("Select output length", min_value=50, max_value=300, value = 100, step = 10)

if st.button("Generate Text"):
     inputs = tokenizer.encode(prompt, return_tensors="pt")
     outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
     generated = tokenizer.decode(outputs[0], skip_special_tokens = True)

     st.session_state["generated_text"] = generated

     st.subheader("Generated Text")
     st.write(generated)

if "generated_text" in st.session_state:
     if st.checkbox("Save this text"):
         filename = f"generated_text{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
         with open(filename, "w", encoding="utf-8") as f:
             f.write(st.session_state["generated_text"])
         st.success(f"Text saved as {filename}")


