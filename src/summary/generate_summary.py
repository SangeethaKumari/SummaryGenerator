import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime

st.set_page_config(page_title="Transcript Summary Generator", page_icon="📝", layout="centered")

st.title("📝 Transcript Summary Generator")
st.write("Upload a transcript and generate a detailed summary with key points")

# --- Function to load the prompt (cached) ---
@st.cache_resource
def load_prompt():
    try:
        with open("prompt.md", "r") as f:
            return f.read()
    except FileNotFoundError:
        st.error("❌ prompt.md file not found!")
        return None

prompt_content = load_prompt()

# Model selection
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta" 

# Define quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# --- Function to load the model (cached) ---
# This function runs only ONCE when the app starts.
@st.cache_resource
def get_model_and_tokenizer(model_name, q_config):
    st.info(f"📥 Loading {model_name} model with 4-bit quantization... (This happens only once)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=q_config,
        device_map="auto", # Automatically maps model to available GPUs
    )
    
    # Check device placement after auto mapping
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"✅ Model loaded and ready on {device.upper()}.")
    

    return model, tokenizer, device

   

# Call the cached function once at the start of the script
model, tokenizer, device = get_model_and_tokenizer(MODEL_NAME, quantization_config)


# File upload remains the same...
uploaded_file = st.file_uploader("Upload transcript (.txt file)", type=["txt"])

if uploaded_file:
    transcript_text = uploaded_file.read().decode("utf-8")
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    st.info(f"📄 Transcript length: {len(transcript_text)} characters")


# --- The generation block is now cleaner ---
if st.button("🔄 Generate Summary", type="primary", use_container_width=True):
    if not prompt_content:
        st.error("❌ Could not load prompt file")
    elif not uploaded_file:
        st.error("❌ Please upload a transcript file first")
    else:
        # The model is already loaded and cached from the start
        with st.spinner("⏳ Generating summary..."):
            try:
                # Parse and format prompt
                parts = prompt_content.split("### User Prompt")
                system_prompt = parts[0].replace("### System Prompt", "").strip()
                user_prompt = parts[1].strip() if len(parts) > 1 else ""
                user_prompt = user_prompt.format(transcript=transcript_text[:4000])
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                st.info(f"⚙️ Generating summary on {device.upper()}...")
                
                # Prepare input
                messages = [{"role": "user", "content": full_prompt}]
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(device)
                
                # Generate
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1500,
                    temperature=0.7,
                    top_p=0.9
                )
                
                # Decode
                summary = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True
                )
                
                # Display summary and download button
                st.success("✅ Summary Generated!")
                st.divider()
                st.markdown(summary)
                st.divider()
                
                st.download_button(
                    label="📥 Download Summary",
                    data=summary,
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Ensure you have enough VRAM for the quantized model.")

