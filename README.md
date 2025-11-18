📝 Transcript Summary Generator

A Streamlit-based web application that allows you to upload a text transcript and automatically generate a detailed summary using a quantized LLM (Zephyr-7B-Beta with 4-bit quantization).

This tool is useful for summarizing:

Meetings

Interviews

Lectures

YouTube transcripts

Long documents

🚀 Features

📄 Upload .txt transcript files

🤖 LLM-based summary generation

⚡ 4-bit quantized model for faster inference

💾 Download summary as a .txt file

🔒 Cached model + prompt loading for performance

🌐 Fully Streamlit-based UI

📂 Project Structure
transcript-summary-generator/
│── app.py                 # Main Streamlit application
│── prompt.md              # System + user prompt used for summary generation
│── README.md              # Documentation
│── requirements.txt       # Python dependencies

🧰 Requirements

requirements.txt:

streamlit
torch
transformers
accelerate
bitsandbytes
sentencepiece
protobuf
typing_extensions


⚠️ Note: Install GPU version of PyTorch manually if you want faster inference:
For CUDA 12.x:
pip install torch --index-url https://download.pytorch.org/whl/cu121

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2️⃣ Create a Virtual Environment (Optional)
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Running the App
streamlit run app.py


Open your browser at: http://localhost:8501

🧠 Model Details

Model: HuggingFaceH4/zephyr-7b-beta

Quantization: 4-bit (nf4) using BitsAndBytes

Device mapping handled automatically using accelerate

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


The model and tokenizer are cached using Streamlit’s @st.cache_resource for performance.

📤 Uploading a Transcript

Accepts .txt files

After uploading, click Generate Summary to generate output

💾 Downloading Output

The generated summary can be downloaded as:

summary_YYYYMMDD_HHMMSS.txt

🧑‍💻 GitHub Commands
Clone the Repo
git clone https://github.com/<username>/<repo-name>.git

Add Files
git add .
# or
git add app.py

Commit Changes
git commit -m "Added Streamlit summary generator"

Push to GitHub
git push origin main
# or master if your branch is master

Pull Latest Updates
git pull

Branch Management
# Create a new branch
git checkout -b feature/update-ui

# Switch branches
git checkout main
