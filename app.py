import streamlit as st
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
import numpy as np
import os
import gdown
import joblib

# Gensim ലൈബ്രറികൾ
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class UltimateJobFraudClassifier(nn.Module):
    def __init__(self, vocab_size, glove_dim, lstm_hidden, num_numeric_features, num_classes):
        super(UltimateJobFraudClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.bi_lstm = nn.LSTM(input_size=glove_dim, hidden_size=lstm_hidden, batch_first=True, bidirectional=True)
        self.numeric_branch = nn.Sequential(
            nn.Linear(num_numeric_features, 16),
            nn.ReLU()
        )
        combined_dim = 768 + (lstm_hidden * 2) + 16
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, roberta_ids, roberta_mask, glove_data, numeric_data, remaininput_data):
        roberta_features = self.roberta(input_ids=roberta_ids, attention_mask=roberta_mask).pooler_output
        lstm_out, (hn, _) = self.bi_lstm(glove_data)
        glove_features = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)

        combined_numeric = torch.cat((numeric_data, remaininput_data), dim=1)
        numeric_features = self.numeric_branch(combined_numeric)

        final_combined = torch.cat((roberta_features, glove_features, numeric_features), dim=1)
        return self.classifier(final_combined)


# ==========================================
# 2. CACHED DATA LOADING (Model & Scaler)
# ==========================================
@st.cache_resource
def load_assets(model_id, scaler_path="salary_scaler.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dest = "model.pth"

    # മോഡൽ ഡൗൺലോഡ് ചെയ്യുന്നു
    if not os.path.exists(model_dest):
        gdown.download(f'https://drive.google.com/uc?id={model_id}', model_dest, quiet=False)

    model = UltimateJobFraudClassifier(10000, 100, 128, 5, 2)
    model.load_state_dict(torch.load(model_dest, map_location=device))
    model.to(device).eval()



    scaler = joblib.load(scaler_path)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    return model, tokenizer, scaler, device


# ==========================================
# 3. DYNAMIC GLOVE LOADER (From Google Drive)
# ==========================================
@st.cache_resource
def load_glove_model_from_drive(glove_drive_id):
    raw_glove_file = "glove.6B.100d.txt"
    word2vec_file = "glove.6B.100d.word2vec.txt"

    if not os.path.exists(raw_glove_file):
        url = f'https://drive.google.com/uc?id={glove_drive_id}'
        gdown.download(url, raw_glove_file, quiet=False)

    if not os.path.exists(word2vec_file):
        glove2word2vec(raw_glove_file, word2vec_file)

    return KeyedVectors.load_word2vec_format(word2vec_file)


def get_mean_glove_vector(text, model):
    if not text or model is None:
        return np.zeros(100, dtype=np.float32)
    words = str(text).lower().split()
    vectors = [model[w] for w in words if w in model]

    if not vectors:
        return np.zeros(100, dtype=np.float32)
    return np.mean(vectors, axis=0)


# ==========================================
# 4. MAIN UI SETUP
# ==========================================
def main():
    st.set_page_config(page_title="Job Fraud Detector", layout="wide")
    st.title("🕵️‍♀️ Hybrid Job Fraud Detection")
    st.write("Enter the details of the job posting below to verify its authenticity.")

    GDRIVE_MODEL_ID = '1-Tz7b-F8iR1NvQ1oHtD_rr5zi4G6wJzz'
    GLOVE_DRIVE_ID = '1uFKfamqHLwedQXuH1KTAWtyV87o6TuSj'
    # ---------------------------------------------------------

    # മോഡലും സ്കെയിലറും ലോഡ് ചെയ്യുന്നു
    model, tokenizer, salary_scaler, device = load_assets(GDRIVE_MODEL_ID)

    # ഗ്ലോവ് ഡാറ്റ ലോഡ് ചെയ്യുന്നു (ആപ്പ് തുറക്കുമ്പോൾ ബാക്ക്ഗ്രൗണ്ടിൽ നടന്നോളും)
    with st.spinner("Initializing AI Components... (This may take a minute)"):
        glove_model = load_glove_model_from_drive(GLOVE_DRIVE_ID)


    with st.form("job_form"):
        st.subheader("📝 Job Details")
        job_title = st.text_input("Job Title")
        comp_profile = st.text_area("Company Profile")
        job_desc = st.text_area("Job Description")

        st.markdown("---")
        st.subheader("📊 Additional Details")
        col_g1, col_g2, col_g3 = st.columns(3)
        industry = col_g1.text_input("Industry (e.g., IT, Healthcare)")
        emp_type = col_g2.text_input("Employment Type (e.g., Full-time)")
        experience = col_g3.text_input("Required Experience")

        col_g4, col_g5 = st.columns(2)
        education = col_g4.text_input("Required Education")
        job_function = col_g5.text_input("Function (e.g., Marketing)")

        col1, col2 = st.columns(2)
        min_sal = col1.number_input("Min Salary", value=0)
        max_sal = col2.number_input("Max Salary", value=0)

        col3, col4, col5 = st.columns(3)
        tele = col3.selectbox("Telecommuting (Remote)?", [0, 1])
        logo = col4.selectbox("Has Company Logo?", [0, 1])
        ques = col5.selectbox("Has Screening Questions?", [0, 1])

        submitted = st.form_submit_button("🔍 Check for Fraud")


    if submitted:
        with st.spinner("Analyzing with Hybrid Deep Learning Model..."):


            salaries = np.array([[min_sal, max_sal]])
            scaled_salaries = salary_scaler.transform(salaries)
            numeric_tensor = torch.tensor(scaled_salaries, dtype=torch.float32).to(device)

            remain_tensor = torch.tensor([[tele, logo, ques]], dtype=torch.float32).to(device)


            combined_text = f"{job_title} {comp_profile} {job_desc}"
            encoded = tokenizer(combined_text, padding='max_length', truncation=True, max_length=512,
                                return_tensors='pt')
            ids = encoded['input_ids'].to(device)
            mask = encoded['attention_mask'].to(device)

           
            vec_ind = get_mean_glove_vector(industry, glove_model)
            vec_emp = get_mean_glove_vector(emp_type, glove_model)
            vec_exp = get_mean_glove_vector(experience, glove_model)
            vec_edu = get_mean_glove_vector(education, glove_model)
            vec_func = get_mean_glove_vector(job_function, glove_model)

            glove_array = np.vstack([vec_ind, vec_emp, vec_exp, vec_edu, vec_func])
            glove_tensor = torch.tensor(glove_array, dtype=torch.float32).unsqueeze(0).to(device)


            with torch.no_grad():
                logits = model(ids, mask, glove_tensor, numeric_tensor, remain_tensor)
                prediction = torch.argmax(logits, dim=1).item()
                probs = torch.softmax(logits, dim=1)[0]

            st.markdown("---")
            if prediction == 1:
                st.error(f"🚩 **Result: Potential Fraud Detected!**")
                st.write(f"Model Confidence: {probs[1] * 100:.2f}%")
            else:
                st.success(f"✅ **Result: Looks like a Legitimate Job.**")
                st.write(f"Model Confidence: {probs[0] * 100:.2f}%")


if __name__ == "__main__":
    main()