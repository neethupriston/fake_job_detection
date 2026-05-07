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
    
    if not os.path.exists(model_dest):
        gdown.download(f'https://drive.google.com/uc?id={model_id}', model_dest, quiet=False)
    
    model = UltimateJobFraudClassifier(10000, 100, 128, 5, 2)
    model.load_state_dict(torch.load(model_dest, map_location=device))
    model.to(device).eval()

    if not os.path.exists(scaler_path):
        st.error(f"⚠️ ഒറിജിനൽ സ്കെയിലർ ഫയൽ '{scaler_path}' കണ്ടെത്താനായില്ല! ദയവായി ഈ ഫയൽ ആപ്പിനൊപ്പം അപ്‌ലോഡ് ചെയ്യുക.")
        st.stop()
    else:
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
    
    # ---------------------------------------------------------
    # നിങ്ങളുടെ ഡ്രൈവ് ID-കൾ ഇവിടെ നൽകുക
    # ---------------------------------------------------------
    GDRIVE_MODEL_ID = 'YOUR_TRAINED_MODEL_ID_HERE' 
    GLOVE_DRIVE_ID = 'YOUR_GLOVE_FILE_ID_HERE'
    # ---------------------------------------------------------

    model, tokenizer, salary_scaler, device = load_assets(GDRIVE_MODEL_ID)
    
    with st.spinner("Initializing AI Components... (This may take a minute)"):
        glove_model = load_glove_model_from_drive(GLOVE_DRIVE_ID)

    with st.form("job_form"):
        st.subheader("📝 RoBERTa Text Details")
        col_t1, col_t2, col_t3 = st.columns(3)
        job_title = col_t1.text_input("Job Title")
        department = col_t2.text_input("Department")
        location = col_t3.text_input("Location")
        
        comp_profile = st.text_area("Company Profile")
        job_desc = st.text_area("Job Description")
        requirements = st.text_area("Requirements")
        benefits = st.text_area("Benefits")
        
        st.markdown("---")
        st.subheader("📊 GloVe Categorical Details")
        col_g1, col_g2, col_g3 = st.columns(3)
        industry = col_g1.text_input("Industry (e.g., IT, Healthcare)")
        emp_type = col_g2.text_input("Employment Type (e.g., Full-time)")
        experience = col_g3.text_input("Required Experience")
        
        col_g4, col_g5 = st.columns(2)
        education = col_g4.text_input("Required Education")
        job_function = col_g5.text_input("Function (e.g., Marketing)")
        
        st.markdown("---")
        st.subheader("💰 Numeric & Boolean Details")
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
            
            # 1. Numeric Processing
            salaries = np.array([[min_sal, max_sal]])
            scaled_salaries = salary_scaler.transform(salaries)
            numeric_tensor = torch.tensor(scaled_salaries, dtype=torch.float32).to(device)

            # 2. Boolean Processing
            remain_tensor = torch.tensor([[tele, logo, ques]], dtype=torch.float32).to(device)

            # 3. RoBERTa Text Processing (നിങ്ങൾ ആവശ്യപ്പെട്ട മാറ്റം)
            def encode_text(text):
                # കാലിയായ കോളങ്ങൾ ആണെങ്കിൽ മോഡൽ ക്രാഷ് ആകാതിരിക്കാൻ ഒരു സ്പേസ് നൽകുന്നു
                safe_text = str(text) if text and str(text).strip() != "" else " "
                return tokenizer(
                    safe_text,
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )

            # 7 പാരാമീറ്ററുകളും വെവ്വേറെ എൻകോഡ് ചെയ്യുന്നു
            t_title = encode_text(job_title)
            t_profile = encode_text(comp_profile)
            t_desc = encode_text(job_desc)
            t_req = encode_text(requirements)
            t_ben = encode_text(benefits)
            t_dept = encode_text(department)
            t_loc = encode_text(location)

            # എല്ലാറ്റിനെയും ഒന്നിപ്പിക്കുന്നു (Concatenation) ഒപ്പം നീളം 512 ആയി ചുരുക്കുന്നു
            ids = torch.cat((
                t_title['input_ids'], t_profile['input_ids'],
                t_desc['input_ids'], t_req['input_ids'],
                t_ben['input_ids'], t_dept['input_ids'],
                t_loc['input_ids']
            ), dim=1)[:, :512].to(device)

            mask = torch.cat((
                t_title['attention_mask'], t_profile['attention_mask'],
                t_desc['attention_mask'], t_req['attention_mask'],
                t_ben['attention_mask'], t_dept['attention_mask'],
                t_loc['attention_mask']
            ), dim=1)[:, :512].to(device)

            # 4. GloVe Processing
            vec_ind = get_mean_glove_vector(industry, glove_model)
            vec_emp = get_mean_glove_vector(emp_type, glove_model)
            vec_exp = get_mean_glove_vector(experience, glove_model)
            vec_edu = get_mean_glove_vector(education, glove_model)
            vec_func = get_mean_glove_vector(job_function, glove_model)
            
            glove_array = np.vstack([vec_ind, vec_emp, vec_exp, vec_edu, vec_func])
            glove_tensor = torch.tensor(glove_array, dtype=torch.float32).unsqueeze(0).to(device)

            # 5. Prediction
            with torch.no_grad():
                logits = model(ids, mask, glove_tensor, numeric_tensor, remain_tensor)
                prediction = torch.argmax(logits, dim=1).item()
                probs = torch.softmax(logits, dim=1)[0]

            st.markdown("---")
            if prediction == 1:
                st.error(f"🚩 **Result: Potential Fraud Detected!**")
                st.write(f"Model Confidence: {probs[1]*100:.2f}%")
            else:
                st.success(f"✅ **Result: Looks like a Legitimate Job.**")
                st.write(f"Model Confidence: {probs[0]*100:.2f}%")

if __name__ == "__main__":
    main()
