Fake Job Posting Detection
Overview
This project involves a deep learning pipeline designed to classify job postings as either legitimate or fraudulent. The architecture employs a hybrid approach, combining large language models for text processing, word embeddings for categorical data, and dense neural networks for numerical data to accurately detect fake listings.  
Dataset
Source: The model is trained on the fake_job_postings.csv dataset, which initially consists of 17,880 rows and 18 columns.  
Target Variable: The fraudulent column serves as the target, where 0 indicates a real job and 1 indicates a fake job. 
Class Imbalance: The dataset is heavily imbalanced, featuring 16,686 legitimate postings and only 847 fraudulent postings.  
Data Cleaning & Preprocessing
Missing Values: Empty fields in categorical and textual columns (like employment_type, industry, and company_profile)
Salary Processing: The salary_range column is cleaned of date-like formatting errors, split into individual min_salary and max_salary columns, converted to numeric values, and normalized using a RobustScaler.
Feature Engineering
The pipeline processes different types of data through three distinct branches before combining them:
Text Features (RoBERTa): Long-form text features—such as title, company_profile, description, requirements, benefits, department, and location—are tokenized using Hugging Face's roberta-base tokenizer, capped at a maximum length of 512 tokens.
Categorical Features (GloVe + Bi-LSTM): Categorical variables (e.g., industry, employment_type, required_experience, required_education, and function) are converted into 100-dimensional word vectors using Stanford's pre-trained glove.6B.100d.txt embeddings. These vectors are then processed through a Bidirectional LSTM layer.
Numeric & Boolean Features: Scaled salary data, along with binary flags like telecommuting, has_company_logo, and has_questions, are concatenated and passed through a dedicated numeric feed-forward neural network branch.
Model Architecture & Training
Architecture: The UltimateJobFraudClassifier is a custom PyTorch model that concatenates the outputs from the RoBERTa text features, the GloVe-fed Bi-LSTM, and the numeric network. This combined vector is passed through a final classification head utilizing linear layers, ReLU activation, and a 40% dropout rate.
Training Setup: The model is trained using a CUDA-enabled GPU with the AdamW optimizer (learning rate: 2e-5). It uses mixed-precision training (torch.cuda.amp.autocast()) for efficiency.
Handling Imbalance: To account for the heavy bias toward real jobs, the training loop utilizes a weighted CrossEntropyLoss function with class weights set to [5.0, 15.0].
Epochs & Batching: The model is trained over 5 epochs with a batch size of 64.
Performance Metrics
Following the 5 epochs of training, the model yielded the following test results:
Overall Accuracy: 97.86%
Fake Job Detection (Class 1): The model achieved a Precision of 0.77, a Recall of 0.82, and an F1-Score of 0.79.  
Legitimate Job Detection (Class 0): The model achieved an F1-Score of 0.99.
Saved Output Files
Upon successful training, the script generates the following artifacts:
hybrid_4_fake_job_model.pth: The saved PyTorch state dictionary containing the trained model weights.
salary_scaler.pkl: The saved RobustScaler object to ensure consistent transformation of salary data during future inferences.





