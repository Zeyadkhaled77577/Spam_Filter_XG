# Spam Filtering with XGBoost and Data Combination

## 📖 Overview
This project develops a **Spam Email Classifier** using **XGBoost** and a **data combination method**.  
Multiple publicly available datasets are cleaned, standardized, and merged into one large corpus to train a robust spam filter.  

The model uses **TF-IDF features** and achieves strong performance on spam detection, making it applicable to email, SMS, and messaging systems.  

---

## 🚀 Features
- Automatic dataset loader (supports CSV, Excel, JSON, TXT)
- Data cleaning and preprocessing pipeline
- Combines multiple datasets into one balanced dataset
- Feature extraction using TF-IDF
- Machine Learning with **XGBoost** (and SVM baseline)
- Evaluation with accuracy, precision, recall, F1-score
- Visualization of model performance

---

## 🛠️ Tech Stack
- **Python** (pandas, numpy, re, matplotlib, seaborn)
- **Scikit-learn** (train/test split, TF-IDF, metrics)
- **XGBoost**
- **Seaborn & Matplotlib** (visualizations)

---

## 📂 Project Structure
```
spam-filtering/
   ├── app.py                 # Streamlit GUI app
   ├──spam-filtering.ipynb    # Jupyter Notebook (main workflow)
   ├── requirements.txt       # Dependencies
   ├── README.md              # Documentation
   ├── data/                  # (optional) sample dataset
   └── models/                # trained models (if saved)
```
## ⚡ Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-filtering.git
   cd spam-filtering
   pip install -r requirements.txt
   jupyter notebook spam-filtering.ipynb
   ```
## 🎯 Demo (Streamlit App)

You can run the GUI locally with:

```bash
streamlit run app.py
```
## 📊 Results

XGBoost outperformed baseline models with high accuracy and strong recall in detecting spam.

Combining multiple datasets significantly improved generalization.

## 🔮 Future Work

Deploy as a Streamlit web app for real-time spam detection.

Extend feature extraction with word embeddings (Word2Vec, DISTILBERT).

Use deep learning models (e.g., LSTMs, Transformers).
