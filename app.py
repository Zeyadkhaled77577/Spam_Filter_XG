import streamlit as st
import joblib
import pandas as pd

# =============================
# Load model & vectorizer (use your existing paths)
# =============================
xgb_model = joblib.load("C://Users//user//Downloads//Spam_Filtering//xgb_model.pkl")
vectorizer = joblib.load("C://Users//user//Downloads//Spam_Filtering//tfidf_vectorizer.pkl")

# Page config
st.set_page_config(page_title="📩 Spam Filter App", layout="wide")
st.title("📩 Unified Spam Filter App")
st.write("Paste one message or many messages (one per line) OR upload a CSV with a 'text' column. Then press **Predict**.")

# Sidebar: threshold
threshold = st.sidebar.slider("Set Spam Threshold", 0.0, 1.0, 0.5, 0.01)

# Inputs (single textarea + CSV uploader)
user_input = st.text_area("Enter message(s) (one per line for multiple):",
                          placeholder="Example:\nHello, are we meeting today?\nCongratulations! You've won a prize...")
uploaded_file = st.file_uploader("Or upload CSV with a 'text' column", type=["csv"])

# Predict button
if st.button("Predict", key="predict_btn"):
    # 1) Priority: CSV if uploaded
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Can't read uploaded CSV: {e}")
        else:
            # try to find text column
            if "text" in df.columns:
                messages = df["text"].astype(str).tolist()
            else:
                # fallback: take first column and warn user
                first_col = df.columns[0]
                st.warning(f"CSV does not have a 'text' column — using first column '{first_col}' as text.")
                messages = df[first_col].astype(str).tolist()

            if len(messages) == 0:
                st.warning("Uploaded CSV contains no messages.")
            else:
                # vectorize & predict
                features = vectorizer.transform(messages)
                probs = xgb_model.predict_proba(features)[:, 1]
                preds = ["Spam" if p >= threshold else "Ham" for p in probs]

                result_df = pd.DataFrame({
                    "Message": messages,
                    "Prediction": preds,
                    "Spam_Score": probs
                })
                st.success(f"Classified {len(messages)} messages from CSV.")
                st.dataframe(result_df, use_container_width=True)

                # download
                st.download_button(
                    "Download Predictions (CSV)",
                    result_df.to_csv(index=False).encode("utf-8"),
                    "predictions_from_csv.csv",
                    "text/csv"
                )

    # 2) Else: use textarea if not empty
    elif user_input.strip():
        # split lines so each line becomes one message
        messages = [line.strip() for line in user_input.splitlines() if line.strip()]

        if len(messages) == 0:
            st.warning("No valid messages detected in the textarea.")
        else:
            features = vectorizer.transform(messages)
            probs = xgb_model.predict_proba(features)[:, 1]
            preds = ["Spam" if p >= threshold else "Ham" for p in probs]

            result_df = pd.DataFrame({
                "Message": messages,
                "Prediction": preds,
                "Spam_Score": probs
            })

            # If single message -> show metric + progress
            if len(messages) == 1:
                prob = float(probs[0])
                label = "🚨 Spam" if prob >= threshold else "✅ Ham"
                st.metric(label="Prediction", value=label)
                st.progress(prob)  # shows a progress bar for the single message
                st.write(f"**Spam Score:** {prob:.2f}")
                # also show a small table and allow download
                st.dataframe(result_df, use_container_width=True)
                st.download_button(
                    "Download Prediction (CSV)",
                    result_df.to_csv(index=False).encode("utf-8"),
                    "single_prediction.csv",
                    "text/csv"
                )
            else:
                # multiple messages -> show table + download
                st.success(f"Classified {len(messages)} messages.")
                st.dataframe(result_df, use_container_width=True)
                st.download_button(
                    "Download Predictions (CSV)",
                    result_df.to_csv(index=False).encode("utf-8"),
                    "predictions_from_textarea.csv",
                    "text/csv"
                )

    else:
        st.warning("Please paste at least one message in the textarea or upload a CSV file.")
