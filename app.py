# app.py
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from streamlit.components.v1 import html


from catboost import CatBoostClassifier
st.set_page_config(page_title="Injury Score Only", layout="wide")

# === Top header: title (left) + score (right) ===
hdr_left, hdr_right = st.columns([6, 1])
with hdr_left:
    st.title("🚑 Injury Prediction — Single Score")

with hdr_right:
    score_top = st.empty()      # upper-right metric
    caption_top = st.empty()    # small caption under the metric


# --- Upload trained model ---
with open("catboost_model.pkl", "rb") as model_file:
    # Load the model using pickle
    # Note: Using BytesIO is not necessary here since we are reading directly from a file
    # However, if you want to use BytesIO, you can uncomment the next two lines
    # import io
    # model_bytes = model_file.read()
    # model = pickle.load(io.BytesIO(model_bytes))
    
    model = pickle.load(model_file)

#upload csv
df=pd.read_csv('processed_data.csv',index_col=0).drop(columns='Injury Severity')
len(df.columns)
len(model.feature_names_)
# ---- Feature schema (your list) ----
feature_names = [
 'Agency Name','ACRS Report Type','Route Type','Collision Type','Weather',
 'Surface Condition','Light','Traffic Control','Driver Substance Abuse',
 'Driver At Fault','Driver Distracted By','Drivers License State',
 'Vehicle Damage Extent','Vehicle First Impact Location','Vehicle Body Type',
 'Vehicle Movement','Vehicle Going Dir','Speed Limit','Vehicle Year',
 'Vehicle Make','Crash Hour','Crash Day','Crash Month','h3_index'
]

   

# Features to treat as numeric; others are categorical/text
numeric_features = {'Speed Limit','Vehicle Year','Crash Hour','Crash Day','Crash Month'}
# If your h3_index is numeric, add: numeric_features.add('h3_index')



# ---- Helpers ----
def coerce_value(name: str, val: str):
    """Convert input string to the right type or None."""
    val = val.strip()
    if val == "":
        return None  # let CatBoost handle missing
    if name in numeric_features:
        try:
            # Prefer int when appropriate; else float
            return int(val) if val.isdigit() or (val.startswith('-') and val[1:].isdigit()) else float(val)
        except ValueError:
            return None
    return val  # categorical/text

#add categorical option usig unique values from the dataset
categorical_options = {}
for feat in feature_names:
    if feat not in numeric_features:
        # Get unique values from the dataset for categorical features
        categorical_options[feat] = df[feat].dropna().unique().tolist()

# ---- Input form ----
st.subheader("Enter Feature Values and Click Predict")
with st.form("manual_input"):
    cols = st.columns(3)
    user_row = {}
    for i, feat in enumerate(feature_names):
        with cols[i % 3]:
            if feat in numeric_features:
                # Numeric input
                user_val = st.number_input(feat, step=1)
            elif feat in categorical_options:
                # Dropdown with predefined values
                user_val = st.selectbox(feat, categorical_options[feat])
            else:
                # Default text input for categorical with no predefined list
                user_val = st.text_input(feat, value="", placeholder="type text")
            
            user_row[feat] = coerce_value(feat, str(user_val))

    submitted = st.form_submit_button("Predict")

# Create two columns
col_left, col_right = st.columns([1, 1])

# --- Right column: Score (will be shown after prediction) ---
col_left, col_right = st.columns([1, 1])
with col_left:
    with open("kepler_map.html", "r", encoding="utf-8") as f:
        kepler_html = f.read()
    st.subheader("Kepler Map")
    html(kepler_html, height=600)


# ---- Only run the model after clicking Predict ----

# ---- Only run the model after clicking Predict ----
# ---- Prediction & update the TOP-RIGHT placeholders ----
if submitted:
    X_input = pd.DataFrame([user_row], columns=feature_names)
    try:
        y_prob = float(model.predict_proba(X_input)[:, 1][0])
    except Exception:
        y_prob = float(model.predict(X_input, prediction_type="Probability")[0])
    y_pred = int(y_prob >= 0.25)

    score_top.metric("Predicted probability of injury", f"{y_prob:.4f}")
    caption_top.caption(
        f"Binary prediction at threshold {0.25:.2f}: **{y_pred}** "
        "(0 = No injury, 1 = Injury)"
    )
else:
    st.info("Fill in the fields and click **Predict** to run the model.")