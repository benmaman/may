# app.py
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from streamlit.components.v1 import html
from streamlit import markdown


from catboost import CatBoostClassifier
st.set_page_config(page_title="Injury Score Only", layout="wide")

# === Top header: title (left) + score (right) ===
hdr_left, hdr_right = st.columns([6, 1])
with hdr_left:
    st.title("🚑 Injury Prediction")

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
    cols = st.columns(5)
    user_row = {}
    for i, feat in enumerate(feature_names):
        with cols[i % 5]:
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
    html(kepler_html, height=500)

        # --- Feature importance ---
    st.subheader("Feature Importance")

    # Use CatBoost importances; align with your feature_names
    try:
        importances = model.get_feature_importance()
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    except Exception:
        fi_pretty = model.get_feature_importance(prettified=True)
        col_feat = [c for c in fi_pretty.columns if "Feature" in c][0]
        col_imp  = [c for c in fi_pretty.columns if "Import" in c][0]
        fi_df = fi_pretty.rename(columns={col_feat: "Feature", col_imp: "Importance"})[
            ["Feature", "Importance"]
        ]

    fi_df = fi_df.sort_values("Importance", ascending=False)

    # Optional: let the user choose Top N
    top_n = st.slider("Show top features", 5, min(25, len(fi_df)), 15)
    st.bar_chart(
        fi_df.head(top_n).set_index("Feature")["Importance"],
        use_container_width=True
    )


# ---- Only run the model after clicking Predict ----
def probability_to_color(prob):
    """
    Maps probability [0, 1] to a softer gradient:
    Green (low risk) → Orange (medium) → Red (high risk),
    avoiding overly saturated 'marker' colors.
    """
    # Clamp prob to [0,1]
    prob = max(0.0, min(1.0, prob))

    # Minimum and maximum intensity (avoid pure colors)
    min_val = 100   # darkest channel
    max_val = 220   # lightest channel

    if prob <= 0.5:
        # Green (low) → Orange (mid)
        r = int(min_val + (max_val - min_val) * (prob / 0.5))  # grows from 100 to 220
        g = max_val
    else:
        # Orange (mid) → Red (high)
        r = max_val
        g = int(max_val - (max_val - min_val) * ((prob - 0.5) / 0.5))  # drops from 220 to 100

    b = min_val  # keep blue low for warmer tones
    return f"rgb({r},{g},{b})"


# ---- Only run the model after clicking Predict ----
# ---- Prediction & update the TOP-RIGHT placeholders ----
if submitted:
    X_input = pd.DataFrame([user_row], columns=feature_names)
    try:
        y_prob = float(model.predict_proba(X_input)[:, 1][0])
    except Exception:
        y_prob = float(model.predict(X_input, prediction_type="Probability")[0])
    y_pred = int(y_prob >= 0.25)
    
    bg_color = probability_to_color(y_prob)
    label = "INJURY" if y_pred == 1 else "SAFE"
    icon = "⚠️" if y_pred == 1 else "✅"

    card_html = f"""
    <div style="
        background-color:{bg_color};
        width:250px;
        height:250px;
        border-radius:10px;
        padding:20px;
        text-align:center;
        color:black;
        font-family:sans-serif;">
        <div style="font-size:40px;">{icon}</div>
        <div style="font-size:28px; font-weight:bold;">{label}</div>
        <div style="font-size:20px;">
            The model estimates a <b>{y_prob*100:.0f}%</b> probability of injury for this accident.
        </div>
    </div>
    """
    
    with col_right:
        st.subheader("Prediction Result")
        st.markdown(card_html, unsafe_allow_html=True)
    
else:
    st.info("Fill in the fields and click **Predict** to run the model.")