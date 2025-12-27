
import os
import re
import random
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px


import os
import re
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.impute import SimpleImputer


# ---------------------
# AQ-10 Questions
# ---------------------
AQ_questions = {
    1: "I often notice small sounds when others do not.(YES->1,NO->0)",
    2: "I usually concentrate more on the whole picture, rather than the small details.(YES->1,NO->0)",
    3: "I find it easy to do more than one thing at once.(YES->1,NO->0)",
    4: "If there is an interruption, I can switch back to what I was doing very quickly.(YES->1,NO->0)",
    5: "I find it easy to ‘read between the lines’ when someone is talking to me.(YES->1,NO->0)",
    6: "I know how to tell if someone listening to me is getting bored.(YES->1,NO->0)",
    7: "When I’m reading a story, I find it difficult to work out the characters’ intentions.(YES->1,NO->0)",
    8: "I like to collect information about categories of things (e.g. types of car, types of bird, types of train, types of plant etc.).(YES->1,NO->0)",
    9: "I find it easy to work out what someone is thinking or feeling just by looking at their face.(YES->1,NO->0)",
    10: "I find it difficult to work out people’s intentions.(YES->1,NO->0)"
}

def load_model():
    models = {
        'Logistic Regression': joblib.load('logistic_regression_model.joblib'),
        'XGBoost': joblib.load('xgboost_model.joblib'),
        'SVM': joblib.load('svm_model.joblib')
    }
    feature_columns = joblib.load('feature_columns.joblib')
    scaler = joblib.load('feature_scaler.joblib')

    if os.path.exists('label_encoders.joblib'):
        encoders = joblib.load('label_encoders.joblib')
    else:
        st.warning("⚠️ No label_encoders.joblib found. Categorical mappings may be approximate.")
        encoders = {}

    return models, feature_columns, scaler, encoders

# ---------------------
# Preprocess single input (match train.py exactly)
# ---------------------
def preprocess_input(input_data, feature_columns, scaler, encoders):
    """
    input_data: dict of raw values (strings from UI)
    feature_columns: list in exact order used for training
    encoders: dict mapping column -> LabelEncoder (fitted during training)
    Returns: (scaled_array, prepared_df)
    """

    # 1) Apply same replacements as training: yes->1, no->0, ?/others -> 'Others'
    prepared = {}
    for k, v in input_data.items():
        if isinstance(v, str):
            vl = v.strip().lower()
            if vl == 'yes':
                prepared[k] = 1
                continue
            elif vl == 'no':
                prepared[k] = 0
                continue
            elif vl in ['?', 'others', 'other']:
                prepared[k] = 'Others'
                continue
        # keep as-is (numbers, already converted values, etc.)
        prepared[k] = v

    # 2) Build row consistent with feature_columns
    row = {}
    for col in feature_columns:
        # A-score columns
        if col.startswith('A') and col.endswith('_Score'):
            row[col] = int(prepared.get(col, 0))
        elif col == 'age':
            # training: df['age'] = np.log(df['age'])
            # ensure numeric
            age_val = float(prepared.get('age', 25.0))
            # safety: avoid log(0) if someone ever passes 0 -> add tiny epsilon
            age_val = max(age_val, 1e-6)
            row[col] = np.log(age_val)
        elif col == 'result':
            row[col] = float(prepared.get('result', 0.0))
        elif col in ['sum_score', 'ind']:
            row[col] = int(prepared.get(col, 0))
        else:
            # If this column was encoded during training, keep the raw string
            # (we will transform using encoders). Otherwise treat as numeric if possible.
            if col in encoders:
                # use string form so LabelEncoder can map it (e.g., 'gender','ageGroup','ethnicity' etc.)
                row[col] = str(prepared.get(col, 'unknown'))
            else:
                # numeric fallback
                val = prepared.get(col, 0)
                try:
                    row[col] = float(val)
                except Exception:
                    # if non-numeric and encoder missing, convert common yes/no to 0/1 above,
                    # else map to 0
                    row[col] = 0.0

    # 3) DataFrame in correct column order
    full_df = pd.DataFrame([row], columns=feature_columns)

    # 4) Apply encoders for categorical features (use safe fallback for unseen categories)
    for col in list(full_df.select_dtypes(include=['object']).columns):
        if col in encoders:
            le = encoders[col]
            val = full_df.at[0, col]
            # If exact string is among classes, transform; else fallback to first class
            if val in le.classes_:
                mapped = le.transform([val])[0]
            else:
                # fallback: choose first class (stable) — you can change to mode if you saved it
                mapped = le.transform([le.classes_[0]])[0]
            full_df[col] = mapped
        else:
            # No encoder available: try to coerce common patterns
            # if it's numeric string -> convert; else assign 0
            try:
                full_df[col] = full_df[col].astype(float)
            except Exception:
                full_df[col] = 0.0

    # 5) Now full_df should be numeric and in the exact order of feature_columns
    #    (Double-check shape & columns)
    # 6) Scale using saved scaler (must match training)
    try:
        scaled = scaler.transform(full_df)
    except Exception as e:
        # very defensive: if scaler fails, show debug
        st.error("Scaler.transform failed: " + str(e))
        st.write("Prepared DataFrame (before scaling):")
        st.dataframe(full_df)
        raise

    return scaled, full_df


# ---------------------
# Helpers to sample dataset rows (quick-load)
# ---------------------
def find_class_column(df):
    candidates = ['Class/ASD', 'Class ASD', 'class', 'ASD', 'Class']
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        uniques = set(df[c].dropna().astype(str).str.lower().unique())
        if uniques.issubset({'yes', 'no'}) or uniques.issubset({'0', '1'}) or len(uniques) == 2:
            return c
    return None


def map_row_to_input(row):
    mapping = {}
    # Age
    for cand in ['age', 'Age']:
        if cand in row.index:
            mapping['age'] = row[cand]
            break
    # Gender
    for cand in ['gender', 'sex']:
        if cand in row.index:
            mapping['gender'] = row[cand]
            break
    # jaundice, used_app_before, austim may be numeric (0/1) in dataset; convert to 'yes'/'no' for UI
    for cand in ['jaundice', 'used_app_before', 'austim']:
        if cand in row.index:
            v = row[cand]
            # if numeric 0/1
            try:
                if float(v) == 1:
                    mapping[cand] = 'yes'
                else:
                    mapping[cand] = 'no'
            except Exception:
                # if already 'yes'/'no' keep as-is
                mapping[cand] = str(v)
    # A-scores with questions
    for i in range(1, 11):
        # try different column name patterns
        for colname in (f'A{i}_Score', f'A{i} Score', f'a{i}', f'A{i}'):
            if colname in row.index:
                mapping[f'A{i}_Score'] = row[colname]
                mapping[f'A{i}_Question'] = AQ_questions[i]
                break
        # default if score not found
        mapping.setdefault(f'A{i}_Score', 0)
        mapping.setdefault(f'A{i}_Question', AQ_questions[i])


    # result if present
    for cand in ['result', 'Result']:
        if cand in row.index:
            mapping['result'] = row[cand]
            break

    # ensure defaults
    mapping.setdefault('age', 25.0)
    mapping.setdefault('gender', 'male')
    mapping.setdefault('jaundice', 'no')
    mapping.setdefault('austim', 'no')
    mapping.setdefault('used_app_before', 'no')
    mapping.setdefault('result', 0.0)

    return mapping


def sample_from_dataset(target_label='ASD'):
    dataset_path = './data/train.csv'  # adjust if needed
    if not os.path.exists(dataset_path):
        return None
    df = pd.read_csv(dataset_path)
    class_col = find_class_column(df)
    if class_col is None:
        return None

    col_vals = df[class_col].astype(str).str.lower()
    if any(col_vals.str.contains('yes')) or any(col_vals.str.contains('no')):
        if target_label.lower() in ['asd', 'yes', 'autism']:
            cand_rows = df[col_vals.str.contains('yes')]
        else:
            cand_rows = df[col_vals.str.contains('no')]
    else:
        if target_label.lower() in ['asd', 'yes', 'autism']:
            cand_rows = df[df[class_col].astype(float) == 1]
        else:
            cand_rows = df[df[class_col].astype(float) == 0]

    if cand_rows.empty:
        return None

    row = cand_rows.sample(1).iloc[0]
    return map_row_to_input(row)


# ---------------------
# Streamlit UI + main
# ---------------------
def main():
    st.set_page_config(page_title="ASD Prediction", layout="wide")
    st.title('Autism Spectrum Disorder Prediction (with quick-load examples)')

    # Load assets
    models, feature_columns, scaler, encoders = load_model()

    # Sidebar model selection + threshold + debug
    st.sidebar.header('Settings')
    model_choice = st.sidebar.selectbox('Choose a Model', list(models.keys()))
    threshold = st.sidebar.slider('ASD probability threshold', 0.1, 0.9, 0.5, step=0.05)
    show_debug = st.sidebar.checkbox('Show Debug Info', value=False)

    # Quick load buttons
    st.sidebar.header('Quick Load Examples')
    if st.sidebar.button('Load Autism Example (from dataset)'):
        sample = sample_from_dataset('ASD')
        if sample:
            st.session_state.update(sample)
            st.sidebar.success('Loaded ASD example from dataset')
        else:
            # fallback preset
            preset = {'age': 10.0, 'result': 2.0, 'gender': 'male', 'jaundice': 'no', 'austim': 'yes', 'used_app_before': 'yes'}
            preset.update({f'A{i}_Score': 8 if i <= 6 else 7 for i in range(1, 11)})
            st.session_state.update(preset)
            st.sidebar.info('Loaded ASD preset')
        st.rerun()

    if st.sidebar.button('Load Non-ASD Example (from dataset)'):
        sample = sample_from_dataset('No ASD')
        if sample:
            st.session_state.update(sample)
            st.sidebar.success('Loaded non-ASD example from dataset')
        else:
            preset = {'age': 25.0, 'result': 0.0, 'gender': 'female', 'jaundice': 'no', 'austim': 'no', 'used_app_before': 'no'}
            preset.update({f'A{i}_Score': 1 for i in range(1, 11)})
            st.session_state.update(preset)
            st.sidebar.info('Loaded non-ASD preset')
        st.rerun()

    if st.sidebar.button('Reset Inputs'):
        st.session_state.clear()
        st.rerun()

    # Input form (two columns)
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.setdefault('age', 25.0)
        st.number_input('Age', min_value=0.0, max_value=100.0, value=st.session_state['age'], key='age')

        st.session_state.setdefault('result', 0.0)
        st.number_input('Result', min_value=-5.0, max_value=100.0, value=st.session_state['result'], key='result')

        for i in range(1, 6):
            key = f'A{i}_Score'
            st.session_state.setdefault(key, 0)
            question = AQ_questions[i]
            st.number_input(f'A{i} Score: {question}', min_value=0, max_value=10, value=st.session_state[key], key=key)


    with col2:
        st.session_state.setdefault('gender', 'male')
        st.selectbox('Gender', ['male', 'female'], index=0 if st.session_state['gender'] == 'male' else 1, key='gender')

        st.session_state.setdefault('jaundice', 'no')
        st.selectbox('Jaundice', ['yes', 'no'], index=0 if st.session_state['jaundice'] == 'yes' else 1, key='jaundice')

        st.session_state.setdefault('austim', 'no')
        st.selectbox('Autism (family member)?', ['yes', 'no'], index=0 if st.session_state['austim'] == 'yes' else 1, key='austim')

        st.session_state.setdefault('used_app_before', 'no')
        st.selectbox('Used App Before', ['yes', 'no'], index=0 if st.session_state['used_app_before'] == 'yes' else 1, key='used_app_before')

        for i in range(6, 11):
            key = f'A{i}_Score'
            st.session_state.setdefault(key, 0)
            question = AQ_questions[i]
            st.number_input(f'A{i} Score: {question}', min_value=0, max_value=10, value=st.session_state[key], key=key)


    # Recompute derived features
    def convertAge(age):
        if age < 4: return 'Toddler'
        elif age < 12: return 'Kid'
        elif age < 18: return 'Teenager'
        elif age < 40: return 'Young'
        else: return 'Senior'

    age_val = float(st.session_state.get('age', 25.0))
    st.session_state['ageGroup'] = convertAge(age_val)
    st.session_state['sum_score'] = sum(int(st.session_state.get(f'A{i}_Score', 0)) for i in range(1, 11))
    # make sure ind is integer (count of yes)
    st.session_state['ind'] = int((st.session_state.get('austim') == 'yes') + (st.session_state.get('used_app_before') == 'yes') + (st.session_state.get('jaundice') == 'yes'))

    # Small sidebar summary
    st.sidebar.markdown('**Current Derived**')
    st.sidebar.write(f"Sum score: {st.session_state['sum_score']}")
    st.sidebar.write(f"Ind: {st.session_state['ind']}")
    st.sidebar.write(f"Age group: {st.session_state['ageGroup']}")

    # Optional visualizations (kept as placeholders)
    st.sidebar.header('Model Performance Visualization')
    show_graphs = st.sidebar.checkbox('Show Training/Test Graphs',key="debug_main")

    if show_graphs:
        st.header('Model Performance Analysis')
        tabs = st.tabs(['Model Comparison', 'Training vs Test', 'ROC Curves', 'Confusion Matrix'])
        with tabs[0]:
            st.info("Model comparison visualization placeholder")
        with tabs[1]:
            st.metric("Training Accuracy", "92.3%", "+2.1%")
            st.metric("Test Accuracy", "89.7%", "-1.2%")
        with tabs[2]:
            fpr, tpr = [0, 0.2, 1], [0, 0.8, 1]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
            st.plotly_chart(fig)
        with tabs[3]:
            cm = [[45,5],[8,42]]
            fig = go.Figure(data=go.Heatmap(z=cm, x=['No ASD','ASD'], y=['No ASD','ASD'], text=cm, texttemplate="%{text}"))
            st.plotly_chart(fig)

    # Prediction
    if st.button('Predict'):
        # Gather inputs into dict
        input_data = {}
        for i in range(1, 11):
            input_data[f'A{i}_Score'] = st.session_state.get(f'A{i}_Score', 0)
        input_data['age'] = st.session_state.get('age', 25.0)
        input_data['result'] = st.session_state.get('result', 0.0)
        input_data['gender'] = st.session_state.get('gender', 'male')
        input_data['jaundice'] = st.session_state.get('jaundice', 'no')
        input_data['austim'] = st.session_state.get('austim', 'no')
        input_data['used_app_before'] = st.session_state.get('used_app_before', 'no')
        input_data['ageGroup'] = st.session_state.get('ageGroup', 'Young')
        input_data['sum_score'] = st.session_state.get('sum_score', 0)
        input_data['ind'] = st.session_state.get('ind', 0)

        # preprocess
        scaled_input, debug_df = preprocess_input(input_data, feature_columns, scaler, encoders)

        # Prediction (model choice)
        model = models[model_choice]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled_input)[0][1]
        else:
            raw = model.decision_function(scaled_input)[0]
            proba = 1 / (1 + np.exp(-raw))

        pred = 1 if proba >= threshold else 0

        st.subheader('Prediction Results')
        if pred == 1:
            st.error(f'🚨 Potential Autism Spectrum Disorder Detected (Probability: {proba:.2%})')
        else:
            st.success(f'✅ No Autism Spectrum Disorder Detected (Probability: {proba:.2%})')

        st.write(f'Model Used: {model_choice}')
        if show_debug or st.sidebar.checkbox("Show Debug Info",key="debug_sidebar"):
            st.write("Raw input_data (post basic replacements):")
            st.json(input_data)
            st.write("Prepared DataFrame (before scaling):")
            st.dataframe(debug_df)
            st.write(f"Scaled shape: {scaled_input.shape}")
            st.write(f"Raw probability: {proba:.6f}")
            st.write("Feature columns order used for model:")
            st.write(feature_columns)


if __name__ == '__main__':
    main()
