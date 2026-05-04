import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="DeepBF", layout="wide")

st.title("DeepBF: Malicious URL Detection")
st.write("Bloom Filter + evoCNN based detection system")

# =========================
# MODEL + SCALER
# =========================
@st.cache_resource
def load_artifacts():
    model = load_model("ga_evocnn_best_model.keras", compile=False)
    scaler = joblib.load("minmax_scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# =========================
# METRICS DISPLAY
# =========================
st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "98.9%")
col2.metric("Precision", "98.8%")
col3.metric("Recall", "99.8%")
col4.metric("F1-score", "99.3%")

st.subheader("Confusion Matrix")

cm_df = pd.DataFrame(
    [[1487, 69],
     [12, 5774]],
    index=["Actual Benign", "Actual Malicious"],
    columns=["Predicted Benign", "Predicted Malicious"]
)

st.table(cm_df)

# =========================
# BLOOM FILTER
# =========================
class BloomFilter2D:
    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.beta = 61
        self.filter = np.zeros((M, N), dtype=np.uint64)

    def insert(self, key):
        for seed in range(5):
            h = hash(key + str(seed))
            i = h % self.M
            j = h % self.N
            rho = h % self.beta
            self.filter[i, j] |= (1 << rho)

    def lookup(self, key):
        for seed in range(5):
            h = hash(key + str(seed))
            i = h % self.M
            j = h % self.N
            rho = h % self.beta
            if not (self.filter[i, j] & (1 << rho)):
                return False
        return True

@st.cache_resource
def create_filters():
    return BloomFilter2D(10007, 10009), BloomFilter2D(10007, 10009)

malicious_filter, benign_filter = create_filters()

# =========================
# FEATURE PROCESSING
# =========================
def prepare(features):
    features = np.array(features, dtype=np.float32).flatten()
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    features = np.clip(features, -1e6, 1e6)

    if len(features) == 79:
        features = np.pad(features, (0, 2))

    features = scaler.transform(features.reshape(1, -1))
    return features.reshape(1, 9, 9, 1)

def vector_to_string(features):
    features = np.array(features).astype(int)
    return "_".join(map(str, features))

# =========================
# CLASSIFIER
# =========================
def classify(features):
    key = vector_to_string(features)

    # Bloom filter check
    if malicious_filter.lookup(key):
        return "Malicious (Bloom Filter)"

    if benign_filter.lookup(key):
        return "Benign (Bloom Filter)"

    # CNN fallback
    x = prepare(features)
    prob = float(model.predict(x, verbose=0)[0][0])

    if prob >= 0.5:
        malicious_filter.insert(key)
        return f"Malicious (evoCNN) - Confidence: {round(prob, 4)}"
    else:
        benign_filter.insert(key)
        return f"Benign (evoCNN) - Confidence: {round(prob, 4)}"

# =========================
# UI INPUT
# =========================
st.subheader("Upload URL Feature Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file (79 or 81 features)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    row_index = st.number_input(
        "Select row index",
        min_value=0,
        max_value=len(df) - 1,
        value=0
    )

    if st.button("Predict"):
        features = df.iloc[row_index].values

        result = classify(features)

        st.subheader("Prediction Result")

        if "Malicious" in result:
            st.error(result)
        else:
            st.success(result)

else:
    st.info("Upload a CSV file to start prediction.")
