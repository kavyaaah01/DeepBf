import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="DeepBF", layout="wide")

st.title("DeepBF: Malicious URL Detection")
st.write("Bloom Filter + evoCNN based malicious URL detection system")


# =========================
# LOAD MODEL + SCALER
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
# HASH FUNCTION
# =========================
def mmurmur(key: str, seed: int = 0) -> int:
    data = key.encode("utf-8")
    length = len(data)
    h = seed ^ length

    for c in data:
        k = c & 0xFF
        k = ((k << 13) | (k >> 19)) & 0xFFFFFFFF
        k ^= (k << 7) & 0xFFFFFFFF
        k = ((k << 5) | (k >> 27)) & 0xFFFFFFFF

        h ^= k
        h = ((h << 11) | (h >> 21)) & 0xFFFFFFFF
        h ^= (h << 3) & 0xFFFFFFFF

    h ^= length
    h ^= (h >> 16)
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h ^= (h >> 13)

    return h & 0xFFFFFFFF


# =========================
# BLOOM FILTER
# =========================
class BloomFilter2D:
    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.beta = 61
        self.filter = np.zeros((M, N), dtype=np.uint64)

    def _get_positions(self, key):
        positions = []

        for seed in range(5):
            h = mmurmur(key, seed)
            i = h % self.M
            j = h % self.N
            rho = h % self.beta
            positions.append((i, j, rho))

        return positions

    def insert(self, key):
        for i, j, rho in self._get_positions(key):
            mask = np.uint64(1) << np.uint64(rho)
            self.filter[i, j] = self.filter[i, j] | mask

    def lookup(self, key):
        for i, j, rho in self._get_positions(key):
            mask = np.uint64(1) << np.uint64(rho)

            if (self.filter[i, j] & mask) == 0:
                return False

        return True


@st.cache_resource
def create_filters():
    malicious_filter = BloomFilter2D(10007, 10009)
    benign_filter = BloomFilter2D(10007, 10009)
    return malicious_filter, benign_filter


malicious_filter, benign_filter = create_filters()


# =========================
# FEATURE PROCESSING
# =========================
def clean_feature_vector(features):
    features = np.array(features, dtype=np.float32).flatten()
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    features = np.clip(features, -1e6, 1e6)
    return features


def vector_to_string(features):
    features = clean_feature_vector(features)
    return "_".join(map(str, features.astype(int)))


def prepare(features):
    features = clean_feature_vector(features)

    if len(features) == 79:
        features = np.pad(features, (0, 2), mode="constant")
    elif len(features) != 81:
        raise ValueError(f"Expected 79 or 81 feature columns, got {len(features)}")

    features = scaler.transform(features.reshape(1, -1))
    return features.reshape(1, 9, 9, 1)


# =========================
# CLASSIFIER
# =========================
def classify(features):
    key = vector_to_string(features)

    if malicious_filter.lookup(key):
        return "Malicious", "Malicious Bloom Filter", 1.0

    if benign_filter.lookup(key):
        return "Benign", "Benign Bloom Filter", 1.0

    x = prepare(features)
    prob = float(model.predict(x, verbose=0)[0][0])

    if prob >= 0.5:
        malicious_filter.insert(key)
        return "Malicious", "evoCNN", prob
    else:
        benign_filter.insert(key)
        return "Benign", "evoCNN", prob


# =========================
# UI
# =========================
st.subheader("Upload URL Feature Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file with 79 or 81 feature columns",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(
        df,
        use_container_width=True,
        height=500
    )

    row_index = st.number_input(
        "Select row index for prediction",
        min_value=0,
        max_value=len(df) - 1,
        value=0
    )

    if st.button("Predict"):
        selected_row = df.iloc[row_index].copy()
        actual_label = None

        # Case 1: label column exists by name
        if "label" in df.columns:
            actual_label = selected_row["label"]
            features = selected_row.drop(labels=["label"]).values

        # Case 2: label column exists but unnamed; total columns = 80 or 82
        elif len(selected_row.values) in [80, 82]:
            actual_label = selected_row.values[-1]
            features = selected_row.values[:-1]

        # Case 3: only features
        else:
            features = selected_row.values

        features = pd.to_numeric(
            pd.Series(features),
            errors="coerce"
        ).fillna(0).values

        try:
            label, source, confidence = classify(features)

            st.subheader("Prediction Result")

            if label == "Malicious":
                st.error(f"Prediction: {label}")
            else:
                st.success(f"Prediction: {label}")

            st.write("Source:", source)
            st.write("Confidence:", round(confidence, 4))

            if actual_label is not None:
                st.write("Actual Label:", actual_label)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.info("Upload a CSV file to begin prediction.")
