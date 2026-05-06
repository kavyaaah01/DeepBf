import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="DeepBF", layout="wide")

st.title("DeepBF: Malicious URL Detection")
st.write("Bloom Filter + evoCNN based detection system")

@st.cache_resource
def load_artifacts():
    model = load_model("ga_evocnn_best_model.keras", compile=False)
    scaler = joblib.load("minmax_scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

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

            mask = np.uint64(1) << np.uint64(rho)
            self.filter[i, j] = self.filter[i, j] | mask

    def lookup(self, key):
        for seed in range(5):
            h = hash(key + str(seed))
            i = h % self.M
            j = h % self.N
            rho = h % self.beta

            mask = np.uint64(1) << np.uint64(rho)

            if (self.filter[i, j] & mask) == 0:
                return False

        return True

@st.cache_resource
def create_filters():
    return BloomFilter2D(10007, 10009), BloomFilter2D(10007, 10009)

malicious_filter, benign_filter = create_filters()


def prepare(features):
    features = np.array(features, dtype=np.float32).flatten()
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    features = np.clip(features, -1e6, 1e6)

    if len(features) == 79:
        features = np.pad(features, (0, 2))
    elif len(features) != 81:
        raise ValueError(f"Expected 79 or 81 feature columns, got {len(features)}")

    features = scaler.transform(features.reshape(1, -1))
    return features.reshape(1, 9, 9, 1)

def vector_to_string(features):
    features = np.array(features).astype(int)
    return "_".join(map(str, features))


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

    selected_row = df.iloc[row_index].copy()

    # Remove label column if present
    if "label" in df.columns:
        actual_label = selected_row["label"]
        features = selected_row.drop(labels=["label"]).values
    else:
        actual_label = None
        features = selected_row.values

    # Ensure numeric
    features = pd.to_numeric(pd.Series(features), errors="coerce").fillna(0).values

    result = classify(features)

    st.subheader("Prediction Result")

    if "Malicious" in result:
        st.error(result)
    else:
        st.success(result)

    if actual_label is not None:
        st.write("Actual Label:", actual_label)
   
