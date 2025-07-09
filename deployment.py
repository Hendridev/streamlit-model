import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Load model ---
@st.cache_resource
def load_model():
    logger.info("Loading model from tuned_classifier.pkl...")
    model = joblib.load("tuned_classifier.pkl")
    logger.info("Model loaded successfully.")
    return model

model = load_model()

# --- Download Template ---
st.title("Employee Attrition Predictor")
st.write("by: Asterisk Celestials | Team 5")

st.markdown("### 1. Unduh Template Input")
with open("User_Template.xlsx", "rb") as file:
    st.download_button(label="📥 Download Template",
                       data=file,
                       file_name="input_template.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- Upload File ---
st.markdown("### 2. Upload File Data")
uploaded_file = st.file_uploader("Unggah file CSV sesuai template", type=["csv", "xlsx"])

if uploaded_file:
    logger.info(f"File uploaded: {uploaded_file.name}")

    try:
        # Support CSV and XLSX
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)

        st.write("📝 Data yang diunggah:")
        st.dataframe(df_input)

        # Tombol prediksi
        if st.button("🔮 Predict"):
            logger.info("Prediction button clicked, starting prediction...")

            # --- Pipeline preprocessing ---
            employee_ids = df_input['EmployeeID'].copy()

            # Fitur numerik dan kategorikal
            num_columns = ['Age', 'NumCompaniesWorked', 'TotalWorkingYears',
                           'TrainingTimesLastYear', 'YearsSinceLastPromotion',
                           'YearsWithCurrManager', 'AvgWorkingHours']
            ordinal_cat_columns = ['BusinessTravel', 'JobLevel', 'EnvironmentSatisfaction',
                                   'JobSatisfaction', 'WorkLifeBalance']
            ohe_columns = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']

            # --- Imputasi numerik: median ---
            for col in num_columns:
                if df_input[col].isnull().any():
                    median_value = df_input[col].median()
                    df_input[col] = df_input[col].fillna(median_value)
                    logger.info(f"Filled missing numeric values in '{col}' with median: {median_value}")

            # --- Imputasi kategorikal: modus ---
            for col in ordinal_cat_columns + ohe_columns:
                if df_input[col].isnull().any():
                    mode_value = df_input[col].mode(dropna=True)[0]
                    df_input[col] = df_input[col].fillna(mode_value)
                    logger.info(f"Filled missing categorical values in '{col}' with mode: {mode_value}")

            # Custom Log Transformer
            class LogTransformer(BaseEstimator, TransformerMixin):
                def fit(self, x, y=None): return self
                def transform(self, x): return np.log1p(x)

            # Pipeline numerik
            num_pipeline = Pipeline([
                ('log', LogTransformer()),
                ('scaler', RobustScaler())
            ])

            # Encoder ordinal dengan urutan kategori otomatis dari data
            ordinal_encoder = OrdinalEncoder(categories=[
                sorted(df_input[col].unique().tolist()) for col in ordinal_cat_columns
            ])

            # Preprocessor gabungan
            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, num_columns),
                ('ordinal', ordinal_encoder, ordinal_cat_columns),
                ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), ohe_columns)
            ])

            features_to_process = num_columns + ordinal_cat_columns + ohe_columns
            processed_data = preprocessor.fit_transform(df_input[features_to_process])
            logger.info("Data preprocessing complete.")

            # Prediksi
            prediction = model.predict(processed_data)
            logger.info(f"Prediction complete. Predictions: {prediction}")

            # Ubah prediksi ke label (optional)
            df_input["Prediction"] = np.where(prediction == 1, "Leave", "Stay")

            # Tampilkan hasil
            st.markdown("### 📊 Hasil Prediksi")
            st.dataframe(df_input)

            # Unduh hasil
            csv_result = df_input.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Unduh Hasil Prediksi", csv_result, "hasil_prediksi.csv", "text/csv")

    except Exception as e:
        logger.exception("Terjadi kesalahan saat memproses data")
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
