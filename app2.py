# package
import streamlit as st
import pandas as pd
import numpy as np
import statistics as stats
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sx
import math
import warnings
import sklearn
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2
import regex as re
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, confusion_matrix, classification_report, log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import pickle
import xlsxwriter
import joblib # Pastikan joblib diimpor
import io     # Diperlukan untuk fitur unduh
warnings.filterwarnings('ignore')
from datetime import datetime as dt

# ==============================================================================
# BAGIAN STREAMLIT (UI)
# ==============================================================================

# Fungsi bantuan untuk mengonversi DataFrame ke file Excel di memori
@st.cache_data
def to_excel(df):
    output = io.BytesIO()
    # Gunakan 'with' untuk memastikan writer ditutup dengan benar
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
    processed_data = output.getvalue()
    return processed_data

st.title(":bar_chart: Aesterik")
st.write("By: Bhima Fairul Rifqi")
st.link_button("LinkedIn Profile", "https://linkedin.com/in/fairulrifqi962")
st.divider()

st.header("Upload 5 File Data yang Dibutuhkan")

# Membuat uploader file yang terpisah untuk setiap file
general_data_file = st.file_uploader("1. General Data", type=['xlsx', 'csv'])
employee_survey_file = st.file_uploader("2. Employee Survey Data", type=['xlsx', 'csv'])
manager_survey_file = st.file_uploader("3. Manager Survey Data", type=['xlsx', 'csv'])
in_time_file = st.file_uploader("4. In-Time Data", type=['xlsx', 'csv'])
out_time_file = st.file_uploader("5. Out-Time Data", type=['xlsx', 'csv'])


# Tombol untuk memulai proses prediksi
if st.button("Submit & Prediksi", type="primary"):
    # Cek apakah semua file sudah diunggah
    if all([general_data_file, employee_survey_file, manager_survey_file, in_time_file, out_time_file]):
        with st.spinner("Sedang memproses data dan menjalankan prediksi..."):
            try:
                # Membaca setiap file ke dalam DataFrame yang sesuai
                # Secara otomatis mendeteksi apakah file tersebut csv atau xlsx
                df = pd.read_csv(general_data_file) if general_data_file.name.endswith('csv') else pd.read_excel(general_data_file)
                df_employee_survey = pd.read_csv(employee_survey_file) if employee_survey_file.name.endswith('csv') else pd.read_excel(employee_survey_file)
                df_manager_survey = pd.read_csv(manager_survey_file) if manager_survey_file.name.endswith('csv') else pd.read_excel(manager_survey_file)
                df_in_time = pd.read_csv(in_time_file) if in_time_file.name.endswith('csv') else pd.read_excel(in_time_file)
                df_out_time = pd.read_csv(out_time_file) if out_time_file.name.endswith('csv') else pd.read_excel(out_time_file)
                
                # Menyimpan EmployeeID untuk hasil akhir
                final_employee_ids = df['EmployeeID'].copy()

                # ==============================================================================
                # KODE AWAL ANDA (TIDAK DIUBAH, HANYA DISESUAIKAN UNTUK BERJALAN DI SINI)
                # ==============================================================================
                
                num_columns = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
                               'PercentSalaryHike', 'TotalWorkingYears',
                               'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',
                               'YearsWithCurrManager']

                cat_columns = df.columns[~df.columns.isin(num_columns) & ~df.columns.isin(['EmployeeID'])]
                single_value_columns = []

                for column_name in cat_columns:
                    if df[column_name].nunique() == 1:
                        single_value_columns.append(column_name)

                num_type = ['int64', 'float64']
                ordinal_cat_columns = []

                for column_name in cat_columns:
                    if column_name not in single_value_columns:
                        if df[column_name].dtype in num_type:
                            ordinal_cat_columns.append(column_name)

                cat_columns = cat_columns[(~cat_columns.isin(ordinal_cat_columns))]
                ohe_columns = []

                for column_name in cat_columns:
                    if df[column_name].nunique() > 2:
                        ohe_columns.append(column_name)

                binary_columns = cat_columns[(~cat_columns.isin(ohe_columns)) & (~cat_columns.isin(single_value_columns))]

                df[cat_columns] = df[cat_columns].astype('category')
                df.drop(columns=single_value_columns, inplace=True)
                cat_columns = cat_columns.drop(single_value_columns)

                ordered_categories = [1, 2, 3, 4]

                for column in df_employee_survey.columns[1:]:
                    df_employee_survey[column] = df_employee_survey[column].astype(CategoricalDtype(categories=ordered_categories, ordered=True))
                
                for column in df_manager_survey.columns[1:]:
                    df_manager_survey[column] = df_manager_survey[column].astype(CategoricalDtype(categories=ordered_categories, ordered=True))

                df_in_time.rename(columns={'Unnamed: 0':'EmployeeID'}, inplace=True)
                for i in df_in_time:
                    df_in_time[i] = df_in_time[i].replace(np.nan, 'Tidak Hadir')
                df_in_time.drop(columns='EmployeeID', inplace=True)

                time_in = []
                for i, j in df_in_time.iterrows():
                    for k in j.values:
                        if k != 'Tidak Hadir':
                            time_in.append(k)

                df_out_time.rename(columns={'Unnamed: 0': 'EmployeeID'}, inplace=True)
                for i in df_out_time:
                    df_out_time[i] = df_out_time[i].replace(np.nan, 'Tidak Hadir')
                df_out_time.drop(columns='EmployeeID', inplace=True)

                time_out = []
                for i, j in df_out_time.iterrows():
                    for k in j.values:
                        if k != 'Tidak Hadir':
                            time_out.append(k)

                time_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
                time_in = pd.DataFrame(time_in, columns=['time_in'])
                time_out = pd.DataFrame(time_out, columns=['time_out'])

                time_in = pd.to_datetime(time_in['time_in'])
                time_out = pd.to_datetime(time_out['time_out'])

                time_diff = time_out - time_in
                time_diff = time_diff.dt.total_seconds() / 3600
                time_diff = round(time_diff, 1)

                df_working_hours = df_out_time.copy()
                replaced = 0
                for i in range(len(df_working_hours)):
                    for j in df_working_hours.columns:
                        value = df_working_hours.at[i, j]
                        if isinstance(value, str) and re.fullmatch(time_pattern, value):
                            if replaced < len(time_diff):
                                df_working_hours.at[i, j] = time_diff[replaced]
                                replaced += 1
                
                for index in df_working_hours.index:
                    for col in df_working_hours.columns:
                        if df_working_hours.at[index, col] == 'Tidak Hadir':
                            df_working_hours.at[index, col] = 0

                df_working_hours['avg_working_hours'] = df_working_hours.mean(axis=1)
                df_working_hours.insert(loc=0, column="EmployeeID", value = df['EmployeeID'])

                all_df = pd.merge(df, df_employee_survey, on='EmployeeID', how='left')
                all_df = pd.merge(all_df, df_manager_survey, on='EmployeeID', how='left')
                all_df['AvgWorkingHours'] = df_working_hours['avg_working_hours'].astype(float)
                all_df.drop(columns=['EmployeeID'], inplace=True)

                x = all_df.drop(columns=['Attrition'])
                y = all_df['Attrition']

                x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
                x_test = x

                num_columns_2 = x_train.select_dtypes(include=['int64', 'float64']).columns
                num_columns_2 = num_columns_2.drop(ordinal_cat_columns, errors='ignore')
                cat_columns_2 = x_train.columns[~x_train.columns.isin(num_columns_2)]

                business_travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
                x_train['BusinessTravel'] = x_train['BusinessTravel'].map(business_travel_map)
                x_test['BusinessTravel'] = x_test['BusinessTravel'].map(business_travel_map)

                order = [1, 2, 3, 4]
                ordinal_cat_columns_2 = [col for col in cat_columns_2 if x_train[col].isin(order).any()]
                ohe_columns_2 = cat_columns_2[~cat_columns_2.isin(ordinal_cat_columns_2)]

                x_train[ordinal_cat_columns_2] = x_train[ordinal_cat_columns_2].astype('category')
                x_test[ordinal_cat_columns_2] = x_test[ordinal_cat_columns_2].astype('category')
                x_train[ohe_columns_2] = x_train[ohe_columns_2].astype('category')
                x_test[ohe_columns_2] = x_test[ohe_columns_2].astype('category')

                x_train.drop_duplicates(inplace=True)
                y_train = y_train[x_train.index]

                num_imputer = Pipeline([('imputer', SimpleImputer(strategy='median'))])
                cat_imputer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))])

                x_train[num_columns_2] = num_imputer.fit_transform(x_train[num_columns_2])
                x_train[cat_columns_2] = cat_imputer.fit_transform(x_train[cat_columns_2])
                x_test[num_columns_2] = num_imputer.transform(x_test[num_columns_2])
                x_test[cat_columns_2] = cat_imputer.transform(x_test[cat_columns_2])
                
                # ... (lanjutan kode asli Anda) ...
                class StatisticalFeatureSelector(BaseEstimator, TransformerMixin):
                    def __init__(self, num_cols, cat_cols, target='attrition', p_threshold=0.05):
                        self.num_cols = num_cols
                        self.cat_cols = cat_cols
                        self.target = target
                        self.p_threshold = p_threshold
                        self.significant_features = []
                    def fit(self, x, y):
                        df = x.copy()
                        df[self.target] = y.map({'No': 0, 'Yes': 1}).astype(int)
                        formula = self.target + ' ~ ' + ' + '.join(self.num_cols)
                        base_model = smf.logit(formula, data=df).fit(disp=False)
                        numeric_pvalues = base_model.pvalues.drop('Intercept')
                        significant_num = numeric_pvalues[numeric_pvalues < self.p_threshold].index.tolist()
                        significant_cat = []
                        for var in self.cat_cols:
                            formula_full = formula + f' + C({var})'
                            try:
                                model_full = smf.logit(formula_full, data=df).fit(disp=False)
                                model_reduced = base_model
                                lr_stat = 2 * (model_full.llf - model_reduced.llf)
                                df_diff = model_full.df_model - model_reduced.df_model
                                p_val = chi2.sf(lr_stat, df_diff)
                                if p_val < self.p_threshold:
                                    significant_cat.append(var)
                            except:
                                pass
                        self.significant_features = significant_num + significant_cat
                        return self
                    def transform(self, x):
                        return x[self.significant_features]

                selector = StatisticalFeatureSelector(num_columns_2, cat_columns_2, target='Attrition')
                x_train = selector.fit_transform(x_train, y_train)
                x_test = selector.transform(x_test)

                significants = selector.significant_features
                final_num_columns = [col for col in num_columns_2 if col in significants]
                final_ohe_columns = [col for col in ohe_columns_2 if col in significants]
                final_ordinal_cat_columns = [col for col in ordinal_cat_columns_2 if col in significants]

                class LogTransformer(BaseEstimator, TransformerMixin):
                    def fit(self, x, y=None): return self
                    def transform(self, x): return np.log1p(x)

                num_pipeline = Pipeline([('log', LogTransformer()), ('scaler', RobustScaler())])
                
                # Final Preprocessing Pipeline
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', num_pipeline, final_num_columns),
                        ('ordinal', OrdinalEncoder(categories=[sorted(x_train[col].dropna().unique().tolist()) for col in final_ordinal_cat_columns]), final_ordinal_cat_columns),
                        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), final_ohe_columns)
                    ],
                    remainder='passthrough' # Penting agar kolom tidak hilang
                )

                x_train_processed = preprocessor.fit_transform(x_train)
                x_test_processed = preprocessor.transform(x_test)
                
                model = joblib.load("tuned_classifier.pkl")
                predictions = model.predict(x_test_processed)
                
                # Membuat hasil akhir
                res = pd.DataFrame(predictions, columns=['Attrition'])
                res['Attrition'] = res['Attrition'].map({1: 'Yes', 0: 'No'})
                # Menambahkan kembali EmployeeID
                res.insert(0, 'EmployeeID', final_employee_ids)

                # ==============================================================================
                # AKHIR DARI KODE AWAL ANDA
                # ==============================================================================

                st.success("✅ Prediksi berhasil dijalankan!")
                st.dataframe(res)

                # Menyiapkan data untuk diunduh
                excel_data = to_excel(res)

                st.download_button(
                    label="📥 Unduh Hasil Prediksi (Excel)",
                    data=excel_data,
                    file_name="hasil_prediksi_atrisi.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"Terjadi kesalahan saat pemrosesan: {e}")
                st.warning("Mohon periksa kembali format dan konten file yang Anda unggah.")

    else:
        st.warning("⚠️ Mohon unggah kelima file yang diperlukan untuk melanjutkan.")