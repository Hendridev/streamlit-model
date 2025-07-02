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
from sklearn.model_selection import train_test_split
import regex as re
from sklearn.pipeline import Pipeline
import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, confusion_matrix, classification_report, log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import joblib
warnings.filterwarnings('ignore')
from datetime import datetime as dt

st.title(":bar_chart: Judul")
st.write("By: ")
st.link_button("Bhima Fairul Rifqi on LinkedIn", "https://linkedin.com/in/fairulrifqi962")
st.divider()

# dataframe multi
df = st.file_uploader(
    "Pilih file", accept_multiple_files = False, type = ['xlsx','csv']
)

# Logic: arr[0,1,2,3,4,5]
df = pd.read_csv('/content/drive/MyDrive/Kerja/Rakamin Bootcamp/Final Project/Dataset/general_data.csv')
# df_employee_survey
df_employee_survey = pd.read_csv('/content/drive/MyDrive/Kerja/Rakamin Bootcamp/Final Project/Dataset/employee_survey_data.csv')
# df_manager_survey
df_manager_survey = pd.read_csv('/content/drive/MyDrive/Kerja/Rakamin Bootcamp/Final Project/Dataset/manager_survey_data.csv')
# df_in_time
df_in_time = pd.read_csv('/content/drive/MyDrive/Kerja/Rakamin Bootcamp/Final Project/Dataset/in_time.csv')
# df_out_time
df_out_time = pd.read_csv('/content/drive/MyDrive/Kerja/Rakamin Bootcamp/Final Project/Dataset/out_time.csv')
# df_working_hours
df_working_hours = pd.read_csv('/content/drive/MyDrive/Kerja/Rakamin Bootcamp/Final Project/Dataset/df_working_hours.csv')

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


# df
df[cat_columns] = df[cat_columns].astype('category')
df.drop(columns=single_value_columns, inplace=True)
cat_columns = cat_columns.drop(single_value_columns)

ordered_categories = [1,2,3,4]

# Employee Survey
for column in df_employee_survey.columns[1:]:
  df_employee_survey[column] = df_employee_survey[column].astype(CategoricalDtype(categories=ordered_categories, ordered=True))

df_employee_survey_2 = df_employee_survey.copy()
df_employee_survey_2 = df_employee_survey_2.drop(columns='EmployeeID')


# Manager Survey
for column in df_manager_survey.columns[1:]:
  df_manager_survey[column] = df_manager_survey[column].astype(CategoricalDtype(categories=ordered_categories, ordered=True))

df_manager_survey_2 = df_manager_survey.copy()
df_manager_survey_2 = df_manager_survey_2.drop(columns='EmployeeID')

# Df time
df_in_time.rename(columns={'Unnamed: 0':'EmployeeID'}, inplace=True)
for i in df_in_time:
  df_in_time[i] = df_in_time[i].replace(np.nan, 'Tidak Hadir')

df_in_time.drop(columns = 'EmployeeID', inplace= True)

time_in = []

for i, j in df_in_time.iterrows():
  for k in j.values:
    if k != 'Tidak Hadir':
      time_in.append(k)

df_out_time.rename(columns={'Unnamed: 0': 'EmployeeID'}, inplace= True)

for i in df_out_time:
  df_out_time[i] = df_out_time[i].replace(np.nan, 'Tidak Hadir')

df_out_time.drop(columns = 'EmployeeID', inplace = True)

time_out = []

for i, j in df_out_time.iterrows():
  for k in j.values:
    if k != 'Tidak Hadir':
      time_out.append(k)

# Data time extraction

time_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
time_in = pd.DataFrame(time_in)
time_out = pd.DataFrame(time_out)

time_in.rename(columns={0:'time_in'}, inplace=True)
time_out.rename(columns={0:'time_out'}, inplace=True)

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

df_working_hours['avg_working_hours'] = df_working_hours.mean(axis= 1)
df_working_hours.insert(loc=0, column="EmployeeID", value = df['EmployeeID'])

df_working_hours_resigned = df_working_hours[df_working_hours['EmployeeID'].isin(df[df['Attrition'] == 'Yes']['EmployeeID'])]
df_working_hours_not_resigned = df_working_hours[df_working_hours['EmployeeID'].isin(df[df['Attrition'] == 'No']['EmployeeID'])]

df_working_hours_resigned = df_working_hours_resigned.drop(columns=['EmployeeID', 'avg_working_hours'])
df_working_hours_not_resigned = df_working_hours_not_resigned.drop(columns=['EmployeeID', 'avg_working_hours'])

# transpose

df_working_hours_resigned = df_working_hours_resigned.transpose()
df_working_hours_not_resigned = df_working_hours_not_resigned.transpose()

# getting its average

df_working_hours_not_resigned['mean'] = df_working_hours_not_resigned.mean(axis=1)
df_working_hours_resigned['mean'] = df_working_hours_resigned.mean(axis=1)

# average cleansing

df_working_hours_resigned = df_working_hours_resigned[df_working_hours_resigned['mean'] != 0]
df_working_hours_not_resigned = df_working_hours_not_resigned[df_working_hours_not_resigned['mean'] != 0]

# Merge

all_df = pd.merge(df, df_employee_survey, on='EmployeeID', how='left')
all_df = pd.merge(all_df, df_manager_survey, on='EmployeeID', how='left')
all_df['AvgWorkingHours'] = df_working_hours['avg_working_hours'].astype(float)
all_df.drop(columns=['EmployeeID'], inplace=True)

x = all_df.drop(columns=['Attrition'])
y = all_df['Attrition']

#########


x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_test = x

num_columns_2 = x_train.select_dtypes(include=['int64', 'float64']).columns
num_columns_2 = num_columns_2.drop(ordinal_cat_columns)

cat_columns_2 = x_train.columns[~x_train.columns.isin(num_columns_2)]

business_travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}

# Apply
x_train['BusinessTravel'] = x_train['BusinessTravel'].map(business_travel_map)
x_test['BusinessTravel'] = x_test['BusinessTravel'].map(business_travel_map)


order = [1, 2, 3, 4]

# Post-Merged Ordinal Categoric Columns
ordinal_cat_columns_2 = []

for col in cat_columns_2:
  if x_train[col].isin(order).any():
    ordinal_cat_columns_2.append(col)

# Post-Merged Nominal Categoric Columns
ohe_columns_2 = cat_columns_2[(~cat_columns_2.isin(ordinal_cat_columns_2))]

x_train[ordinal_cat_columns_2] = x_train[ordinal_cat_columns_2].astype('category')
x_test[ordinal_cat_columns_2] = x_test[ordinal_cat_columns_2].astype('category')

x_train[ohe_columns_2] = x_train[ohe_columns_2].astype('category')
x_test[ohe_columns_2] = x_test[ohe_columns_2].astype('category')

x_train.duplicated().sum()
x_train.drop_duplicates(inplace=True)
y_train = y_train[x_train.index]

for i in x_train:
  missing = x_train[i].isna().sum()
  proportion = missing / len(df) * 100
  print(f'{i}: {missing} ({proportion:.2f}%)')

num_imputer = Pipeline([
  ('imputer', SimpleImputer(strategy='median'))
])

cat_imputer = Pipeline([
  ('imputer', SimpleImputer(strategy='most_frequent'))
])

x_train[num_columns_2] = num_imputer.fit_transform(x_train[num_columns_2])
x_train[cat_columns_2] = cat_imputer.fit_transform(x_train[cat_columns_2])

x_test[num_columns_2] = num_imputer.transform(x_test[num_columns_2])
x_test[cat_columns_2] = cat_imputer.transform(x_test[cat_columns_2])

for i in x_train:
  missing = x_train[i].isna().sum()
  proportion = missing / len(df) * 100
  print(f'{i}: {missing} ({proportion:.2f}%)')


# Significant Features
significant_num = []
significant_cat = []
significants = []

# Pipeline
class StatisticalFeatureSelector(BaseEstimator, TransformerMixin):
	def __init__(self, num_cols, cat_cols, target='attrition', p_threshold=0.05):
		self.num_cols = num_cols
		self.cat_cols = cat_cols
		self.target = target
		self.p_threshold = p_threshold
		self.significant_features = []

	def fit(self, x, y):
		global significant_num, significant_cat, significants

		df = x.copy()
		df[self.target] = y.map({'No': 0, 'Yes': 1}).astype(int)

		# Numeric feature p-values
		formula = self.target + ' ~ ' + ' + '.join(self.num_cols)
		base_model = smf.logit(formula, data=df).fit(disp=False)
		numeric_pvalues = base_model.pvalues.drop('Intercept')

		significant_num = numeric_pvalues[numeric_pvalues < self.p_threshold].index.tolist()

		# Categorical feature p-values
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

		significants = significant_num + significant_cat
		self.significant_features = significants
		return self

	def transform(self, x):
		return x[self.significant_features]
   
selector = StatisticalFeatureSelector(num_columns_2, cat_columns_2)
x_train = selector.fit_transform(x_train, y_train)
x_test = selector.transform(x_test)

final_num_columns = [col for col in num_columns_2 if col in significants]
final_cat_columns = [col for col in cat_columns_2 if col in significants]
final_ordinal_cat_columns = [col for col in ordinal_cat_columns_2 if col in significants]
final_ohe_columns = [col for col in ohe_columns_2 if col in significants]

# Custom log1p transformer
class LogTransformer(BaseEstimator, TransformerMixin):
	def fit(self, x, y=None):
		return self

	def transform(self, x):
		return np.log1p(x)

# Numeric transformation pipeline
num_pipeline = Pipeline([
	('log', LogTransformer()),
	('scaler', RobustScaler())
])

# Pipeline
num_preprocessor = ColumnTransformer([
	('num', num_pipeline, final_num_columns)
])

x_train[final_num_columns] = num_preprocessor.fit_transform(x_train[final_num_columns])
x_test[final_num_columns] = num_preprocessor.transform(x_test[final_num_columns])

for col in final_ordinal_cat_columns:
  print(f'{col}: {x_train[col].unique()}')

# Declare the order
ordinal_categories = [
	sorted(x_train[col].dropna().unique().tolist())
	for col in final_ordinal_cat_columns
]

# Pipeline
ordinal_encode = Pipeline([
	('ordinal_encoder', OrdinalEncoder(categories=ordinal_categories))
])

x_train[final_ordinal_cat_columns] = ordinal_encode.fit_transform(x_train[final_ordinal_cat_columns])
x_test[final_ordinal_cat_columns] = ordinal_encode.transform(x_test[final_ordinal_cat_columns])


ohe_encode = Pipeline([
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

ohe_encode.fit(x_train[final_ohe_columns])


x_train_ohe = ohe_encode.transform(x_train[final_ohe_columns])
x_test_ohe = ohe_encode.transform(x_test[final_ohe_columns])

x_train_ohe = pd.DataFrame(x_train_ohe.toarray(), columns=ohe_encode.get_feature_names_out())
x_test_ohe = pd.DataFrame(x_test_ohe.toarray(), columns=ohe_encode.get_feature_names_out())

x_train = x_train.drop(columns=final_ohe_columns)
x_train = pd.merge(x_train, x_train_ohe, on=x_train.index)
x_train.index = x_train.pop('key_0')

x_test = x_test.drop(columns=final_ohe_columns)
x_test = pd.merge(x_test, x_test_ohe, on=x_test.index)
x_test.index = x_test.pop('key_0')
# remove index name
x_train.index.name = None
x_test.index.name = None

preprocessor = ColumnTransformer([
	('num', num_preprocessor, final_num_columns),
	('ordinal', ordinal_encode, final_ordinal_cat_columns),
	('ohe', ohe_encode, final_ohe_columns)
])


model = joblib.load("tuned_classifier.pkl")
res = model.predict(x_test)
res = pd.DataFrame(res, columns=['Attrition'])
res['Attrition'] = res['Attrition'].map({1: 'Yes', 0: 'No'})
res





