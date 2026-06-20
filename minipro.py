import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib



data = pd.read_csv('student_performance.csv')

# Handle missing entries and drop duplicate records (Assignment rule)
data = data.dropna() 
data = data.drop_duplicates()

# Keep exactly the first 100 rows and omit everything else
data = data.head(100).copy()

print("--- Data Preparation ---")
print(f"Dataset successfully truncated. Operational Shape: {data.shape}\n")

# *******************************************
# LABEL ENCODING
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()
label_encoder3 = LabelEncoder()

data['gender_encoded'] = label_encoder1.fit_transform(data['gender'])
data['major_encoded'] = label_encoder2.fit_transform(data['major'])
data['part_time_job_encoded'] = label_encoder3.fit_transform(data['part_time_job'])

# 
#  SCALING
# 
x = data[['age', 'gender_encoded', 'study_hours', 'attendance', 'major_encoded', 'part_time_job_encoded']]
y = data[['gpa']]

x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_scaled = x_scaler.fit_transform(x)
y_scaled = y_scaler.fit_transform(y)

# *****@$$$$$$
# 4. TRAIN-TEST SPLIT & MODEL TRAINING
# ***********-----@@@@!!!!!!

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

# ***********@@@@@@@@@@@!!!!!!!!!!!
# 5. EVALUATION
# ^^^^^^***********************

y_pred = model.predict(x_test)
error = mean_squared_error(y_test, y_pred)
rms = np.sqrt(error)
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation Metrics ---")
print(f"Root Mean Squared Error (RMSE): {rms:<.5f}")
print(f"R-squared (R2) Score: {r2:.5f}")

#**************
# 6. SAVE ARTIFACTS
#$$$$$$$$$$$$$$%$$$$$$$$$$$

joblib.dump(model, 'student_gpa_model.pkl')
joblib.dump(x_scaler, 'x_scaler_student.pkl')
joblib.dump(y_scaler, 'y_scaler_student.pkl')
joblib.dump(label_encoder1, 'label_encoder1_student.pkl')
joblib.dump(label_encoder2, 'label_encoder2_student.pkl')
joblib.dump(label_encoder3, 'label_encoder3_student.pkl')

print("All components and the model have been successfully saved.")