import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data = pd.read_csv('obesity_data.csv')

# Step 2: Perform data preprocessing
# Handle missing entries
data = data.dropna()

# Drop duplicate records
data = data.drop_duplicates()
print(f"Dataset preprocessed. Operational Shape: {data.shape}\n")

# Step 3: Conduct exploratory data analysis (EDA)
target_col = 'NObeyesdad'
print("Target Class Distribution")
print(data[target_col].value_counts())

#Step 4: Perform feature engineering as required
categorical_cols = data.select_dtypes(include=['object']).columns.drop(target_col)
##Convert string categorical features into mathematical binary layout (0s and 1s)
x = pd.get_dummies(data.drop(target_col, axis=1), columns=categorical_cols, drop_first=True)
#If Gender_Male is 0, the neural network is smart enough to mathematically deduce right away that the person must be Female.

#Transform text-based target classifications into target tracking integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data[target_col])

# Step 5: Train an Artificial Neural Network (ANN) model to predict obesity levels based on the given attributes
# Train-Test Split (80% Train, 20% Test)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
# Normalize numerical features using scaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Determine the size of the output tracking classes (Should equal 7)
num_classes = len(np.unique(y))

# Construct Neural Network structural architecture with multiple Hidden Layers
model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)), # Hidden Layer 1
    Dense(32, activation='relu'),                                      # Hidden Layer 2
    Dense(16, activation='relu'),                                      # Hidden Layer 3
    Dense(num_classes, activation='softmax')                           # Output Layer (Softmax)
])

# Compile the model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\n--- Starting Neural Network Training Loops ---")
# Continue Task 5: Train the network
# Running for 60 epochs with a validation split to track performance
history = model.fit(
    x_train_scaled, 
    y_train, 
    epochs=60, 
    batch_size=32, 
    validation_split=0.1
)

# --- Evaluate the Model ---
print("\n--- Model Evaluation Metrics ---")
loss, accuracy = model.evaluate(x_test_scaled, y_test)
print(f"Test Set Categorical Crossentropy Loss: {loss:<.5f}")
print(f"Test Set Overall Classification Accuracy: {accuracy * 100:.2f}%")





# STEP 6: SAVE ARTIFACTS FOR FLASK REST API

import joblib

# 1. Save the trained Keras neural network model
model.save("obesity_prediction_ANN_model.keras")

# 2. Save the fitted scaler framework
joblib.dump(scaler, "ANN_scaler.pkl")

# 3. Save the label encoder to decode predictions back to strings (e.g., 3 -> 'Obesity_Type_I')
joblib.dump(label_encoder, "target_encoder.pkl")

# 4. Save the exact column layout so the Flask API handles One-Hot encoding alignment flawlessly
feature_columns = x.columns.tolist()
joblib.dump(feature_columns, 'feature_columns.pkl')

print("\nAll artifacts successfully exported!")
print("- Model saved as 'obesity_prediction_ANN_model.keras'")
print("- Preprocessing structures saved (.pkl files)")