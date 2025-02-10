import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import flask
from flask import Flask, render_template, jsonify
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from cryptography.fernet import Fernet
from joblib import dump

app = Flask(__name__)
# Load Dataset
df = pd.read_csv("Test_data.csv")

# First, let's check the data
print("Unique values in 'hot' column:", df["hot"].unique())
print("Number of null values:", df["hot"].isnull().sum())

# Feature selection
features = df.drop(columns=["hot"])

# Improved label encoding for target variable
le = LabelEncoder()
labels = le.fit_transform(df["hot"].astype(str))  # Convert to string first to handle any non-string values

# Label encoding for categorical features
for col in features.select_dtypes(include=["object"]).columns:
    features[col] = le.fit_transform(features[col].astype(str))

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# AutoEncoder
input_dim = features_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation="relu")(input_layer)
encoded = Dropout(0.2)(encoded)
encoded = Dense(64, activation="relu")(encoded)
encoded = Dropout(0.2)(encoded)
encoded = Dense(32, activation="relu")(encoded)

decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(128, activation="relu")(decoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# Train AutoEncoder
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
autoencoder.fit(
    features_scaled, 
    features_scaled, 
    epochs=10, 
    batch_size=32, 
    shuffle=True,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Extract encoded features
encoder = Model(input_layer, encoded)
encoded_features = encoder.predict(features_scaled)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(encoded_features, labels, test_size=0.2, random_state=42)

# Print shapes and unique values for debugging
print("\nDebug Information:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Unique values in y_train: {np.unique(y_train)}")

# Compute class weights only if we have at least two classes
unique_classes = np.unique(y_train)
if len(unique_classes) >= 2:
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train
    )
    class_weight_dict = dict(zip(unique_classes, class_weights))
else:
    print("Warning: Less than 2 classes in training data")
    class_weight_dict = None

# Train SVM
svm = SVC(kernel="rbf", class_weight=class_weight_dict)
svm.fit(X_train, y_train)



#Evaluate
y_pred = svm.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Secure Data Storage using AES Encryption
key = Fernet.generate_key()
cipher = Fernet(key)

encrypted_data = cipher.encrypt(b"Sensitive network data")
decrypted_data = cipher.decrypt(encrypted_data)
print("Decrypted:", decrypted_data.decode())


joblib.dump(svm, 'intrusion_detection_model.joblib')
print("joblin create sucessfully")