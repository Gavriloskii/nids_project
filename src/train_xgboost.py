import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import sys

# Add path to use corrected feature extraction
sys.path.append('../src')
from realtime_detection import NetworkIntrusionDetector

def extract_features_from_pcap(pcap_file, detector, max_packets=10000):
    """Extract features from PCAP file using corrected flow-based method"""
    print(f"Processing {pcap_file}...")
    
    # Use the corrected detection method to extract features
    detector.flows = {}  # Reset flows
    detector.completed_flows = []
    
    # Process PCAP file
    alerts = detector.detect_intrusions_from_file(pcap_file, max_packets=max_packets)
    
    # Get the features that were extracted during processing
    # This is a simplified approach - you'll need to modify detect_intrusions_from_file
    # to return the features as well as alerts
    
    return features, labels

# Load data using corrected feature extraction
print("Extracting features using corrected flow-based method...")
detector = NetworkIntrusionDetector(model_type='xgboost')

# Define your PCAP files and their labels
training_data = [
    ('../data/Monday-WorkingHours.pcap', 0),    # Benign
    ('../data/Tuesday-WorkingHours.pcap', 1),   # Contains attacks
    ('../data/Wednesday-WorkingHours.pcap', 1), # Contains attacks
    ('../data/Thursday-WorkingHours.pcap', 0),  # Benign
]

test_data = [
    ('../data/Friday-WorkingHours.pcap', 1),    # Contains attacks
]

# Extract training features
X_train_list = []
y_train_list = []

for pcap_file, label in training_data:
    if os.path.exists(pcap_file):
        features, _ = extract_features_from_pcap(pcap_file, detector, max_packets=5000)
        X_train_list.extend(features)
        y_train_list.extend([label] * len(features))
    else:
        print(f"Warning: {pcap_file} not found")

# Extract test features
X_test_list = []
y_test_list = []

for pcap_file, label in test_data:
    if os.path.exists(pcap_file):
        features, _ = extract_features_from_pcap(pcap_file, detector, max_packets=2000)
        X_test_list.extend(features)
        y_test_list.extend([label] * len(features))

# Convert to numpy arrays
X_train = np.array(X_train_list)
y_train = np.array(y_train_list)
X_test = np.array(X_test_list)
y_test = np.array(y_test_list)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Validate features
print("Validating extracted features...")
constant_features = []
for i in range(X_train.shape[1]):
    if X_train[:, i].std() == 0:
        constant_features.append(i)

if constant_features:
    print(f"WARNING: Found {len(constant_features)} constant features")
else:
    print("✓ No constant features found")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model with updated hyperparameters
print("Training XGBoost model...")
start_time = time.time()

model = xgb.XGBClassifier(
    max_depth=8,
    learning_rate=0.05,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

model.fit(X_train_scaled, y_train, 
          eval_set=[(X_test_scaled, y_test)], 
          verbose=True)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Evaluate model performance
print("Evaluating model performance...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Print prediction probability distribution
print(f"Prediction probabilities - min: {y_pred_proba.min():.4f}, max: {y_pred_proba.max():.4f}, mean: {y_pred_proba.mean():.4f}")

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix (Corrected Features)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Save the trained model and results
os.makedirs('../models/xgboost', exist_ok=True)
model.save_model('../models/xgboost/xgboost_model_corrected.json')
plt.savefig('../models/xgboost/confusion_matrix_corrected.png')

# Save scaler for use in prediction
import joblib
joblib.dump(scaler, '../models/xgboost/scaler.pkl')

# Save metrics to file
with open('../models/xgboost/metrics_corrected.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Training Time: {training_time:.2f} seconds\n")
    f.write(f"Prediction Probability Range: {y_pred_proba.min():.4f} - {y_pred_proba.max():.4f}\n")

print("Model training completed with corrected features and results saved.")
