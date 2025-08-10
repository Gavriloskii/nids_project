import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib


def create_bilstm_model(input_shape, dropout_rate=0.3, lstm_units=128):
    """
    Create BiLSTM model architecture optimized for CIC-IDS 2017 intrusion detection
    """
    model = Sequential([
        # Reshape input for LSTM (samples, timesteps, features)
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        
        # First BiLSTM layer with return sequences
        Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)),
        BatchNormalization(),
        
        # Second BiLSTM layer
        Bidirectional(LSTM(lstm_units//2, dropout=dropout_rate)),
        BatchNormalization(),
        
        # Dense layers for classification
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        
        # Output layer for binary classification
        Dense(1, activation='sigmoid')
    ])
    
    return model


def main():
    """
    Train BiLSTM model using preprocessed CIC-IDS 2017 CSV data.
    This script loads the preprocessed numpy arrays created by preprocessing.py
    and trains a BiLSTM classifier for network intrusion detection.
    """
    
    print("=" * 60)
    print("BiLSTM NETWORK INTRUSION DETECTION TRAINING")
    print("=" * 60)
    
    # Step 1: Load preprocessed data
    print("Loading preprocessed data...")
    try:
        X_train = np.load('data/preprocessed_csv/X_train.npy')
        X_test = np.load('data/preprocessed_csv/X_test.npy')
        y_train = np.load('data/preprocessed_csv/y_train.npy')
        y_test = np.load('data/preprocessed_csv/y_test.npy')
        
        print(f"✅ Data loaded successfully!")
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Training labels - Benign: {(y_train==0).sum()}, Attack: {(y_train==1).sum()}")
        print(f"   Test labels - Benign: {(y_test==0).sum()}, Attack: {(y_test==1).sum()}")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading preprocessed data: {e}")
        print("Make sure you've run 'python src/preprocessing.py' first to generate the preprocessed data.")
        return
    except Exception as e:
        print(f"❌ Unexpected error loading data: {e}")
        return
    
    # Step 2: Load feature names for reference
    try:
        with open('data/preprocessed_csv/feature_names.txt', 'r') as f:
            feature_names = [line.strip().split(': ')[1] if ': ' in line else line.strip() for line in f.readlines()]
        print(f"   Feature names loaded: {len(feature_names)} features")
    except:
        print("   Warning: Could not load feature names, using indices")
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    
    # Step 3: Validate data quality
    print("\nValidating data quality...")
    
    # Check for problematic values
    inf_count = np.isinf(X_train).sum() + np.isinf(X_test).sum()
    nan_count = np.isnan(X_train).sum() + np.isnan(X_test).sum()
    
    if inf_count > 0:
        print(f"   Warning: Found {inf_count} infinite values")
    if nan_count > 0:
        print(f"   Warning: Found {nan_count} NaN values")
    
    # Check feature variance
    feature_variances = np.var(X_train, axis=0)
    informative_features = np.sum(feature_variances > 0)
    print(f"   ✅ Informative features: {informative_features}/{X_train.shape[1]}")
    
    # Step 4: Initialize BiLSTM model
    print("\nInitializing BiLSTM model...")
    input_shape = X_train.shape[1]  # 53 features
    
    model = create_bilstm_model(input_shape, dropout_rate=0.3, lstm_units=128)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    print("   BiLSTM model architecture:")
    model.summary()
    
    # Step 5: Setup training callbacks
    print("\nSetting up training callbacks...")
    
    # Create output directory
    os.makedirs('models/bilstm', exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath='models/bilstm/bilstm_model_corrected.h5',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
    ]
    
    # Step 6: Use the same scaler from preprocessing
    print("\nLoading and applying scaler...")
    try:
        scaler = joblib.load('data/preprocessed_csv/scaler.pkl')
        # Data is already scaled, so we just need to reshape for LSTM
        X_train_scaled = X_train
        X_test_scaled = X_test
        print("   ✅ Using preprocessed scaled data")
    except:
        print("   Warning: Could not load scaler, data should already be scaled")
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Reshape for LSTM input (samples, timesteps, features)
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
    
    print(f"   Reshaped data for LSTM: Train {X_train_reshaped.shape}, Test {X_test_reshaped.shape}")
    
    # Step 7: Train the model
    print("\nTraining BiLSTM model...")
    start_time = time.time()
    
    # Train with validation data
    history = model.fit(
        X_train_reshaped, y_train,
        batch_size=1024,  # Larger batch size for efficiency
        epochs=50,
        validation_data=(X_test_reshaped, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Step 8: Make predictions
    print("\nMaking predictions...")
    y_pred_proba = model.predict(X_test_reshaped, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Step 9: Evaluate model performance
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE EVALUATION")
    print("=" * 50)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"🎯 Overall Performance Metrics:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Prediction probability analysis
    print(f"\n📊 Prediction Probability Analysis:")
    print(f"   Min probability:  {y_pred_proba.min():.4f}")
    print(f"   Max probability:  {y_pred_proba.max():.4f}")
    print(f"   Mean probability: {y_pred_proba.mean():.4f}")
    
    # Probability distribution by threshold
    for threshold in [0.1, 0.2, 0.3, 0.5, 0.8, 0.9]:
        count = (y_pred_proba > threshold).sum()
        print(f"   Predictions > {threshold}: {count} ({count/len(y_pred_proba)*100:.1f}%)")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['Benign', 'Attack'], digits=4)
    print(report)
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n🔍 Confusion Matrix Analysis:")
    print(f"   True Negatives (Benign correctly classified):  {tn:,}")
    print(f"   False Positives (Benign misclassified as Attack): {fp:,}")
    print(f"   False Negatives (Attack misclassified as Benign): {fn:,}")
    print(f"   True Positives (Attack correctly classified):   {tp:,}")
    print(f"   False Positive Rate: {fp/(fp+tn)*100:.2f}%")
    print(f"   False Negative Rate: {fn/(fn+tp)*100:.2f}%")
    
    # Step 10: Create and save visualizations
    print("\n📈 Creating visualizations...")
    
    # Training history plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', color='blue')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Precision plot
    ax3.plot(history.history['precision'], label='Training Precision', color='blue')
    ax3.plot(history.history['val_precision'], label='Validation Precision', color='red')
    ax3.set_title('Model Precision')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Precision')
    ax3.legend()
    ax3.grid(True)
    
    # Recall plot
    ax4.plot(history.history['recall'], label='Training Recall', color='blue')
    ax4.plot(history.history['val_recall'], label='Validation Recall', color='red')
    ax4.set_title('Model Recall')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Recall')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/bilstm/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Attack'], 
                yticklabels=['Benign', 'Attack'],
                cbar_kws={'label': 'Count'})
    plt.title('BiLSTM Confusion Matrix\n(CIC-IDS 2017 Dataset)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add performance metrics to the plot
    plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}', 
                fontsize=10, ha='left')
    
    plt.tight_layout()
    plt.savefig('models/bilstm/confusion_matrix_corrected.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Step 11: Save model and results
    print("\n💾 Saving model and results...")
    
    # Model is already saved by ModelCheckpoint callback
    print(f"   ✅ Model saved to: models/bilstm/bilstm_model_corrected.h5")
    
    # Copy scaler for consistency
    try:
        import shutil
        source_scaler = 'data/preprocessed_csv/scaler.pkl'
        target_scaler = 'models/bilstm/scaler.pkl'
        shutil.copy2(source_scaler, target_scaler)
        print(f"   ✅ Scaler copied to: {target_scaler}")
    except Exception as e:
        print(f"   ⚠️  Could not copy scaler: {e}")
    
    # Save detailed metrics
    metrics_path = 'models/bilstm/metrics_corrected.txt'
    with open(metrics_path, 'w') as f:
        f.write("BiLSTM Network Intrusion Detection Model - Performance Metrics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Training Samples: {X_train.shape[0]:,}\n")
        f.write(f"Test Samples: {X_test.shape[0]:,}\n")
        f.write(f"Features: {X_train.shape[1]}\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"  Accuracy:  {accuracy:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall:    {recall:.4f}\n")
        f.write(f"  F1 Score:  {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  True Negatives:  {tn:,}\n")
        f.write(f"  False Positives: {fp:,}\n")
        f.write(f"  False Negatives: {fn:,}\n")
        f.write(f"  True Positives:  {tp:,}\n\n")
        f.write(f"Prediction Probability Range: {y_pred_proba.min():.4f} - {y_pred_proba.max():.4f}\n")
        f.write(f"Mean Prediction Probability: {y_pred_proba.mean():.4f}\n")
    
    print(f"   ✅ Metrics saved to: {metrics_path}")
    
    # Step 12: Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"🎯 Final Performance: {accuracy*100:.2f}% Accuracy")
    print(f"⚡ Training Time: {training_time:.2f} seconds")
    print(f"📁 Model saved to: models/bilstm/")
    print(f"📊 Visualizations created: training_history.png, confusion_matrix_corrected.png")
    
    if accuracy > 0.97:
        print("🏆 EXCELLENT: Model achieved >97% accuracy!")
    elif accuracy > 0.90:
        print("✅ GOOD: Model achieved >90% accuracy")
    else:
        print("⚠️  Model accuracy could be improved - check data quality and hyperparameters")
    
    print("\n📋 Next Steps:")
    print("   1. Test detection: python src/realtime_detection.py --pcap data/Friday-WorkingHours.pcap --model bilstm --threshold 0.3")
    print("   2. Compare with XGBoost performance")
    print("   3. Tune thresholds for optimal detection")
    
    return model, accuracy, precision, recall, f1


if __name__ == "__main__":
    main()
