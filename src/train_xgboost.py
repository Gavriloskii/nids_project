import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib


def main():
    """
    Train XGBoost model using preprocessed CIC-IDS 2017 CSV data.
    This script loads the preprocessed numpy arrays created by preprocessing.py
    and trains an XGBoost classifier for network intrusion detection.
    """
    
    print("=" * 60)
    print("XGBOOST NETWORK INTRUSION DETECTION TRAINING")
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
    
    if informative_features < 30:
        print(f"   ⚠️  Warning: Only {informative_features} features have variance")
    
    # Step 4: Initialize XGBoost model with optimized hyperparameters
    print("\nInitializing XGBoost model...")
    model = xgb.XGBClassifier(
        max_depth=8,                    # Deep enough for complex patterns
        learning_rate=0.05,             # Conservative learning rate
        n_estimators=200,               # More trees for better performance
        subsample=0.8,                  # Row subsampling to prevent overfitting
        colsample_bytree=0.8,          # Column subsampling
        objective='binary:logistic',    # Binary classification
        eval_metric='logloss',         # Evaluation metric
        use_label_encoder=False,       # Suppress warning
        random_state=42,               # Reproducibility
        n_jobs=-1,                     # Use all CPU cores
        tree_method='hist',            # Fast histogram-based method
        grow_policy='lossguide',       # Loss-guided splitting
    )
    
    print("   XGBoost hyperparameters:")
    print(f"     Max depth: {model.max_depth}")
    print(f"     Learning rate: {model.learning_rate}")
    print(f"     N estimators: {model.n_estimators}")
    print(f"     Subsample: {model.subsample}")
    
    # Step 5: Train the model
    print("\nTraining XGBoost model...")
    start_time = time.time()
    
    # Train with evaluation set for monitoring
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False  # Set to True to see training progress
    )
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Step 6: Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Step 7: Evaluate model performance
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
    for threshold in [0.1, 0.2, 0.5, 0.8, 0.9]:
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
    
    # Feature importance analysis
    print(f"\n🔥 Top 10 Most Important Features:")
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    for i in range(min(10, len(importance_df))):
        feat_name = importance_df.iloc[i]['Feature']
        importance = importance_df.iloc[i]['Importance']
        print(f"   {i+1:2d}. {feat_name:<30}: {importance:.4f}")
    
    # Step 8: Create and save visualizations
    print("\n📈 Creating visualizations...")
    
    # Create output directory
    os.makedirs('models/xgboost', exist_ok=True)
    
    # Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Attack'], 
                yticklabels=['Benign', 'Attack'],
                cbar_kws={'label': 'Count'})
    plt.title('XGBoost Confusion Matrix\n(CIC-IDS 2017 Dataset)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add performance metrics to the plot
    plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}', 
                fontsize=10, ha='left')
    
    plt.tight_layout()
    plt.savefig('models/xgboost/confusion_matrix_corrected.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature Importance Plot
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'], color='skyblue')
    plt.yticks(range(len(top_features)), [name[:30] + '...' if len(name) > 30 else name 
                                         for name in top_features['Feature']])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top 15 Feature Importances (XGBoost)', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('models/xgboost/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Step 9: Save model and results
    print("\n💾 Saving model and results...")
    
    # Save the trained model
    model_path = 'models/xgboost/xgboost_model_corrected.json'
    model.save_model(model_path)
    print(f"   ✅ Model saved to: {model_path}")
    
    # Save scaler (copy from preprocessing directory for consistency)
    try:
        import shutil
        source_scaler = 'data/preprocessed_csv/scaler.pkl'
        target_scaler = 'models/xgboost/scaler.pkl'
        shutil.copy2(source_scaler, target_scaler)
        print(f"   ✅ Scaler copied to: {target_scaler}")
    except Exception as e:
        print(f"   ⚠️  Could not copy scaler: {e}")
    
    # Save detailed metrics
    metrics_path = 'models/xgboost/metrics_corrected.txt'
    with open(metrics_path, 'w') as f:
        f.write("XGBoost Network Intrusion Detection Model - Performance Metrics\n")
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
        f.write(f"Mean Prediction Probability: {y_pred_proba.mean():.4f}\n\n")
        f.write("Top 10 Feature Importances:\n")
        for i in range(min(10, len(importance_df))):
            feat_name = importance_df.iloc[i]['Feature']
            importance = importance_df.iloc[i]['Importance']
            f.write(f"  {i+1:2d}. {feat_name}: {importance:.4f}\n")
    
    print(f"   ✅ Metrics saved to: {metrics_path}")
    
    # Save feature importance data
    importance_path = 'models/xgboost/feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"   ✅ Feature importance saved to: {importance_path}")
    
    # Step 10: Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"🎯 Final Performance: {accuracy*100:.2f}% Accuracy")
    print(f"⚡ Training Time: {training_time:.2f} seconds")
    print(f"📁 Model saved to: models/xgboost/")
    print(f"📊 Visualizations created: confusion_matrix_corrected.png, feature_importance.png")
    
    if accuracy > 0.97:
        print("🏆 EXCELLENT: Model achieved >97% accuracy!")
    elif accuracy > 0.90:
        print("✅ GOOD: Model achieved >90% accuracy")
    else:
        print("⚠️  Model accuracy could be improved - check data quality and hyperparameters")
    
    print("\n📋 Next Steps:")
    print("   1. Test detection: python src/realtime_detection.py --pcap data/Friday-WorkingHours.pcap --model xgboost --threshold 0.1")
    print("   2. Train BiLSTM: python src/train_bilstm.py")
    print("   3. Compare model performance and tune thresholds")
    
    return model, accuracy, precision, recall, f1


if __name__ == "__main__":
    main()
