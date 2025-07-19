import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import sys
import pyshark
from collections import defaultdict

# Add path to use corrected feature extraction
sys.path.append('../src')

class BiLSTMTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.flows = {}
        self.flow_timeout = 120
        
    def get_flow_key(self, packet):
        """Generate unique flow identifier for bidirectional flow tracking"""
        try:
            if hasattr(packet, 'ip'):
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
                
                if hasattr(packet, 'tcp'):
                    src_port = int(packet.tcp.srcport)
                    dst_port = int(packet.tcp.dstport)
                    protocol = 'TCP'
                elif hasattr(packet, 'udp'):
                    src_port = int(packet.udp.srcport)
                    dst_port = int(packet.udp.dstport)
                    protocol = 'UDP'
                else:
                    return (min(src_ip, dst_ip), max(src_ip, dst_ip), packet.highest_layer), 'forward', (src_ip, 0), (dst_ip, 0)
                
                # Create bidirectional flow key
                endpoint1 = (src_ip, src_port)
                endpoint2 = (dst_ip, dst_port)
                if endpoint1 < endpoint2:
                    flow_key = (endpoint1, endpoint2, protocol)
                    direction = 'forward'
                else:
                    flow_key = (endpoint2, endpoint1, protocol)
                    direction = 'backward'
                
                return flow_key, direction, (src_ip, src_port), (dst_ip, dst_port)
        except Exception as e:
            print(f"Error creating flow key: {e}")
            return None, None, None, None
        
        return None, None, None, None
    
    def update_flow(self, packet):
        """Update flow statistics with new packet"""
        flow_info = self.get_flow_key(packet)
        if flow_info[0] is None:
            return None
            
        flow_key, direction, src_endpoint, dst_endpoint = flow_info
        current_time = float(packet.sniff_timestamp)
        packet_length = int(packet.length) if hasattr(packet, 'length') else 0
        
        # Initialize flow if new
        if flow_key not in self.flows:
            self.flows[flow_key] = {
                'start_time': current_time,
                'last_time': current_time,
                'packets': [],
                'fwd_packets': 0,
                'bwd_packets': 0,
                'fwd_bytes': 0,
                'bwd_bytes': 0,
                'fwd_lengths': [],
                'bwd_lengths': [],
                'timestamps': [],
                'tcp_flags': {
                    'fin': 0, 'syn': 0, 'rst': 0, 'psh': 0, 'ack': 0, 'urg': 0, 'cwe': 0, 'ece': 0
                },
                'fwd_header_lengths': [],
                'bwd_header_lengths': [],
                'first_endpoint': src_endpoint,
                'second_endpoint': dst_endpoint
            }
        
        flow = self.flows[flow_key]
        flow['last_time'] = current_time
        flow['timestamps'].append(current_time)
        
        # Update direction-specific statistics
        if direction == 'forward':
            flow['fwd_packets'] += 1
            flow['fwd_bytes'] += packet_length
            flow['fwd_lengths'].append(packet_length)
            if hasattr(packet, 'tcp'):
                flow['fwd_header_lengths'].append(int(getattr(packet.tcp, 'hdr_len', 20)))
        else:
            flow['bwd_packets'] += 1
            flow['bwd_bytes'] += packet_length
            flow['bwd_lengths'].append(packet_length)
            if hasattr(packet, 'tcp'):
                flow['bwd_header_lengths'].append(int(getattr(packet.tcp, 'hdr_len', 20)))
        
        # Update TCP flags
        if hasattr(packet, 'tcp'):
            def safe_flag(flag_name):
                val = getattr(packet.tcp, flag_name, False)
                if isinstance(val, bool):
                    return int(val)
                if isinstance(val, int):
                    return val
                if isinstance(val, str):
                    return 1 if val.lower() == 'true' else 0
                return 0
            
            flow['tcp_flags']['fin'] += safe_flag('flags_fin')
            flow['tcp_flags']['syn'] += safe_flag('flags_syn')
            flow['tcp_flags']['rst'] += safe_flag('flags_reset')
            flow['tcp_flags']['psh'] += safe_flag('flags_push')
            flow['tcp_flags']['ack'] += safe_flag('flags_ack')
            flow['tcp_flags']['urg'] += safe_flag('flags_urg')
        
        return flow_key
    
    def extract_flow_features(self, flow_data):
        """Extract comprehensive flow features for ML model"""
        try:
            features = {}
            
            # Basic flow information
            duration = max(0.000001, flow_data['last_time'] - flow_data['start_time'])
            total_packets = len(flow_data['packets']) if 'packets' in flow_data else flow_data['fwd_packets'] + flow_data['bwd_packets']
            total_bytes = flow_data['fwd_bytes'] + flow_data['bwd_bytes']
            
            # Get destination port
            dest_port = 0
            if flow_data['fwd_packets'] > 0:
                dest_port = flow_data['second_endpoint'][1] if len(flow_data['second_endpoint']) > 1 else 0
            
            # Basic features
            features[' Destination Port'] = dest_port
            features[' Flow Duration'] = duration * 1000000  # Convert to microseconds
            features[' Total Fwd Packets'] = flow_data['fwd_packets']
            features[' Total Backward Packets'] = flow_data['bwd_packets']
            features['Total Length of Fwd Packets'] = flow_data['fwd_bytes']
            features[' Total Length of Bwd Packets'] = flow_data['bwd_bytes']
            
            # Forward packet length statistics
            if flow_data['fwd_lengths']:
                features[' Fwd Packet Length Max'] = max(flow_data['fwd_lengths'])
                features[' Fwd Packet Length Min'] = min(flow_data['fwd_lengths'])
                features[' Fwd Packet Length Mean'] = np.mean(flow_data['fwd_lengths'])
                features[' Fwd Packet Length Std'] = np.std(flow_data['fwd_lengths'])
            else:
                features[' Fwd Packet Length Max'] = 0
                features[' Fwd Packet Length Min'] = 0
                features[' Fwd Packet Length Mean'] = 0
                features[' Fwd Packet Length Std'] = 0
            
            # Backward packet length statistics
            if flow_data['bwd_lengths']:
                features['Bwd Packet Length Max'] = max(flow_data['bwd_lengths'])
                features[' Bwd Packet Length Min'] = min(flow_data['bwd_lengths'])
                features[' Bwd Packet Length Mean'] = np.mean(flow_data['bwd_lengths'])
                features[' Bwd Packet Length Std'] = np.std(flow_data['bwd_lengths'])
            else:
                features['Bwd Packet Length Max'] = 0
                features[' Bwd Packet Length Min'] = 0
                features[' Bwd Packet Length Mean'] = 0
                features[' Bwd Packet Length Std'] = 0
            
            # Flow rate features
            features['Flow Bytes/s'] = total_bytes / duration
            features[' Flow Packets/s'] = total_packets / duration
            
            # Inter-arrival time features
            if len(flow_data['timestamps']) > 1:
                iats = [flow_data['timestamps'][i+1] - flow_data['timestamps'][i] 
                       for i in range(len(flow_data['timestamps'])-1)]
                features[' Flow IAT Mean'] = np.mean(iats) * 1000000
                features[' Flow IAT Std'] = np.std(iats) * 1000000
                features[' Flow IAT Max'] = max(iats) * 1000000
                features[' Flow IAT Min'] = min(iats) * 1000000
            else:
                features[' Flow IAT Mean'] = 0
                features[' Flow IAT Std'] = 0
                features[' Flow IAT Max'] = 0
                features[' Flow IAT Min'] = 0
            
            # Simplified IAT features for forward and backward
            features['Fwd IAT Total'] = 0
            features[' Fwd IAT Mean'] = 0
            features[' Fwd IAT Std'] = 0
            features[' Fwd IAT Max'] = 0
            features[' Fwd IAT Min'] = 0
            features['Bwd IAT Total'] = 0
            features[' Bwd IAT Mean'] = 0
            features[' Bwd IAT Std'] = 0
            features[' Bwd IAT Max'] = 0
            features[' Bwd IAT Min'] = 0
            
            # TCP flag features
            tcp_flags = flow_data['tcp_flags']
            features['Fwd PSH Flags'] = tcp_flags['psh']
            features[' Bwd PSH Flags'] = 0
            features[' Fwd URG Flags'] = tcp_flags['urg']
            features[' Bwd URG Flags'] = 0
            features['FIN Flag Count'] = tcp_flags['fin']
            features[' SYN Flag Count'] = tcp_flags['syn']
            features[' RST Flag Count'] = tcp_flags['rst']
            features[' PSH Flag Count'] = tcp_flags['psh']
            features[' ACK Flag Count'] = tcp_flags['ack']
            features[' URG Flag Count'] = tcp_flags['urg']
            features[' CWE Flag Count'] = tcp_flags['cwe']
            features[' ECE Flag Count'] = tcp_flags['ece']
            
            # Header length features
            if flow_data['fwd_header_lengths']:
                features[' Fwd Header Length'] = np.mean(flow_data['fwd_header_lengths'])
            else:
                features[' Fwd Header Length'] = 0
                
            if flow_data['bwd_header_lengths']:
                features[' Bwd Header Length'] = np.mean(flow_data['bwd_header_lengths'])
            else:
                features[' Bwd Header Length'] = 0
            
            # Packet rate features
            features['Fwd Packets/s'] = flow_data['fwd_packets'] / duration
            features[' Bwd Packets/s'] = flow_data['bwd_packets'] / duration
            
            # Packet length statistics (overall)
            all_lengths = flow_data['fwd_lengths'] + flow_data['bwd_lengths']
            if all_lengths:
                features[' Min Packet Length'] = min(all_lengths)
                features[' Max Packet Length'] = max(all_lengths)
                features[' Packet Length Mean'] = np.mean(all_lengths)
                features[' Packet Length Std'] = np.std(all_lengths)
                features[' Packet Length Variance'] = np.var(all_lengths)
                features[' Average Packet Size'] = np.mean(all_lengths)
            else:
                features[' Min Packet Length'] = 0
                features[' Max Packet Length'] = 0
                features[' Packet Length Mean'] = 0
                features[' Packet Length Std'] = 0
                features[' Packet Length Variance'] = 0
                features[' Average Packet Size'] = 0
            
            # Down/Up ratio
            if flow_data['fwd_bytes'] > 0:
                features[' Down/Up Ratio'] = flow_data['bwd_bytes'] / flow_data['fwd_bytes']
            else:
                features[' Down/Up Ratio'] = 0
            
            return features
            
        except Exception as e:
            print(f"Error extracting flow features: {e}")
            return {}
    
    def get_completed_flows(self, force_completion=False):
        """Get flows that are ready for analysis"""
        current_time = time.time()
        completed_flows = []
        flows_to_remove = []
        
        for flow_key, flow_data in self.flows.items():
            time_since_last = current_time - flow_data['last_time']
            has_enough_packets = (flow_data['fwd_packets'] + flow_data['bwd_packets']) >= 2
            
            if (time_since_last > self.flow_timeout or force_completion) and has_enough_packets:
                features = self.extract_flow_features(flow_data)
                if features:
                    completed_flows.append((flow_key, features, flow_data))
                flows_to_remove.append(flow_key)
        
        # Remove completed flows
        for flow_key in flows_to_remove:
            del self.flows[flow_key]
        
        return completed_flows
    
    def extract_features_from_pcap(self, pcap_file, label, max_packets=10000):
        """Extract features from PCAP file using corrected flow-based method"""
        print(f"Processing {pcap_file} (label: {label})...")
        
        if not os.path.exists(pcap_file):
            print(f"Warning: {pcap_file} not found")
            return [], []
        
        # Reset flows
        self.flows = {}
        
        # Use FileCapture to process PCAP
        capture = pyshark.FileCapture(pcap_file)
        packet_count = 0
        flow_features = []
        flow_labels = []
        
        try:
            for packet in capture:
                packet_count += 1
                
                if packet_count > max_packets:
                    print(f"Reached maximum packet limit ({max_packets}). Stopping analysis.")
                    break
                    
                if packet_count % 1000 == 0:
                    print(f"  Processed {packet_count} packets...")
                
                # Update flow with packet
                self.update_flow(packet)
                
                # Check for completed flows periodically
                if packet_count % 500 == 0:
                    completed_flows = self.get_completed_flows()
                    for flow_key, features, flow_data in completed_flows:
                        flow_features.append(features)
                        flow_labels.append(label)
            
            # Process remaining flows
            completed_flows = self.get_completed_flows(force_completion=True)
            for flow_key, features, flow_data in completed_flows:
                flow_features.append(features)
                flow_labels.append(label)
        
        except Exception as e:
            print(f"Error processing {pcap_file}: {e}")
        finally:
            capture.close()
        
        print(f"  Extracted {len(flow_features)} flows from {packet_count} packets")
        return flow_features, flow_labels
    
    def get_required_features(self):
        """Return the list of features required by the model in the correct order"""
        return [
            ' Destination Port', ' Flow Duration', ' Total Fwd Packets', 
            ' Total Backward Packets', 'Total Length of Fwd Packets', 
            ' Total Length of Bwd Packets', ' Fwd Packet Length Max', 
            ' Fwd Packet Length Min', ' Fwd Packet Length Mean', 
            ' Fwd Packet Length Std', 'Bwd Packet Length Max', 
            ' Bwd Packet Length Min', ' Bwd Packet Length Mean', 
            ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', 
            ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', 
            ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', 
            ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 
            'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', 
            ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', 
            ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', 
            ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', 
            ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length', 
            ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', 
            'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', 
            ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count', 
            ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', 
            ' Average Packet Size'
        ]
    
    def preprocess_features(self, features_list):
        """Preprocess extracted features to match the format used in training"""
        if not features_list:
            return np.array([])
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # Get the required features list
        required_features = self.get_required_features()
        
        # Fill missing values with 0
        for col in required_features:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the required features in the correct order
        try:
            X = df[required_features]
        except KeyError as e:
            missing_cols = [col for col in required_features if col not in df.columns]
            print(f"Error: Missing columns: {missing_cols}")
            raise ValueError(f"Missing required features: {missing_cols}")
        
        return X.values

def main():
    # Initialize trainer
    trainer = BiLSTMTrainer()
    
    # Define your PCAP files and their labels
    # Adjust these paths and labels based on your actual data
    training_data = [
        ('../data/Monday-WorkingHours.pcap', 0),    # Benign
        ('../data/Tuesday-WorkingHours.pcap', 1),   # Contains attacks
        ('../data/Wednesday-WorkingHours.pcap', 1), # Contains attacks
        ('../data/Thursday-WorkingHours.pcap', 0),  # Benign
    ]
    
    test_data = [
        ('../data/Friday-WorkingHours.pcap', 1),    # Contains attacks
    ]
    
    print("Extracting features using corrected flow-based method...")
    
    # Extract training features
    X_train_list = []
    y_train_list = []
    
    for pcap_file, label in training_data:
        features, labels = trainer.extract_features_from_pcap(pcap_file, label, max_packets=5000)
        X_train_list.extend(features)
        y_train_list.extend(labels)
    
    # Extract test features
    X_test_list = []
    y_test_list = []
    
    for pcap_file, label in test_data:
        features, labels = trainer.extract_features_from_pcap(pcap_file, label, max_packets=2000)
        X_test_list.extend(features)
        y_test_list.extend(labels)
    
    if not X_train_list or not X_test_list:
        print("Error: No features extracted. Please check your PCAP files.")
        return
    
    # Convert to numpy arrays
    X_train = trainer.preprocess_features(X_train_list)
    y_train = np.array(y_train_list)
    X_test = trainer.preprocess_features(X_test_list)
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
    
    # Print feature statistics
    print("Feature value ranges (first 10 features):")
    for i in range(min(10, X_train.shape[1])):
        print(f"  Feature {i}: min={X_train[:, i].min():.4f}, max={X_train[:, i].max():.4f}, mean={X_train[:, i].mean():.4f}")
    
    # Scale features
    X_train_scaled = trainer.scaler.fit_transform(X_train)
    X_test_scaled = trainer.scaler.transform(X_test)
    
    # Reshape input data for LSTM (samples, timesteps, features)
    # For BiLSTM, we treat each feature as a timestep
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
    
    # Create BiLSTM model with improved architecture
    print("Creating BiLSTM model...")
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train_scaled.shape[1], 1)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Define callbacks
    os.makedirs('../models/bilstm', exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath='../models/bilstm/bilstm_model_corrected.h5',
            save_best_only=True,
            monitor='val_loss'
        )
    ]
    
    # Train model
    print("Training BiLSTM model...")
    start_time = time.time()
    
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=20,
        batch_size=512,
        validation_data=(X_test_reshaped, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model performance
    print("Evaluating model performance...")
    y_pred_prob = model.predict(X_test_reshaped)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print prediction probability distribution
    print(f"Prediction probabilities - min: {y_pred_prob.min():.4f}, max: {y_pred_prob.max():.4f}, mean: {y_pred_prob.mean():.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('BiLSTM Confusion Matrix (Corrected Features)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('../models/bilstm/confusion_matrix_corrected.png')
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('../models/bilstm/training_history_corrected.png')
    
    # Save scaler for use in prediction
    import joblib
    joblib.dump(trainer.scaler, '../models/bilstm/scaler.pkl')
    
    # Save metrics to file
    with open('../models/bilstm/metrics_corrected.txt', 'w', encoding='utf-8') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")  # Fixed typo
        f.write(f"Prediction Probability Range: {y_pred_prob.min():.4f} - {y_pred_prob.max():.4f}\n")
    
    print("BiLSTM model training completed with corrected features and results saved.")

if __name__ == "__main__":
    main()
