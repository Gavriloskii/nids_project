import pandas as pd
import numpy as np
import pyshark
import os
import time
import joblib
import glob
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import sys

# Add path to use corrected feature extraction
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Required features in exact order expected by your models (53 features)
REQUIRED_FEATURES = [
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
    ' Packet Length Mean', ' Packet Length Std',
    ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count',
    ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count',
    ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count',
    ' Down/Up Ratio', ' Average Packet Size'
]


class FlowBasedPreprocessor:
    def __init__(self):
        """Initialize the flow-based preprocessor for network intrusion detection"""
        self.flows = {}
        self.flow_timeout = 120  # Flow timeout in seconds
        self.scaler = StandardScaler()
        
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
            total_packets = flow_data['fwd_packets'] + flow_data['bwd_packets']
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
        """Extract features from PCAP file using flow-based method"""
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
                    print(f"  Reached maximum packet limit ({max_packets}). Stopping analysis.")
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
        return REQUIRED_FEATURES
    
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


def balance_training_set(X, y):
    """Balance minority class by upsampling to match majority class"""
    print("Balancing training dataset...")
    
    maj_mask = (y == 0)  # Benign
    min_mask = (y == 1)  # Attack
    
    maj_count = maj_mask.sum()
    min_count = min_mask.sum()
    
    print(f"  Original - Benign: {maj_count}, Attack: {min_count}")
    
    if min_count == 0 or maj_count == 0:
        print("  Warning: One class is missing, cannot balance")
        return X, y
    
    # Upsample minority class to match majority
    if min_count < maj_count:
        X_min_upsampled, y_min_upsampled = resample(
            X[min_mask], y[min_mask],
            replace=True,
            n_samples=maj_count,
            random_state=42
        )
        
        # Combine majority + upsampled minority
        X_balanced = np.vstack([X[maj_mask], X_min_upsampled])
        y_balanced = np.hstack([y[maj_mask], y_min_upsampled])
    else:
        # If attack samples >= benign samples, no need to upsample
        X_balanced, y_balanced = X, y
    
    # Shuffle the balanced dataset
    shuffle_idx = np.random.RandomState(42).permutation(len(y_balanced))
    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]
    
    print(f"  Balanced - Benign: {(y_balanced == 0).sum()}, Attack: {(y_balanced == 1).sum()}")
    return X_balanced, y_balanced


def clean_and_validate_data(df):
    """Clean and validate the dataset"""
    print("Cleaning and validating data...")
    
    original_shape = df.shape
    print(f"  Original shape: {original_shape}")
    
    # Replace infinite values with NaN, then fill with 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Remove completely duplicate rows
    before_dedup = len(df)
    df = df.drop_duplicates()
    after_dedup = len(df)
    if before_dedup != after_dedup:
        print(f"  Removed {before_dedup - after_dedup} duplicate rows")
    
    print(f"  Final clean shape: {df.shape}")
    return df


def load_and_combine_csvs(csv_dir):
    """Load and combine all CSV files from the directory"""
    print(f"Loading CSV files from: {csv_dir}")
    
    # Find all CSV files
    csv_pattern = os.path.join(csv_dir, '*.csv')
    csv_paths = glob.glob(csv_pattern)
    
    if not csv_paths:
        raise FileNotFoundError(f'No CSV files found in {csv_dir}')
    
    print(f"Found {len(csv_paths)} CSV files:")
    for path in csv_paths:
        print(f"  - {os.path.basename(path)}")
    
    # Load and combine all CSV files
    dataframes = []
    total_rows = 0
    
    for csv_path in csv_paths:
        try:
            print(f"  Loading {os.path.basename(csv_path)}...")
            
            # Load with appropriate settings for CIC-IDS files
            df = pd.read_csv(csv_path, low_memory=False, encoding='utf-8')
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            dataframes.append(df)
            total_rows += len(df)
            
            print(f"    Shape: {df.shape}")
            
        except Exception as e:
            print(f"    ERROR loading {csv_path}: {e}")
            continue
    
    if not dataframes:
        raise ValueError("No CSV files could be loaded successfully")
    
    # Combine all dataframes
    print(f"Combining {len(dataframes)} dataframes...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Total rows loaded: {total_rows}")
    
    return combined_df


def create_binary_labels(df):
    """Create binary labels from the Label column"""
    print("Creating binary labels...")
    
    # Find the label column (handle different naming conventions)
    label_col = None
    possible_label_cols = ['Label', ' Label', 'label', 'LABEL']
    
    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError(f"No label column found. Available columns: {list(df.columns)}")
    
    print(f"  Using label column: '{label_col}'")
    
    # Drop rows with missing labels
    original_len = len(df)
    df = df.dropna(subset=[label_col])
    after_drop = len(df)
    
    if original_len != after_drop:
        print(f"  Dropped {original_len - after_drop} rows with missing labels")
    
    # Show original label distribution
    print("  Original label distribution:")
    label_counts = df[label_col].value_counts()
    for label, count in label_counts.head(10).items():  # Show top 10
        print(f"    {label}: {count:,}")
    
    # Create binary labels: BENIGN/Normal = 0, everything else = 1
    benign_labels = ['benign', 'normal', 'BENIGN', 'NORMAL', 'Benign', 'Normal']
    df['binary_label'] = df[label_col].apply(
        lambda x: 0 if str(x).strip() in benign_labels else 1
    )
    
    # Show binary distribution
    binary_counts = df['binary_label'].value_counts()
    print("  Binary label distribution:")
    print(f"    Benign (0): {binary_counts.get(0, 0):,}")
    print(f"    Attack (1): {binary_counts.get(1, 0):,}")
    
    return df


def extract_features(df):
    """Extract the required 53 features with proper column mapping"""
    print("Extracting required features...")
    
    # Map CSV columns (0-52) directly to required features
    csv_feature_names = [
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
        ' Packet Length Mean', ' Packet Length Std',
        ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count',
        ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count',
        ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count',
        ' Down/Up Ratio', ' Average Packet Size'
    ]
    
    # Extract features directly from CSV columns 0-52
    X = df.iloc[:, 0:53].values.astype(np.float32)
    y = df['binary_label'].values.astype(np.int32)
    
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Label vector shape: {y.shape}")
    
    # Check for problematic values
    inf_count = np.isinf(X).sum()
    nan_count = np.isnan(X).sum()
    
    if inf_count > 0:
        print(f"  Warning: Found {inf_count} infinite values, replacing with 0")
        X[np.isinf(X)] = 0
    
    if nan_count > 0:
        print(f"  Warning: Found {nan_count} NaN values, replacing with 0")
        X[np.isnan(X)] = 0
    
    # Check for constant features
    feature_variances = np.var(X, axis=0)
    non_constant_features = np.sum(feature_variances > 0)
    
    print(f"  ✅ Features with variance > 0: {non_constant_features}/53")
    
    if non_constant_features < 30:
        print(f"  ⚠️  Warning: Only {non_constant_features} features have variance")
    else:
        print(f"  ✅ Good feature diversity: {non_constant_features} informative features")
    
    return X, y


def preprocess_csv_data(csv_dir, output_dir=None, test_size=0.2, random_state=42):
    """
    Main preprocessing function for CIC-IDS 2017 CSV files
    
    Parameters:
    - csv_dir: Directory containing CSV files
    - output_dir: Directory to save preprocessed data
    - test_size: Fraction for test set (default: 0.2)
    - random_state: Random seed for reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test: Preprocessed datasets
    """
    
    print("="*60)
    print("CIC-IDS 2017 CSV PREPROCESSING")
    print("="*60)
    
    # Step 1: Load and combine CSV files
    df = load_and_combine_csvs(csv_dir)
    
    # Step 2: Clean and validate data
    df = clean_and_validate_data(df)
    
    # Step 3: Create binary labels
    df = create_binary_labels(df)
    
    # Step 4: Extract features
    X, y = extract_features(df)
    
    # Step 5: Train/test split (stratified to maintain class balance)
    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    print(f"  Train set: {X_train.shape[0]:,} samples")
    print(f"  Test set: {X_test.shape[0]:,} samples")
    print(f"  Train labels - Benign: {(y_train==0).sum():,}, Attack: {(y_train==1).sum():,}")
    print(f"  Test labels  - Benign: {(y_test==0).sum():,}, Attack: {(y_test==1).sum():,}")
    
    # Step 6: Balance training set (BEFORE scaling)
    X_train, y_train = balance_training_set(X_train, y_train)
    
    # Step 7: Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("  Scaling completed")
    print(f"  Train set scaled shape: {X_train_scaled.shape}")
    print(f"  Test set scaled shape: {X_test_scaled.shape}")
    
    # Step 8: Final validation
    print("Final validation...")
    print(f"  Feature ranges (train): min={X_train_scaled.min():.3f}, max={X_train_scaled.max():.3f}")
    print(f"  No infinite values: {not np.isinf(X_train_scaled).any()}")
    print(f"  No NaN values: {not np.isnan(X_train_scaled).any()}")
    
    # Step 9: Save preprocessed data
    if output_dir:
        save_preprocessed_data(
            X_train_scaled, X_test_scaled, y_train, y_test, 
            scaler, output_dir, random_state
        )
    
    print("="*60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test, scaler, output_dir, random_state):
    """Save all preprocessed data and metadata"""
    print(f"Saving preprocessed data to: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numpy arrays
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save the fitted scaler (CRITICAL for inference!)
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Scaler saved to: {scaler_path}")
    
    # Save feature names
    feature_file = os.path.join(output_dir, 'feature_names.txt')
    with open(feature_file, 'w') as f:
        for i, feature in enumerate(REQUIRED_FEATURES):
            f.write(f"{i:02d}: {feature}\n")
    
    # Save metadata
    metadata = {
        'created_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'preprocessing_version': '2.0',
        'total_features': len(REQUIRED_FEATURES),
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'train_benign': int((y_train == 0).sum()),
        'train_attack': int((y_train == 1).sum()),
        'test_benign': int((y_test == 0).sum()),
        'test_attack': int((y_test == 1).sum()),
        'test_size': X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]),
        'random_state': random_state,
        'feature_scaling': 'StandardScaler',
        'class_balancing': 'Upsample minority class in training set',
        'data_source': 'CIC-IDS 2017 CSV files'
    }
    
    metadata_file = os.path.join(output_dir, 'preprocessing_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("  Files saved:")
    print("    ✓ X_train.npy, X_test.npy")
    print("    ✓ y_train.npy, y_test.npy")
    print("    ✓ scaler.pkl")
    print("    ✓ feature_names.txt")
    print("    ✓ preprocessing_metadata.json")


def preprocess_pcap_data(pcap_files_with_labels, output_dir=None, test_size=0.2, max_packets_per_file=10000):
    """
    Preprocess PCAP files for intrusion detection using flow-based feature extraction.
    
    Parameters:
    - pcap_files_with_labels: List of tuples (pcap_file_path, label)
    - output_dir: Directory to save preprocessed data (optional)
    - test_size: Fraction of data to use for testing
    - max_packets_per_file: Maximum packets to process per PCAP file
    
    Returns:
    - X_train, X_test, y_train, y_test: Preprocessed and split datasets
    """
    print("Starting flow-based preprocessing of PCAP files...")
    
    # Initialize preprocessor
    preprocessor = FlowBasedPreprocessor()
    
    # Extract features from all PCAP files
    all_features = []
    all_labels = []
    
    for pcap_file, label in pcap_files_with_labels:
        if not os.path.exists(pcap_file):
            print(f"Warning: {pcap_file} not found, skipping...")
            continue
            
        features, labels = preprocessor.extract_features_from_pcap(
            pcap_file, label, max_packets=max_packets_per_file
        )
        all_features.extend(features)
        all_labels.extend(labels)
    
    if not all_features:
        print("Error: No features extracted from PCAP files")
        return None, None, None, None
    
    print(f"Total flows extracted: {len(all_features)}")
    print(f"Label distribution - Benign: {all_labels.count(0)}, Attack: {all_labels.count(1)}")
    
    # Convert features to numpy array
    X = preprocessor.preprocess_features(all_features)
    y = np.array(all_labels)
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Validate features
    print("Validating extracted features...")
    constant_features = []
    for i in range(X.shape[1]):
        if X[:, i].std() == 0:
            constant_features.append(i)
    
    if constant_features:
        print(f"WARNING: Found {len(constant_features)} constant features: {constant_features}")
    else:
        print("✓ No constant features found")
    
    # Check for extreme values
    print("Checking for extreme values...")
    extreme_features = []
    for i in range(X.shape[1]):
        if np.any(np.abs(X[:, i]) > 1e9):
            extreme_features.append(i)
    
    if extreme_features:
        print(f"WARNING: Found extreme values in features: {extreme_features}")
        # Replace extreme values with median
        for i in extreme_features:
            median_val = np.median(X[:, i])
            X[X[:, i] > 1e9, i] = median_val
            X[X[:, i] < -1e9, i] = median_val
    else:
        print("✓ No extreme values found")
    
    # Split data into training and testing sets (before balancing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Train/test split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Balance the training data (before scaling)
    X_train_balanced, y_train_balanced = balance_training_set(X_train, y_train)
    
    # Scale features (fit on balanced training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Final training set shape: {X_train_scaled.shape}")
    print(f"Final testing set shape: {X_test_scaled.shape}")
    print(f"Attack samples in training: {sum(y_train_balanced)}/{len(y_train_balanced)} ({sum(y_train_balanced)/len(y_train_balanced)*100:.2f}%)")
    print(f"Attack samples in testing: {sum(y_test)}/{len(y_test)} ({sum(y_test)/len(y_test)*100:.2f}%)")
    
    # Save preprocessed data if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save arrays
        np.save(os.path.join(output_dir, 'X_train_flow_based.npy'), X_train_scaled)
        np.save(os.path.join(output_dir, 'X_test_flow_based.npy'), X_test_scaled)
        np.save(os.path.join(output_dir, 'y_train_flow_based.npy'), y_train_balanced)
        np.save(os.path.join(output_dir, 'y_test_flow_based.npy'), y_test)
        
        # Save scaler for prediction consistency
        joblib.dump(scaler, os.path.join(output_dir, 'scaler_flow_based.pkl'))
        
        # Save feature names for reference
        feature_names = preprocessor.get_required_features()
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            for i, name in enumerate(feature_names):
                f.write(f"{i}: {name}\n")
        
        # Save preprocessing metadata
        metadata = {
            'total_flows': len(all_features),
            'feature_count': X.shape[1],
            'train_samples': X_train_scaled.shape[0],
            'test_samples': X_test_scaled.shape[0],
            'attack_ratio_train': sum(y_train_balanced) / len(y_train_balanced),
            'attack_ratio_test': sum(y_test) / len(y_test),
            'constant_features': constant_features,
            'extreme_features': extreme_features,
            'preprocessing_method': 'flow_based',
            'timestamp': time.time()
        }
        
        with open(os.path.join(output_dir, 'preprocessing_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nFlow-based preprocessed data saved to {output_dir}")
        print("Files created:")
        print("  - X_train_flow_based.npy, X_test_flow_based.npy")
        print("  - y_train_flow_based.npy, y_test_flow_based.npy")
        print("  - scaler_flow_based.pkl")
        print("  - feature_names.txt")
        print("  - preprocessing_metadata.json")
    
    return X_train_scaled, X_test_scaled, y_train_balanced, y_test


def validate_preprocessing_results(X_train, X_test, y_train, y_test):
    """Validate the preprocessing results"""
    print("\n" + "="*50)
    print("PREPROCESSING VALIDATION RESULTS")
    print("="*50)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Feature count: {X_train.shape[1]}")
    
    print(f"\nLabel distribution:")
    print(f"  Training - Benign: {sum(y_train == 0)}, Attack: {sum(y_train == 1)}")
    print(f"  Testing - Benign: {sum(y_test == 0)}, Attack: {sum(y_test == 1)}")
    
    print(f"\nFeature statistics (first 10 features):")
    for i in range(min(10, X_train.shape[1])):
        print(f"  Feature {i}: min={X_train[:, i].min():.4f}, max={X_train[:, i].max():.4f}, mean={X_train[:, i].mean():.4f}")
    
    print(f"\nData quality checks:")
    print(f"  NaN values in training: {np.isnan(X_train).sum()}")
    print(f"  Infinite values in training: {np.isinf(X_train).sum()}")
    print(f"  Feature variance > 0: {sum(X_train.var(axis=0) > 0)}/{X_train.shape[1]}")
    
    print("="*50)


if __name__ == "__main__":
    # Configuration for CSV-based preprocessing
    CSV_SOURCE_DIR = 'data/MachineLearningCVE'  # Your CSV files location
    OUTPUT_DIR = 'data/preprocessed_csv'        # Where to save preprocessed data
    TEST_SIZE = 0.2                             # 20% for testing
    RANDOM_STATE = 42                           # For reproducibility
    
    print("CIC-IDS 2017 Network Traffic Preprocessing")
    print("="*50)
    
    try:
        # CSV-based preprocessing (recommended for CIC-IDS 2017)
        X_train, X_test, y_train, y_test = preprocess_csv_data(
            csv_dir=CSV_SOURCE_DIR,
            output_dir=OUTPUT_DIR,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        
        if X_train is not None:
            # Validate results
            validate_preprocessing_results(X_train, X_test, y_train, y_test)
            
            # Final summary
            print(f"\n🎉 SUCCESS! Preprocessed data ready for training:")
            print(f"   📁 Location: {OUTPUT_DIR}")
            print(f"   📊 Training samples: {X_train.shape[0]:,}")
            print(f"   📊 Test samples: {X_test.shape[0]:,}")
            print(f"   🎯 Features: {X_train.shape[1]}")
            print(f"   ⚖️  Balanced training set: {(y_train==0).sum():,} benign, {(y_train==1).sum():,} attack")
            
            print(f"\n📋 Next steps:")
            print(f"   1. Run: python src/train_xgboost.py")
            print(f"   2. Run: python src/train_bilstm.py") 
            print(f"   3. Test: python src/realtime_detection.py --pcap data/Friday-WorkingHours.pcap --model xgboost --threshold 0.1")
        else:
            print("CSV preprocessing failed. Trying PCAP-based preprocessing...")
            
            # Fallback to PCAP-based preprocessing if CSV fails
            pcap_files_with_labels = [
                ('data/Monday-WorkingHours.pcap', 0),    # Benign traffic
                ('data/Tuesday-WorkingHours.pcap', 1),   # Contains attacks
                ('data/Wednesday-WorkingHours.pcap', 1), # Contains attacks
                ('data/Thursday-WorkingHours.pcap', 0),  # Benign traffic
                ('data/Friday-WorkingHours.pcap', 1),    # Contains attacks
            ]
            
            # PCAP-based preprocessing
            X_train, X_test, y_train, y_test = preprocess_pcap_data(
                pcap_files_with_labels, 
                output_dir=OUTPUT_DIR,
                test_size=TEST_SIZE,
                max_packets_per_file=10000
            )
            
            if X_train is not None:
                validate_preprocessing_results(X_train, X_test, y_train, y_test)
                print("\nFlow-based PCAP preprocessing completed successfully!")
                print(f"Use the files in '{OUTPUT_DIR}' for training your corrected models.")
            else:
                print("Both CSV and PCAP preprocessing failed. Please check your data files.")
        
    except Exception as e:
        print(f"\n❌ PREPROCESSING FAILED: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   • Check that CSV files exist in: {CSV_SOURCE_DIR}")
        print(f"   • Verify CSV files are readable and not corrupted")
        print(f"   • Ensure sufficient disk space for output directory")
        
        # Show available CSV files for debugging
        try:
            csv_files = glob.glob(os.path.join(CSV_SOURCE_DIR, '*.csv'))
            if csv_files:
                print(f"   • Found CSV files:")
                for f in csv_files:
                    print(f"     - {os.path.basename(f)}")
            else:
                print(f"   • No CSV files found in {CSV_SOURCE_DIR}")
        except:
            pass
