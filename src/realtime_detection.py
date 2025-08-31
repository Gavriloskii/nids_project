from __future__ import annotations
import os
import time
import json
import joblib
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import pyshark
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

class NetworkIntrusionDetector:
    """
    Enhanced Flow-based Network Intrusion Detection System supporting XGBoost.

    Features:
    - TCP/UDP-only flow tracking for accurate intrusion detection
    - Real-time feature extraction matching training data format
    - Support for both live capture and PCAP file analysis
    - Batch processing for optimal performance
    - Enhanced alert validation and filtering
    - Production-optimized thresholds based on extensive testing
    - Proper stop signal handling and resource cleanup
    """

    # Renamed to reduce collision risk and used defensively
    OPTIMAL_CFG = {
        'xgboost': {
            'threshold': 0.01,
            'description': 'Optimized for 99.74% accuracy with <1% false positives',
            'expected_alert_rate': '1.21% on mixed traffic',
            'performance_notes': 'Excellent balance: 7 alerts on attack traffic, 2 on benign'
        }
    }

    def __init__(self, interface: str = 'eth0', model_type: str = 'xgboost', threshold: Optional[float] = None):
        """
        Initialize the Network Intrusion Detection System.
        """
        self.interface = interface
        self.model_type = (model_type or 'xgboost').lower()
        if self.model_type != 'xgboost':
            print(f"⚠️ Unsupported model '{self.model_type}'. Falling back to XGBoost.")
            self.model_type = 'xgboost'  # [attached_file:1]

        # Load model and scaler
        self.model = self._load_model()  # diagnostics cannot flip a success into failure
        self.scaler = self._load_scaler()

        # Resolve threshold robustly
        self.threshold = self._get_optimal_threshold(threshold)  # [attached_file:1]

        print("🎯 Production Configuration Loaded:")
        print(f"   Model: {self.model_type.upper()}")
        print(f"   Threshold: {self.threshold}")
        try:
            cfg = self.OPTIMAL_CFG.get(self.model_type) if isinstance(self.OPTIMAL_CFG, dict) else None
            if isinstance(cfg, dict):
                print(f"   Expected Performance: {cfg.get('description', 'n/a')}")
                print(f"   Expected Alert Rate: {cfg.get('expected_alert_rate', 'n/a')}")
        except Exception:
            pass  # [attached_file:1]

        # Flow tracking - TCP/UDP only for accurate detection
        self.flows: Dict[Tuple, Dict] = {}
        self.flow_timeout = 60  # Reduced for faster detection of short-lived scans  # [attached_file:1]

        # Performance metrics
        self.performance = {
            'total_packets_analyzed': 0,
            'total_tcp_udp_packets': 0,
            'total_alerts': 0,
            'processing_time': 0.0,
            'alert_rate': 0.0,
            'flows_processed': 0,
            'invalid_flows_filtered': 0
        }  # [attached_file:1]

        # Runtime state
        self.packet_count = 0
        self.start_time = None
        self.capture_active = False
        self.stop_requested = False
        self.alerts: List[Dict] = []
        self.capture = None  # [attached_file:1]

        # Enforce one alert per flow across the session
        self._alerted_flow_ids: set = set()  # [attached_file:1]

        # Required features list (cached for performance)
        self._required_features = self._get_required_features()  # [attached_file:1]

        # Resolve alerts dir to the app's alerts folder
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._alerts_dir = os.path.join(project_root, 'alerts')
        os.makedirs(self._alerts_dir, exist_ok=True)  # [attached_file:1]

    def stop_detection(self):
        """Stop the current detection process gracefully."""
        print("🛑 Stop signal received...")
        self.stop_requested = True
        self.capture_active = False
        if self.capture:
            try:
                self.capture.close()
            except Exception:
                pass
        if self.flows:
            print("🔄 Force completing remaining flows...")
            completed_flows = self.get_completed_flows(force_completion=True)
            return len(completed_flows)
        return 0  # [attached_file:1]

    def _get_optimal_threshold(self, user_threshold: Optional[float] = None) -> float:
        """Resolve threshold safely, without assuming configs are intact."""
        # Explicit numeric wins
        if isinstance(user_threshold, (int, float)):
            print(f"⚠️  Using user-specified threshold: {user_threshold}")
            try:
                if isinstance(self.OPTIMAL_CFG, dict):
                    cfg = self.OPTIMAL_CFG.get(self.model_type)
                    if isinstance(cfg, dict) and 'threshold' in cfg:
                        print(f"   (Production-optimized threshold for {self.model_type}: {cfg['threshold']})")
            except Exception:
                pass
            return float(user_threshold)

        # Fallback to production-optimized
        try:
            if isinstance(self.OPTIMAL_CFG, dict):
                cfg = self.OPTIMAL_CFG.get(self.model_type)
                if isinstance(cfg, dict):
                    t = cfg.get('threshold', None)
                    if isinstance(t, (int, float)):
                        return float(t)
        except Exception:
            pass

        print(f"⚠️  Unknown or unavailable config for model {self.model_type}, using conservative default 0.1")
        return 0.1  # [attached_file:1]

    def _load_model(self):
        """Load the trained XGBoost model; optional diagnostics never abort a success."""
        print(f"Loading {self.model_type} model...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if self.model_type == 'xgboost':
            model_paths = [
                os.path.join(project_root, 'models', 'xgboost', 'xgboost_model_corrected.json'),
                os.path.join(project_root, 'models', 'xgboost', 'xgboost_model.json')
            ]
            model = xgb.XGBClassifier()
            for model_path in model_paths:
                if not os.path.exists(model_path):
                    continue
                try:
                    model.load_model(model_path)  # Only this is allowed to fail loading
                except Exception as e:
                    print(f"⚠️  Failed to load {model_path}: {e}")
                    continue
                print(f"✅ XGBoost model loaded from: {model_path}")
                # Optional diagnostics in their own guards
                try:
                    feat_cnt = getattr(model, 'n_features_in_', 'unknown')
                    print(f"   Model feature count: {feat_cnt}")
                except Exception:
                    print("   Model feature count: unknown")
                try:
                    if hasattr(model, 'feature_importances_'):
                        top_features = sorted(
                            enumerate(model.feature_importances_),
                            key=lambda x: float(x[1]) if isinstance(x[1], (int, float, np.floating)) else 0.0,
                            reverse=True
                        )[:5]
                        print(f"   Top 5 important features: {top_features}")
                except Exception as e:
                    print(f"   Feature importances unavailable: {e}")
                return model
            raise FileNotFoundError("❌ No XGBoost model file found or could be loaded")
        raise ValueError(f"❌ Unsupported model type: {self.model_type}")  # [attached_file:1]

    def _load_scaler(self):
        """Load the saved StandardScaler from training."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scaler_paths = [
            os.path.join(project_root, 'data', 'preprocessed_csv', 'scaler.pkl'),
            os.path.join(project_root, 'models', 'xgboost', 'scaler.pkl'),
            os.path.join(project_root, 'models', 'scaler.pkl'),
        ]
        for scaler_path in scaler_paths:
            if os.path.exists(scaler_path):
                try:
                    scaler = joblib.load(scaler_path)
                    print(f"✅ Scaler loaded from: {scaler_path}")
                    return scaler
                except Exception as e:
                    print(f"⚠️  Failed to load scaler from {scaler_path}: {e}")
                    continue
        print("❌ Warning: No saved scaler found. Using default StandardScaler!")
        return StandardScaler()  # [attached_file:1]

    def _safe_packet_length(self, packet) -> int:
        """Best-effort packet length extraction across protocol layers."""
        try:
            if hasattr(packet, 'length'):
                return int(packet.length)
        except Exception:
            pass
        try:
            if hasattr(packet, 'ip') and hasattr(packet.ip, 'len'):
                return int(packet.ip.len)
        except Exception:
            pass
        try:
            if hasattr(packet, 'udp') and hasattr(packet.udp, 'length'):
                return int(packet.udp.length)
        except Exception:
            pass
        try:
            if hasattr(packet, 'frame_info') and hasattr(packet.frame_info, 'len'):
                return int(packet.frame_info.len)
        except Exception:
            pass
        return 0  # [attached_file:1]

    def get_flow_key(self, packet) -> Tuple[Optional[Tuple], Optional[str]]:
        """
        Generate unique flow identifier for TCP/UDP packets only, with IPv4/IPv6 support.
        """
        try:
            ip_layer = packet.ip if hasattr(packet, 'ip') else (packet.ipv6 if hasattr(packet, 'ipv6') else None)
            if ip_layer is None:
                return None, None
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst

            if hasattr(packet, 'tcp'):
                src_port = int(packet.tcp.srcport)
                dst_port = int(packet.tcp.dstport)
                protocol = 'TCP'
            elif hasattr(packet, 'udp'):
                src_port = int(packet.udp.srcport)
                dst_port = int(packet.udp.dstport)
                protocol = 'UDP'
            else:
                return None, None

            if not (1 <= src_port <= 65535) or not (1 <= dst_port <= 65535):
                return None, None

            endpoint1 = (src_ip, src_port)
            endpoint2 = (dst_ip, dst_port)
            if endpoint1 <= endpoint2:
                flow_key = (endpoint1, endpoint2, protocol)
                direction = 'forward'
            else:
                flow_key = (endpoint2, endpoint1, protocol)
                direction = 'backward'
            return flow_key, direction
        except Exception as e:
            print(f"❌ Error creating flow key: {e}")
            return None, None  # [attached_file:1]

    def update_flow(self, packet) -> Optional[Tuple]:
        """Update flow statistics with new packet - TCP/UDP only."""
        flow_key, direction = self.get_flow_key(packet)
        if flow_key is None:
            return None
        if not isinstance(flow_key, tuple) or len(flow_key) != 3:
            return None  # [attached_file:1]

        try:
            current_time = float(packet.sniff_timestamp)
            packet_length = self._safe_packet_length(packet)

            if flow_key not in self.flows:
                # FIX 1: correct protocol index fallback
                protocol_str = 'TCP' if hasattr(packet, 'tcp') else ('UDP' if hasattr(packet, 'udp') else flow_key[2])
                # FIX 2: correct endpoints tuple (no stray literal)
                self.flows[flow_key] = {
                    'start_time': current_time,
                    'last_time': current_time,
                    'fwd_packets': 0,
                    'bwd_packets': 0,
                    'fwd_bytes': 0,
                    'bwd_bytes': 0,
                    'fwd_lengths': [],
                    'bwd_lengths': [],
                    'timestamps': [],
                    'tcp_flags': defaultdict(int),
                    'fwd_header_lengths': [],
                    'bwd_header_lengths': [],
                    'packets': [],
                    'endpoints': (flow_key[0], flow_key[1]),
                    'protocol': protocol_str
                }

            flow = self.flows[flow_key]
            flow['last_time'] = current_time
            flow['timestamps'].append(current_time)

            if direction == 'forward':
                flow['fwd_packets'] += 1
                flow['fwd_bytes'] += packet_length
                flow['fwd_lengths'].append(packet_length)
                if hasattr(packet, 'tcp'):
                    try:
                        flow['fwd_header_lengths'].append(int(getattr(packet.tcp, 'hdr_len', 20)))
                    except Exception:
                        flow['fwd_header_lengths'].append(20)
            else:
                flow['bwd_packets'] += 1
                flow['bwd_bytes'] += packet_length
                flow['bwd_lengths'].append(packet_length)
                if hasattr(packet, 'tcp'):
                    try:
                        flow['bwd_header_lengths'].append(int(getattr(packet.tcp, 'hdr_len', 20)))
                    except Exception:
                        flow['bwd_header_lengths'].append(20)

            if hasattr(packet, 'tcp'):
                def safe_flag(flag_name):
                    try:
                        val = getattr(packet.tcp, flag_name, False)
                        if isinstance(val, bool):
                            return int(val)
                        if isinstance(val, int):
                            return val
                        if isinstance(val, str):
                            return 1 if val.lower() == 'true' or val == '1' else 0
                        return 0
                    except Exception:
                        return 0
                flow['tcp_flags']['fin'] += safe_flag('flags_fin')
                flow['tcp_flags']['syn'] += safe_flag('flags_syn')
                flow['tcp_flags']['rst'] += safe_flag('flags_reset')
                flow['tcp_flags']['psh'] += safe_flag('flags_push')
                flow['tcp_flags']['ack'] += safe_flag('flags_ack')
                flow['tcp_flags']['urg'] += safe_flag('flags_urg')

            if len(flow['packets']) == 0:
                explicit_proto = 'TCP' if hasattr(packet, 'tcp') else ('UDP' if hasattr(packet, 'udp') else flow_key[2])
                flow['packets'].append({
                    'timestamp': current_time,
                    'length': packet_length,
                    'direction': direction,
                    'protocol': explicit_proto
                })

            return flow_key
        except Exception as e:
            print(f"❌ Error updating flow: {e}")
            return None  # [attached_file:1]

    def extract_flow_features(self, flow_data: Dict) -> Dict[str, float]:
        """Extract comprehensive flow features for ML model."""
        try:
            duration = max(0.000001, flow_data['last_time'] - flow_data['start_time'])
            total_packets = flow_data['fwd_packets'] + flow_data['bwd_packets']
            total_bytes = flow_data['fwd_bytes'] + flow_data['bwd_bytes']

            dest_port = flow_data['endpoints'][1][1]

            fwd_lengths = flow_data['fwd_lengths']
            if fwd_lengths:
                fwd_max = max(fwd_lengths); fwd_min = min(fwd_lengths)
                fwd_mean = np.mean(fwd_lengths); fwd_std = np.std(fwd_lengths)
            else:
                fwd_max = fwd_min = fwd_mean = fwd_std = 0

            bwd_lengths = flow_data['bwd_lengths']
            if bwd_lengths:
                bwd_max = max(bwd_lengths); bwd_min = min(bwd_lengths)
                bwd_mean = np.mean(bwd_lengths); bwd_std = np.std(bwd_lengths)
            else:
                bwd_max = bwd_min = bwd_mean = bwd_std = 0

            timestamps = flow_data['timestamps']
            if len(timestamps) > 1:
                iats = np.diff(timestamps)
                iat_mean = np.mean(iats) * 1_000_000
                iat_std = np.std(iats) * 1_000_000
                iat_max = float(np.max(iats)) * 1_000_000
                iat_min = float(np.min(iats)) * 1_000_000
            else:
                iat_mean = iat_std = iat_max = iat_min = 0

            all_lengths = fwd_lengths + bwd_lengths
            if all_lengths:
                all_min = min(all_lengths); all_max = max(all_lengths)
                all_mean = np.mean(all_lengths); all_std = np.std(all_lengths)
                all_var = np.var(all_lengths)
            else:
                all_min = all_max = all_mean = all_std = all_var = 0

            fwd_header_mean = np.mean(flow_data['fwd_header_lengths']) if flow_data['fwd_header_lengths'] else 0
            bwd_header_mean = np.mean(flow_data['bwd_header_lengths']) if flow_data['bwd_header_lengths'] else 0

            tcp_flags = flow_data['tcp_flags']

            features = {
                ' Destination Port': dest_port,
                ' Flow Duration': duration * 1_000_000,
                ' Total Fwd Packets': flow_data['fwd_packets'],
                ' Total Backward Packets': flow_data['bwd_packets'],
                'Total Length of Fwd Packets': flow_data['fwd_bytes'],
                ' Total Length of Bwd Packets': flow_data['bwd_bytes'],
                ' Fwd Packet Length Max': fwd_max,
                ' Fwd Packet Length Min': fwd_min,
                ' Fwd Packet Length Mean': fwd_mean,
                ' Fwd Packet Length Std': fwd_std,
                'Bwd Packet Length Max': bwd_max,
                ' Bwd Packet Length Min': bwd_min,
                ' Bwd Packet Length Mean': bwd_mean,
                ' Bwd Packet Length Std': bwd_std,
                'Flow Bytes/s': total_bytes / duration,
                ' Flow Packets/s': total_packets / duration,
                ' Flow IAT Mean': iat_mean,
                ' Flow IAT Std': iat_std,
                ' Flow IAT Max': iat_max,
                ' Flow IAT Min': iat_min,
                'Fwd IAT Total': 0, ' Fwd IAT Mean': 0, ' Fwd IAT Std': 0, ' Fwd IAT Max': 0, ' Fwd IAT Min': 0,
                'Bwd IAT Total': 0, ' Bwd IAT Mean': 0, ' Bwd IAT Std': 0, ' Bwd IAT Max': 0, ' Bwd IAT Min': 0,
                'Fwd PSH Flags': tcp_flags['psh'],
                ' Bwd PSH Flags': 0,
                ' Fwd URG Flags': tcp_flags['urg'],
                ' Bwd URG Flags': 0,
                ' Fwd Header Length': fwd_header_mean,
                ' Bwd Header Length': bwd_header_mean,
                'Fwd Packets/s': flow_data['fwd_packets'] / duration,
                ' Bwd Packets/s': flow_data['bwd_packets'] / duration,
                ' Min Packet Length': all_min,
                ' Max Packet Length': all_max,
                ' Packet Length Mean': all_mean,
                ' Packet Length Std': all_std,
                ' Packet Length Variance': all_var,
                'FIN Flag Count': tcp_flags['fin'],
                ' SYN Flag Count': tcp_flags['syn'],
                ' RST Flag Count': tcp_flags['rst'],
                ' PSH Flag Count': tcp_flags['psh'],
                ' ACK Flag Count': tcp_flags['ack'],
                ' URG Flag Count': tcp_flags['urg'],
                ' CWE Flag Count': 0,
                ' ECE Flag Count': 0,
                ' Down/Up Ratio': (flow_data['bwd_bytes'] / flow_data['fwd_bytes']) if flow_data['fwd_bytes'] > 0 else 0,
                ' Average Packet Size': all_mean
            }
            return features
        except Exception as e:
            print(f"❌ Error extracting flow features: {e}")
            return {}  # [attached_file:1]

    def _get_required_features(self) -> List[str]:
        """Return the list of features required by the model in the correct order."""
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
        ]  # [attached_file:1]

    def get_completed_flows(self, force_completion: bool = False) -> List[Tuple]:
        """Get flows that are ready for analysis."""
        current_time = time.time()
        completed_flows = []
        flows_to_remove = []
        for flow_key, flow_data in list(self.flows.items()):
            time_since_last = current_time - flow_data['last_time']
            has_enough_packets = (flow_data['fwd_packets'] + flow_data['bwd_packets']) >= 1

            dest_port = flow_data['endpoints'][1][1]
            if dest_port == 0:
                flows_to_remove.append(flow_key)
                self.performance['invalid_flows_filtered'] += 1
                continue

            if (time_since_last > self.flow_timeout or force_completion) and has_enough_packets:
                features = self.extract_flow_features(flow_data)
                if features and features.get(' Destination Port', 0) > 0:
                    completed_flows.append((flow_key, features, flow_data))
                flows_to_remove.append(flow_key)

        for flow_key in flows_to_remove:
            self.flows.pop(flow_key, None)
        return completed_flows  # [attached_file:1]

    def preprocess_features(self, features_list: List[Dict]) -> np.ndarray:
        """Preprocess extracted features to match the format used in training."""
        if not features_list:
            return np.array([])

        features_list = [x for x in features_list if isinstance(x, dict)]
        if not features_list:
            return np.array([])

        try:
            df = pd.DataFrame(features_list)
            for col in self._required_features:
                if col not in df.columns:
                    df[col] = 0
            X = df[self._required_features]
            X_np = X.replace([np.inf, -np.inf], [1e9, -1e9]).fillna(0).to_numpy(dtype=float)

            try:
                X_scaled = self.scaler.transform(X_np)
            except NotFittedError:
                print("⚠️  Scaler not fitted; proceeding without scaling for this batch.")
                X_scaled = X_np
            except Exception as e:
                print(f"⚠️  Scaling failed ({e}); proceeding without scaling for this batch.")
                X_scaled = X_np

            return X_scaled
        except Exception as e:
            print(f"❌ Error preprocessing features: {e}")
            return np.array([])  # [attached_file:1]

    def process_flow_batch(self, flow_features: List[Dict], flow_packets: List[Dict]) -> List[Dict]:
        """Process a batch of flows for intrusion detection with enhanced validation."""
        if not flow_features:
            return []

        n = min(len(flow_features), len(flow_packets))
        if n == 0:
            return []
        flow_features = flow_features[:n]
        flow_packets = flow_packets[:n]

        print(f"🔍 Processing batch of {len(flow_features)} flows...")

        X = self.preprocess_features(flow_features)
        if len(X) == 0:
            return []

        batch_alerts: List[Dict] = []
        try:
            raw_predictions = self.model.predict_proba(X)
            positive_probs = raw_predictions[:, 1]
            print(f"📊 XGBoost predictions - min: {positive_probs.min():.4f}, max: {positive_probs.max():.4f}, mean: {positive_probs.mean():.4f}")

            thresholds_to_check = [0.005, 0.01, 0.02, 0.05, 0.1]
            for thresh in thresholds_to_check:
                count = (positive_probs > thresh).sum()
                print(f"   Probabilities > {thresh}: {count}")
            print("   First 10 probabilities:", positive_probs[:10])

            predictions = (positive_probs > self.threshold).astype(int)
            print(f"🎯 Using production threshold {self.threshold}: Found {predictions.sum()} potential intrusions")

            for i, (pred, prob) in enumerate(zip(predictions, positive_probs)):
                if pred != 1:
                    continue
                packet = flow_packets[i] if i < len(flow_packets) else {}
                alert = self._create_alert('xgboost', flow_features[i], packet, float(prob))
                if not alert:
                    continue
                flow_id = (
                    alert.get('source_ip', 'Unknown'),
                    alert.get('destination_ip', 'Unknown'),
                    alert.get('destination_port', 0),
                    alert.get('protocol', 'Unknown')
                )
                if flow_id in self._alerted_flow_ids:
                    continue
                self._alerted_flow_ids.add(flow_id)
                batch_alerts.append(alert)
                print(f"🚨 ALERT: XGBoost intrusion detected - Port: {alert['destination_port']}, Confidence: {prob:.4f}")
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")

        print(f"✅ Batch complete. Found {len(batch_alerts)} potential intrusions.")
        return batch_alerts  # [attached_file:1]

    def _attach_ips_to_packet(self, flow_key: Tuple, packet_stub: Dict) -> Dict:
        """Attach source/destination IPs and protocol to a packet stub using flow_key."""
        pkt = dict(packet_stub) if isinstance(packet_stub, dict) else {}
        try:
            (ip1, _p1), (ip2, _p2), proto = flow_key
            direction = pkt.get('direction')
            if direction == 'forward':
                pkt['source_ip'] = ip1; pkt['destination_ip'] = ip2
            elif direction == 'backward':
                pkt['source_ip'] = ip2; pkt['destination_ip'] = ip1
            else:
                pkt['source_ip'] = ip1; pkt['destination_ip'] = ip2
            if not pkt.get('protocol'):
                pkt['protocol'] = proto
        except Exception:
            pkt.setdefault('source_ip', 'Unknown')
            pkt.setdefault('destination_ip', 'Unknown')
            pkt.setdefault('protocol', 'Unknown')
        return pkt  # [attached_file:1]

    def _create_alert(self, model_type: str, features: Dict, packet: Dict, confidence: float) -> Optional[Dict]:
        """Create a standardized alert dictionary with validation."""
        destination_port = int(features.get(' Destination Port', 0))
        if destination_port <= 0 or destination_port > 65535:
            return None
        source_ip = packet.get('source_ip', 'Unknown')
        destination_ip = packet.get('destination_ip', 'Unknown')
        return {
            'timestamp': time.time(),
            'model': model_type,
            'source_ip': source_ip,
            'destination_ip': destination_ip,
            'destination_port': destination_port,
            'protocol': packet.get('protocol', 'Unknown'),
            'confidence': confidence,
            'threshold_used': self.threshold,
            'packet_info': {
                'length': packet.get('length', 0),
                'time': packet.get('timestamp', 0)
            }
        }  # [attached_file:1]

    def detect_intrusions_from_file(self, pcap_file: str, max_packets: int = 5000, batch_size: int = 1000) -> List[Dict]:
        """Analyze a PCAP file for intrusions using flow-based feature extraction."""
        print(f"🔍 Analyzing PCAP file: {pcap_file} (limited to {max_packets} packets)")

        capture = None
        packet_count = 0
        tcp_udp_count = 0
        total_alerts: List[Dict] = []

        try:
            # Filter to TCP/UDP packets at capture level
            capture = pyshark.FileCapture(pcap_file, display_filter="tcp or udp")

            for packet in capture:
                if self.stop_requested:
                    print("🛑 Stop requested during PCAP analysis")
                    break

                packet_count += 1
                if packet_count > max_packets:
                    print(f"⏹️  Reached maximum packet limit ({max_packets}). Stopping analysis.")
                    break

                if packet_count % 1000 == 0:
                    print(f"   Processed {packet_count} packets... ({tcp_udp_count} TCP/UDP)")

                flow_key = self.update_flow(packet)
                if flow_key:
                    tcp_udp_count += 1

                if packet_count % 500 == 0:
                    completed = self.get_completed_flows()
                    if completed:
                        flow_features = [features for _, features, _ in completed]
                        flow_packets = []
                        for fk, _, flow_data in completed:
                            # FIX 3: use first packet dict, not the whole list
                            stub = flow_data['packets'][0] if flow_data['packets'] else {}
                            flow_packets.append(self._attach_ips_to_packet(fk, stub))
                        alerts = self.process_flow_batch(flow_features, flow_packets)
                        total_alerts.extend(alerts)

            print("🔄 Processing remaining flows...")
            completed = self.get_completed_flows(force_completion=True)
            if completed:
                flow_features = [features for _, features, _ in completed]
                flow_packets = []
                for fk, _, flow_data in completed:
                    # FIX 4: use first packet dict, not the whole list
                    stub = flow_data['packets'][0] if flow_data['packets'] else {}
                    flow_packets.append(self._attach_ips_to_packet(fk, stub))
                alerts = self.process_flow_batch(flow_features, flow_packets)
                total_alerts.extend(alerts)

        except Exception as e:
            print(f"❌ Error during PCAP analysis: {e}")
        finally:
            if capture:
                try:
                    capture.close()
                except Exception:
                    pass
            self.packet_count = packet_count

        self._update_performance_metrics(packet_count, tcp_udp_count, total_alerts)
        print(f"✅ PCAP analysis complete. Processed {packet_count} packets ({tcp_udp_count} TCP/UDP). Found {len(total_alerts)} potential intrusions.")
        print(f"💾 Saving alerts: {len(total_alerts)}")
        total_alerts = self._dedup_alerts(total_alerts)
        self.save_alerts(total_alerts)
        return total_alerts  # [attached_file:1]

    def detect_intrusions(self, duration: int = 60) -> List[Dict]:
        """Capture packets and detect intrusions with proper timeout handling."""
        print(f"🔴 Starting live packet capture on interface {self.interface} for {duration} seconds...")

        self.start_time = time.time()
        self.capture_active = True
        self.stop_requested = False
        packet_count = 0
        tcp_udp_count = 0
        total_alerts: List[Dict] = []

        try:
            # Filter to TCP/UDP packets at capture level
            self.capture = pyshark.LiveCapture(interface=self.interface, display_filter="tcp or udp")
            print("🎯 Live capture started. Monitoring for intrusions...")

            for packet in self.capture.sniff_continuously():
                current_time = time.time()
                elapsed_time = current_time - self.start_time

                if self.stop_requested or not self.capture_active:
                    print("🛑 Stop signal received, breaking capture loop...")
                    break

                if elapsed_time >= duration:
                    print("⏰ Duration elapsed, stopping capture...")
                    self.capture_active = False
                    break

                packet_count += 1
                flow_key = self.update_flow(packet)
                if flow_key:
                    tcp_udp_count += 1

                if packet_count % 100 == 0:
                    completed = self.get_completed_flows()
                    if completed:
                        flow_features = [features for _, features, _ in completed]
                        flow_packets = []
                        for fk, _, flow_data in completed:
                            # FIX 5: use first packet dict, not the whole list
                            stub = flow_data['packets'][0] if flow_data['packets'] else {}
                            flow_packets.append(self._attach_ips_to_packet(fk, stub))
                        alerts = self.process_flow_batch(flow_features, flow_packets)
                        total_alerts.extend(alerts)

                if packet_count % 1000 == 0:
                    print(f"   Processed {packet_count} packets ({tcp_udp_count} TCP/UDP)... ({elapsed_time:.1f}s elapsed)")

            print("🔄 Processing remaining flows...")
            completed = self.get_completed_flows(force_completion=True)
            if completed:
                flow_features = [features for _, features, _ in completed]
                flow_packets = []
                for fk, _, flow_data in completed:
                    stub = flow_data['packets'][0] if flow_data['packets'] else {}
                    flow_packets.append(self._attach_ips_to_packet(fk, stub))
                alerts = self.process_flow_batch(flow_features, flow_packets)
                total_alerts.extend(alerts)

        except KeyboardInterrupt:
            print("\n⏸️  Capture interrupted by user (Ctrl+C).")
            self.stop_requested = True
        except Exception as e:
            print(f"❌ Error during live capture: {e}")
            self.stop_requested = True
        finally:
            self.capture_active = False
            if self.capture:
                try:
                    self.capture.close()
                except Exception:
                    pass

            elapsed = time.time() - self.start_time
            print(f"\n✅ Live capture completed in {elapsed:.2f} seconds.")
            print(f"   Processed {packet_count} packets ({tcp_udp_count} TCP/UDP) in {len(self.flows)} active flows.")
            self._update_performance_metrics(packet_count, tcp_udp_count, total_alerts, elapsed)

        print(f"🎯 Detection complete. Found {len(total_alerts)} potential intrusions.")
        print(f"💾 Saving alerts: {len(total_alerts)}")
        total_alerts = self._dedup_alerts(total_alerts)
        self.save_alerts(total_alerts)
        return total_alerts  # [attached_file:1]

    def _update_performance_metrics(self, packet_count: int, tcp_udp_count: int, alerts: List[Dict], elapsed_time: float = None):
        """Update performance metrics with enhanced tracking."""
        if elapsed_time is not None:
            self.performance['processing_time'] = elapsed_time
        self.performance['total_packets_analyzed'] = packet_count
        self.performance['total_tcp_udp_packets'] = tcp_udp_count
        self.performance['total_alerts'] = len(alerts)
        self.performance['flows_processed'] = len(self.flows)
        if tcp_udp_count > 0:
            self.performance['alert_rate'] = (len(alerts) / tcp_udp_count) * 100  # [attached_file:1]

    def _dedup_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """Final safety: remove duplicates created for the same flow/session."""
        seen = set()
        unique = []
        for a in alerts or []:
            key = (
                a.get('source_ip', 'Unknown'),
                a.get('destination_ip', 'Unknown'),
                a.get('destination_port', 0),
                a.get('protocol', 'Unknown'),
                round(float(a.get('confidence', 0.0)), 4)
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(a)
        if len(unique) != len(alerts):
            print(f"🧹 Deduplicated alerts: {len(alerts)} -> {len(unique)}")
        return unique  # [attached_file:1]

    def save_alerts(self, alerts: List[Dict]) -> None:
        """Save alerts to a JSON file (../alerts). Always write a file, even when zero alerts."""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self._alerts_dir, f"alerts_{timestamp}.json")
            # Safely include config summary
            cfg = self.OPTIMAL_CFG.get(self.model_type) if isinstance(self.OPTIMAL_CFG, dict) else None
            session_info = {
                'session_metadata': {
                    'timestamp': timestamp,
                    'model_type': self.model_type,
                    'threshold': self.threshold,
                    'threshold_config': cfg if isinstance(cfg, dict) else None,
                    'total_alerts': len(alerts),
                    'no_detections': len(alerts) == 0,
                    'performance_metrics': self.performance
                },
                'alerts': alerts or []
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_info, f, indent=2, default=str)
            print(f"💾 Alerts saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving alerts: {e}")  # [attached_file:1]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return comprehensive performance metrics for the detection system."""
        metrics = self.performance.copy()
        optimized_for = None
        try:
            if isinstance(self.OPTIMAL_CFG, dict):
                cfg = self.OPTIMAL_CFG.get(self.model_type)
                if isinstance(cfg, dict):
                    optimized_for = cfg.get('description')
        except Exception:
            pass
        metrics['threshold_configuration'] = {
            'model_type': self.model_type,
            'threshold': self.threshold,
            'optimized_for': optimized_for or 'n/a'
        }
        return metrics  # [attached_file:1]

def main():
    """Main function to run the Production-Ready Network Intrusion Detection System."""
    import argparse
    parser = argparse.ArgumentParser(
        description='🛡️  AI-Powered Network Intrusion Detection System (Production-Ready)',
        epilog='Production-optimized thresholds: XGBoost=0.01'
    )
    parser.add_argument('--interface', '-i', default='wlan0', help='Network interface to monitor')
    parser.add_argument('--model', '-m', choices=['xgboost'], default='xgboost',
                        help='Machine learning model to use (XGBoost: 99.74% accuracy)')
    parser.add_argument('--duration', '-d', type=int, default=30,
                        help='Duration in seconds to capture packets')
    parser.add_argument('--pcap', '-p', help='PCAP file to analyze instead of live capture')
    parser.add_argument('--max-packets', type=int, default=5000,
                        help='Maximum number of packets to process from PCAP file')
    parser.add_argument('--threshold', '-t', type=float,
                        help='Detection threshold (default: use production-optimized values)')
    args = parser.parse_args()

    print("🛡️  " + "="*80)
    print("    AI-POWERED NETWORK INTRUSION DETECTION SYSTEM")
    print("    Enhanced Version with TCP/UDP Flow Filtering")
    print("="*84)

    detector = NetworkIntrusionDetector(
        interface=args.interface,
        model_type=args.model,
        threshold=args.threshold
    )

    print("\n🚀 Starting detection process...")

    if args.pcap:
        if os.path.exists(args.pcap):
            alerts = detector.detect_intrusions_from_file(args.pcap, max_packets=args.max_packets)
        else:
            print(f"❌ Error: PCAP file {args.pcap} not found.")
            return
    else:
        try:
            alerts = detector.detect_intrusions(duration=args.duration)
        except Exception as e:
            print(f"❌ Live capture failed: {e}")
            print("🔄 Falling back to PCAP file analysis...")
            try:
                data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
                pcap_files = [f for f in os.listdir(data_dir) if f.endswith('.pcap')]
                if pcap_files:
                    first = os.path.join(data_dir, pcap_files[0])
                    print(f"📁 Found PCAP file: {first}")
                    alerts = detector.detect_intrusions_from_file(first, max_packets=args.max_packets)
                else:
                    print("❌ No PCAP files found.")
                    alerts = []
            except Exception:
                print("❌ Fallback failed.")
                alerts = []

    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)
    print(f"🎯 Detected {len(alerts)} potential intrusions")
    for i, alert in enumerate(alerts[:10]):
        model_name = alert.get('model', 'unknown').upper()
        print(f"   Alert {i+1}: {alert['source_ip']} -> {alert['destination_ip']}:{alert['destination_port']} "
              f"({alert['protocol']}) - Confidence: {alert['confidence']:.4f} [{model_name}]")
    if len(alerts) > 10:
        print(f"   ... and {len(alerts) - 10} more alerts")

    performance = detector.get_performance_metrics()
    print(f"\n📊 Performance Metrics:")
    print(f"   Total packets analyzed: {performance['total_packets_analyzed']:,}")
    print(f"   TCP/UDP packets processed: {performance.get('total_tcp_udp_packets', 0):,}")
    print(f"   Total alerts generated: {performance['total_alerts']}")
    print(f"   Processing time: {performance['processing_time']:.2f} seconds")
    print(f"   Alert rate: {performance['alert_rate']:.2f}%")
    print(f"   Model: {performance['threshold_configuration']['model_type'].upper()}")
    print(f"   Threshold: {performance['threshold_configuration']['threshold']}")
    print(f"   Optimized for: {performance['threshold_configuration']['optimized_for']}")

    if performance['alert_rate'] <= 2.0:
        print("✅ EXCELLENT: Alert rate within production standards (<2%)")
    elif performance['alert_rate'] <= 5.0:
        print("⚠️  GOOD: Alert rate acceptable for high-security environments")
    else:
        print("⚠️  HIGH: Alert rate may cause analyst fatigue - consider threshold tuning")

if __name__ == "__main__":
    main()
