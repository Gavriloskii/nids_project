from flask import Flask, render_template, jsonify, request
import os
print(f"Current working directory: {os.getcwd()}")
import json
import sys
import time
import logging
import threading
import signal
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.realtime_detection import NetworkIntrusionDetector

app = Flask(__name__)

# Model configuration (XGBoost only)
MODEL_CONFIG = {
    'xgboost': {
        'model_file': 'xgboost_model_corrected.json',
        'scaler_file': 'scaler.pkl',
        'fallback_model': 'xgboost_model.json',
        'metrics': {
            'accuracy': 0.9974,
            'precision': 0.9911,
            'recall': 0.9971,
            'f1_score': 0.9941,
            'status': 'corrected_features'
        }
    }
}

# Thread-safe globals with proper stop handling
detector = None
detection_thread = None
is_monitoring = threading.Event()
stop_event = threading.Event()
alerts_lock = threading.Lock()
recent_alerts = []
system_status = {
    'status': 'idle',
    'started_at': None,
    'total_packets': 0,
    'alert_count': 0,
    'error': None,
    'progress': 0,
    'is_monitoring': False,
    'model_version': 'corrected',
    'feature_extraction': 'flow_based',
    'can_stop': False,
    'stop_requested': False
}

def load_alerts(limit=100):
    """Load alerts from JSON files with proper error handling and data validation"""
    alerts_dir = os.path.join(os.path.dirname(__file__), '..', 'alerts')
    alerts = []
    try:
        os.makedirs(alerts_dir, exist_ok=True)
        alert_files = sorted([f for f in os.listdir(alerts_dir) if f.endswith('.json')], reverse=True)[:5]
        for file in alert_files:
            file_path = os.path.join(alerts_dir, file)
            try:
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, dict) and 'alerts' in file_data:
                        file_alerts = file_data['alerts']
                    elif isinstance(file_data, list):
                        file_alerts = file_data
                    else:
                        logger.warning(f"Unknown alert file format in {file}: {type(file_data)}")
                        continue
                    for alert in file_alerts:
                        if isinstance(alert, dict):
                            processed_alert = process_single_alert(alert)
                            if processed_alert:
                                alerts.append(processed_alert)
                        elif isinstance(alert, str):
                            logger.warning(f"Skipping string alert in {file}: {alert[:100]}...")
                        else:
                            logger.warning(f"Skipping invalid alert type {type(alert)} in {file}")
                    if len(alerts) >= limit:
                        break
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {file_path}: {str(e)}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Alert loading failed: {str(e)}")
    logger.info(f"Loaded {len(alerts)} valid alerts from {alerts_dir}")
    return alerts[:limit]

def process_single_alert(alert):
    """Process a single alert for dashboard compatibility with comprehensive validation"""
    try:
        if not isinstance(alert, dict):
            logger.warning(f"Invalid alert type: {type(alert)} - {alert}")
            return None
        processed_alert = {
            'timestamp': float(alert.get('timestamp', time.time())),
            'source_ip': str(alert.get('source_ip', 'Flow-based')),
            'destination_ip': str(alert.get('destination_ip', 'Analysis')),
            'destination_port': int(alert.get('destination_port', 0)) if alert.get('destination_port') else 0,
            'protocol': str(alert.get('protocol', 'Unknown')),
            'confidence': float(alert.get('confidence', 0.0)),
            'model': str(alert.get('model', 'xgboost')),
            'threshold_used': float(alert.get('threshold_used', 0.0)),
            'alert_reason': str(alert.get('alert_reason', 'ML Detection')),
            'alert_type': 'Flow-based Detection',
            'severity': get_alert_severity(float(alert.get('confidence', 0.0))),
            'packet_info': alert.get('packet_info', {}),
            'detection_method': 'ML-based'
        }
        if processed_alert['confidence'] < 0 or processed_alert['confidence'] > 1:
            logger.warning(f"Invalid confidence value: {processed_alert['confidence']}")
            processed_alert['confidence'] = max(0.0, min(1.0, processed_alert['confidence']))
        return processed_alert
    except (ValueError, TypeError) as e:
        logger.error(f"Error processing alert (data type error): {e} - Alert: {alert}")
        return None
    except Exception as e:
        logger.error(f"Error processing alert (general error): {e} - Alert: {alert}")
        return None

def get_alert_severity(confidence):
    """Determine alert severity based on confidence score"""
    try:
        confidence = float(confidence)
        if confidence >= 0.8:
            return 'High'
        elif confidence >= 0.5:
            return 'Medium'
        elif confidence >= 0.3:
            return 'Low'
        else:
            return 'Info'
    except (ValueError, TypeError):
        return 'Info'

def process_flow_alerts(alerts):
    """Process flow-based alerts for dashboard display with validation"""
    processed_alerts = []
    for alert in alerts:
        if isinstance(alert, dict):
            processed_alert = process_single_alert(alert)
            if processed_alert:
                processed_alerts.append(processed_alert)
        else:
            logger.warning(f"Skipping non-dict alert: {type(alert)} - {alert}")
    logger.info(f"Processed {len(processed_alerts)} valid alerts from {len(alerts)} total")
    return processed_alerts

def validate_model_files(model_type):
    """Validate that required model files exist (XGBoost only)"""
    config = MODEL_CONFIG.get(model_type, {})
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models', model_type)
    validation_result = {
        'model_type': model_type,
        'model_file_exists': False,
        'scaler_file_exists': False,
        'fallback_available': False,
        'status': 'unknown'
    }
    try:
        model_path = os.path.join(model_dir, config.get('model_file', ''))
        scaler_path = os.path.join(model_dir, config.get('scaler_file', ''))
        fallback_path = os.path.join(model_dir, config.get('fallback_model', ''))
        validation_result['model_file_exists'] = os.path.exists(model_path)
        validation_result['scaler_file_exists'] = os.path.exists(scaler_path)
        validation_result['fallback_available'] = os.path.exists(fallback_path)
        if validation_result['model_file_exists'] and validation_result['scaler_file_exists']:
            validation_result['status'] = 'corrected_ready'
        elif validation_result['fallback_available']:
            validation_result['status'] = 'fallback_available'
        else:
            validation_result['status'] = 'missing_files'
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        validation_result['status'] = 'validation_error'
        validation_result['error'] = str(e)
    return validation_result

def background_detection(interface='lo', model_type='xgboost', duration=300):
    """Run network monitoring in background thread with proper stop handling"""
    global detector, recent_alerts, system_status, stop_event
    try:
        # Coerce model_type to xgboost only
        if model_type != 'xgboost':
            logger.warning(f"Invalid model_type '{model_type}', coercing to 'xgboost'")
            model_type = 'xgboost'

        logger.info(f"Starting {model_type} detection on {interface} (flow-based model)")
        validation = validate_model_files(model_type)
        logger.info(f"Model validation result: {validation}")

        system_status.update({
            'status': 'running',
            'started_at': time.time(),
            'total_packets': 0,
            'alert_count': 0,
            'error': None,
            'progress': 0,
            'is_monitoring': True,
            'can_stop': True,
            'stop_requested': False,
            'model_version': 'corrected' if validation['status'] == 'corrected_ready' else 'fallback',
            'feature_extraction': 'flow_based',
            'model_validation': validation
        })

        detector = NetworkIntrusionDetector(interface=interface, model_type='xgboost')
        logger.info("Model loaded successfully")

        alerts = []
        start_time = time.time()
        while not stop_event.is_set() and (time.time() - start_time) < duration:
            try:
                mini_duration = min(30, duration - (time.time() - start_time))
                if mini_duration <= 0:
                    break
                batch_alerts = detector.detect_intrusions(duration=int(mini_duration))
                alerts.extend(batch_alerts)
                elapsed = time.time() - start_time
                progress = min(100, (elapsed / duration) * 100)
                system_status['progress'] = progress
                if stop_event.is_set():
                    logger.info("Stop event detected in detection loop")
                    break
            except Exception as e:
                logger.error(f"Error in detection mini-loop: {e}")
                break

        if detector:
            remaining_flows = detector.stop_detection()
            logger.info(f"Forced stop completed, processed {remaining_flows} remaining flows")

        processed_alerts = process_flow_alerts(alerts)
        logger.info(f"Detection completed with {len(processed_alerts)} alerts")

        with alerts_lock:
            recent_alerts = processed_alerts + recent_alerts[:100]

        performance = detector.get_performance_metrics() if detector else {}
        system_status.update({
            'status': 'completed',
            'total_packets': performance.get('total_packets_analyzed', 0),
            'alert_count': len(processed_alerts),
            'progress': 100,
            'processing_time': performance.get('processing_time', 0),
            'performance_metrics': performance,
            'can_stop': False
        })

    except Exception as e:
        logger.error(f"Detection failed: {str(e)}", exc_info=True)
        system_status.update({
            'status': 'error',
            'error': str(e),
            'error_type': 'detection_error',
            'can_stop': False
        })
    finally:
        is_monitoring.clear()
        stop_event.clear()
        system_status.update({
            'is_monitoring': False,
            'last_updated': time.time(),
            'stop_requested': False
        })
        if detector:
            try:
                detector.stop_detection()
            except:
                pass

def background_pcap_analysis(pcap_path, model_type, max_packets):
    """Analyze PCAP file in background thread with proper stop handling"""
    global detector, recent_alerts, system_status, stop_event
    try:
        if model_type != 'xgboost':
            logger.warning(f"Invalid model_type '{model_type}', coercing to 'xgboost'")
            model_type = 'xgboost'

        logger.info(f"Starting PCAP analysis of {pcap_path} with {model_type}")
        validation = validate_model_files(model_type)
        logger.info(f"Model validation result: {validation}")

        system_status.update({
            'status': 'analyzing_pcap',
            'started_at': time.time(),
            'total_packets': 0,
            'alert_count': 0,
            'error': None,
            'progress': 0,
            'is_monitoring': True,
            'can_stop': True,
            'stop_requested': False,
            'model_version': 'corrected' if validation['status'] == 'corrected_ready' else 'fallback',
            'feature_extraction': 'flow_based',
            'pcap_file': os.path.basename(pcap_path),
            'max_packets': max_packets,
            'model_validation': validation
        })

        detector = NetworkIntrusionDetector(model_type='xgboost')
        alerts = detector.detect_intrusions_from_file(pcap_path, max_packets=max_packets)

        if stop_event.is_set():
            logger.info("Stop requested during PCAP analysis")
            if detector:
                detector.stop_detection()

        processed_alerts = process_flow_alerts(alerts)
        logger.info(f"PCAP analysis completed with {len(processed_alerts)} alerts")

        with alerts_lock:
            recent_alerts = processed_alerts + recent_alerts[:100]

        performance = detector.get_performance_metrics() if detector else {}
        system_status.update({
            'status': 'completed',
            'total_packets': performance.get('total_packets_analyzed', 0),
            'alert_count': len(processed_alerts),
            'progress': 100,
            'processing_time': performance.get('processing_time', 0),
            'performance_metrics': performance,
            'can_stop': False
        })

    except Exception as e:
        logger.error(f"PCAP analysis failed: {str(e)}", exc_info=True)
        system_status.update({
            'status': 'error',
            'error': str(e),
            'error_type': 'pcap_analysis_error',
            'can_stop': False
        })
    finally:
        is_monitoring.clear()
        stop_event.clear()
        system_status.update({
            'is_monitoring': False,
            'last_updated': time.time(),
            'stop_requested': False
        })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    system_status['is_monitoring'] = is_monitoring.is_set()
    system_status['timestamp'] = time.time()
    if detector:
        try:
            performance = detector.get_performance_metrics()
            system_status['current_performance'] = performance
        except:
            pass
    return jsonify(system_status)

@app.route('/api/alerts')
def api_alerts():
    try:
        limit = min(request.args.get('limit', 50, type=int), 1000)
        min_confidence = request.args.get('min_confidence', 0.0, type=float)
        with alerts_lock:
            fresh_alerts = load_alerts(limit)
            all_alerts = fresh_alerts + recent_alerts
            seen = set()
            unique_alerts = []
            for alert in all_alerts:
                if isinstance(alert, dict):
                    key = (alert.get('timestamp', 0), alert.get('confidence', 0))
                    if key not in seen:
                        seen.add(key)
                        unique_alerts.append(alert)
            filtered_alerts = unique_alerts[:limit]
            if min_confidence > 0:
                filtered_alerts = [a for a in filtered_alerts if a.get('confidence', 0) >= min_confidence]
            return jsonify(filtered_alerts)
    except Exception as e:
        logger.error(f"Error in api_alerts: {e}")
        return jsonify([])

@app.route('/api/start_detection', methods=['POST'])
def start_detection():
    global detection_thread, stop_event
    if is_monitoring.is_set():
        return jsonify({'status': 'error', 'message': 'Already monitoring'})
    try:
        data = request.json or {}
        interface = data.get('interface', 'lo')
        model_type = data.get('model_type', 'xgboost')
        duration = data.get('duration', 300)

        # Coerce to xgboost only
        if model_type != 'xgboost':
            model_type = 'xgboost'

        validation = validate_model_files(model_type)
        if validation['status'] == 'missing_files':
            return jsonify({
                'status': 'error',
                'message': f'Model files missing for {model_type}',
                'validation': validation
            })

        logger.info(f"Starting detection with {model_type} on {interface} for {duration}s")
        stop_event.clear()
        is_monitoring.set()
        detection_thread = threading.Thread(
            target=background_detection,
            args=(interface, model_type, duration),
            name=f"Detection-{model_type}-{int(time.time())}"
        )
        detection_thread.daemon = True
        detection_thread.start()

        return jsonify({
            'status': 'success',
            'message': 'Monitoring started with flow-based model',
            'model_validation': validation,
            'thread_id': detection_thread.name
        })
    except Exception as e:
        logger.error(f"Start detection failed: {str(e)}")
        is_monitoring.clear()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/start', methods=['POST'])
def start_monitoring():
    return start_detection()

@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    global detector, detection_thread, stop_event
    if not is_monitoring.is_set():
        return jsonify({'status': 'error', 'message': 'No detection running'})
    try:
        logger.info("Stop detection requested via API")
        system_status['stop_requested'] = True
        system_status['status'] = 'stopping'
        stop_event.set()

        remaining_flows = 0
        if detector:
            try:
                remaining_flows = detector.stop_detection()
                logger.info(f"Detector stopped, processed {remaining_flows} remaining flows")
            except Exception as e:
                logger.error(f"Error stopping detector: {e}")

        if detection_thread and detection_thread.is_alive():
            detection_thread.join(timeout=5.0)
            if detection_thread.is_alive():
                logger.warning("Detection thread did not stop within timeout")

        is_monitoring.clear()
        system_status.update({
            'status': 'stopped',
            'is_monitoring': False,
            'can_stop': False,
            'stop_requested': False,
            'last_updated': time.time()
        })
        return jsonify({
            'status': 'success',
            'message': 'Detection stopped successfully',
            'remaining_flows': remaining_flows,
            'thread_stopped': not (detection_thread and detection_thread.is_alive())
        })
    except Exception as e:
        logger.error(f"Error stopping detection: {e}")
        is_monitoring.clear()
        stop_event.set()
        system_status.update({
            'status': 'error',
            'error': f'Stop failed: {str(e)}',
            'is_monitoring': False,
            'can_stop': False
        })
        return jsonify({'status': 'error', 'message': f'Error stopping: {str(e)}'})

@app.route('/api/analyze_pcap', methods=['POST'])
def analyze_pcap_endpoint():
    global detection_thread, stop_event
    if is_monitoring.is_set():
        return jsonify({'status': 'error', 'message': 'Already busy'})
    try:
        data = request.json or {}
        pcap_path = data.get('pcap_path')
        model_type = data.get('model_type', 'xgboost')
        max_packets = data.get('max_packets', 5000)

        if not pcap_path:
            return jsonify({'status': 'error', 'message': 'PCAP path required'})
        if not os.path.exists(pcap_path):
            return jsonify({'status': 'error', 'message': f'File not found: {pcap_path}'})

        # Coerce to xgboost only
        if model_type != 'xgboost':
            model_type = 'xgboost'

        validation = validate_model_files(model_type)
        if validation['status'] == 'missing_files':
            return jsonify({
                'status': 'error',
                'message': f'Model files missing for {model_type}',
                'validation': validation
            })

        logger.info(f"Starting PCAP analysis of {pcap_path} with {model_type}")
        stop_event.clear()
        is_monitoring.set()
        detection_thread = threading.Thread(
            target=background_pcap_analysis,
            args=(pcap_path, model_type, max_packets),
            name=f"PCAP-Analysis-{model_type}-{int(time.time())}"
        )
        detection_thread.daemon = True
        detection_thread.start()

        return jsonify({
            'status': 'success',
            'message': 'Analysis started with flow-based model',
            'model_validation': validation,
            'thread_id': detection_thread.name
        })
    except Exception as e:
        logger.error(f"PCAP analysis failed: {str(e)}")
        is_monitoring.clear()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/analyze', methods=['POST'])
def analyze_pcap():
    return analyze_pcap_endpoint()

@app.route('/api/pcap_list')
def api_pcap_list():
    pcap_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    files = []
    try:
        os.makedirs(pcap_dir, exist_ok=True)
        for f in os.listdir(pcap_dir):
            if f.endswith('.pcap'):
                path = os.path.join(pcap_dir, f)
                size_bytes = os.path.getsize(path)
                files.append({
                    'name': f,
                    'path': path,
                    'size_mb': round(size_bytes / (1024 * 1024), 2),
                    'size_bytes': size_bytes,
                    'modified': os.path.getmtime(path),
                    'readable': os.access(path, os.R_OK)
                })
        files.sort(key=lambda x: x['modified'], reverse=True)
        logger.info(f"Found {len(files)} PCAP files in {pcap_dir}")
    except Exception as e:
        logger.error(f"PCAP list failed: {str(e)}")
    return jsonify(files)

@app.route('/api/pcap_files')
def api_pcap_files():
    return api_pcap_list()

@app.route('/api/stats')
def api_stats():
    try:
        alerts = load_alerts(1000)
        stats = {
            'total_alerts': len(alerts),
            'alert_types': {},
            'source_ips': {},
            'destination_ports': {},
            'hourly_distribution': {i: 0 for i in range(24)},
            'severity_distribution': {'High': 0, 'Medium': 0, 'Low': 0, 'Info': 0},
            'confidence_stats': {'min': 0, 'max': 0, 'mean': 0, 'high_confidence_count': 0},
            'model_distribution': {}
        }
        confidence_values = []
        for alert in alerts:
            try:
                if not isinstance(alert, dict):
                    logger.warning(f"Skipping non-dict alert in stats: {type(alert)}")
                    continue
                protocol = alert.get('protocol', 'Unknown')
                stats['alert_types'][protocol] = stats['alert_types'].get(protocol, 0) + 1
                src_ip = alert.get('source_ip', 'Unknown')
                stats['source_ips'][src_ip] = stats['source_ips'].get(src_ip, 0) + 1
                dst_port = str(alert.get('destination_port', 'Unknown'))
                stats['destination_ports'][dst_port] = stats['destination_ports'].get(dst_port, 0) + 1
                model = alert.get('model', 'xgboost')
                stats['model_distribution'][model] = stats['model_distribution'].get(model, 0) + 1
                severity = alert.get('severity', 'Info')
                if severity in stats['severity_distribution']:
                    stats['severity_distribution'][severity] += 1
                else:
                    stats['severity_distribution']['Info'] += 1
                confidence = float(alert.get('confidence', 0.0))
                confidence_values.append(confidence)
                if confidence >= 0.8:
                    stats['confidence_stats']['high_confidence_count'] += 1
                timestamp = alert.get('timestamp', 0)
                if timestamp:
                    hour = datetime.fromtimestamp(float(timestamp)).hour
                    stats['hourly_distribution'][hour] += 1
            except Exception as e:
                logger.error(f"Error processing alert in stats: {e}")
                continue
        if confidence_values:
            stats['confidence_stats']['min'] = min(confidence_values)
            stats['confidence_stats']['max'] = max(confidence_values)
            stats['confidence_stats']['mean'] = sum(confidence_values) / len(confidence_values)
        stats['alert_types'] = dict(sorted(stats['alert_types'].items(), key=lambda x: -x[1])[:10])
        stats['source_ips'] = dict(sorted(stats['source_ips'].items(), key=lambda x: -x[1])[:10])
        stats['destination_ports'] = dict(sorted(stats['destination_ports'].items(), key=lambda x: -x[1])[:10])
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in api_stats: {e}", exc_info=True)
        return jsonify({
            'error': 'Failed to generate statistics',
            'total_alerts': 0,
            'alert_types': {},
            'source_ips': {},
            'destination_ports': {},
            'hourly_distribution': {i: 0 for i in range(24)},
            'severity_distribution': {'High': 0, 'Medium': 0, 'Low': 0, 'Info': 0},
            'confidence_stats': {'min': 0, 'max': 0, 'mean': 0, 'high_confidence_count': 0}
        }), 500

@app.route('/api/model_info')
def api_model_info():
    """Get information about available models and their status (XGBoost only)"""
    model_info = {}
    for model_type, config in MODEL_CONFIG.items():
        validation = validate_model_files(model_type)
        model_info[model_type] = {
            'metrics': config['metrics'],
            'validation': validation,
            'files': {
                'model_file': config['model_file'],
                'scaler_file': config['scaler_file'],
                'fallback_model': config['fallback_model']
            }
        }
    return jsonify(model_info)

@app.route('/api/validate_features', methods=['POST'])
def validate_features():
    """Validate feature extraction for debugging"""
    try:
        data = request.json or {}
        model_type = data.get('model_type', 'xgboost')
        if model_type != 'xgboost':
            return jsonify({'status': 'error', 'error': f'Invalid model type: {model_type}'})
        validation = validate_model_files(model_type)
        test_detector = NetworkIntrusionDetector(model_type='xgboost')
        validation_result = {
            'model_type': 'xgboost',
            'model_loaded': True,
            'feature_count': 53,
            'flow_tracking_enabled': hasattr(test_detector, 'flows'),
            'status': 'ready',
            'model_validation': validation,
            'feature_extraction': 'flow_based',
            'detector_initialized': True
        }
        return jsonify(validation_result)
    except Exception as e:
        logger.error(f"Feature validation failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'model_loaded': False,
            'detector_initialized': False
        })

@app.route('/api/system_health')
def system_health():
    """Get overall system health status"""
    health = {
        'timestamp': time.time(),
        'status': 'healthy',
        'components': {
            'flask_app': 'running',
            'model_files': 'checking',
            'data_directory': 'checking',
            'alerts_directory': 'checking'
        },
        'models': {},
        'directories': {},
        'threads': {
            'monitoring_active': is_monitoring.is_set(),
            'stop_event_set': stop_event.is_set(),
            'detection_thread_alive': detection_thread.is_alive() if detection_thread else False
        }
    }
    try:
        for model_type in MODEL_CONFIG.keys():
            validation = validate_model_files(model_type)
            health['models'][model_type] = validation['status']
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        alerts_dir = os.path.join(os.path.dirname(__file__), '..', 'alerts')
        health['directories']['data'] = 'exists' if os.path.exists(data_dir) else 'missing'
        health['directories']['alerts'] = 'exists' if os.path.exists(alerts_dir) else 'missing'
        health['components']['model_files'] = 'ready' if any(v == 'corrected_ready' for v in health['models'].values()) else 'degraded'
        health['components']['data_directory'] = health['directories']['data']
        health['components']['alerts_directory'] = health['directories']['alerts']
        if all(status in ['running', 'ready', 'exists'] for status in health['components'].values()):
            health['status'] = 'healthy'
        else:
            health['status'] = 'degraded'
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        health['status'] = 'unhealthy'
        health['error'] = str(e)
    return jsonify(health)

# Graceful shutdown handler
def shutdown_handler(signum, frame):
    logger.info(f"Received shutdown signal {signum}")
    if is_monitoring.is_set():
        logger.info("Stopping detection due to shutdown")
        stop_event.set()
        if detector:
            detector.stop_detection()
    if detection_thread and detection_thread.is_alive():
        logger.info("Waiting for detection thread to finish...")
        detection_thread.join(timeout=5.0)
    logger.info("Graceful shutdown complete")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'alerts'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data'), exist_ok=True)

    logger.info("🛡️  AI-Powered NIDS Dashboard Starting...")
    logger.info("Model Configuration:")
    for model_type, config in MODEL_CONFIG.items():
        logger.info(f"  {model_type}: {config['model_file']} (accuracy: {config['metrics']['accuracy']})")

    logger.info("Registered routes:")
    for rule in app.url_map.iter_rules():
        logger.info(f"  {rule.endpoint}: {rule}")

    logger.info("Validating model files on startup:")
    for model_type in MODEL_CONFIG.keys():
        validation = validate_model_files(model_type)
        logger.info(f"  {model_type}: {validation['status']}")

    logger.info("🚀 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
