from flask import Flask, render_template, jsonify, request
import os
print(f"Current working directory: {os.getcwd()}")
import json
import sys
import time
import logging
import threading
import joblib
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.realtime_detection import NetworkIntrusionDetector

app = Flask(__name__)

# Model configuration for corrected models
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
    },
    'bilstm': {
        'model_file': 'bilstm_model_corrected.h5',
        'scaler_file': 'scaler.pkl',
        'fallback_model': 'bilstm_model.h5',
        'metrics': {
            'accuracy': 0.9415,
            'precision': 0.7988,
            'recall': 0.9793,
            'f1_score': 0.8799,
            'status': 'corrected_features'
        }
    }
}

# Thread-safe globals
detector = None
is_monitoring = threading.Event()
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
    'feature_extraction': 'flow_based'
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
                    
                    # Handle both old and new alert file formats
                    if isinstance(file_data, dict) and 'alerts' in file_data:
                        # New format with session metadata
                        file_alerts = file_data['alerts']
                    elif isinstance(file_data, list):
                        # Old format - direct list of alerts
                        file_alerts = file_data
                    else:
                        logger.warning(f"Unknown alert file format in {file}: {type(file_data)}")
                        continue
                    
                    # Process each alert with validation
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
        # Validate that alert is a dictionary
        if not isinstance(alert, dict):
            logger.warning(f"Invalid alert type: {type(alert)} - {alert}")
            return None
        
        # Extract and validate required fields with defaults
        processed_alert = {
            'timestamp': float(alert.get('timestamp', time.time())),
            'source_ip': str(alert.get('source_ip', 'Flow-based')),
            'destination_ip': str(alert.get('destination_ip', 'Analysis')),
            'destination_port': int(alert.get('destination_port', 0)) if alert.get('destination_port') else 0,
            'protocol': str(alert.get('protocol', 'Unknown')),
            'confidence': float(alert.get('confidence', 0.0)),
            'model': str(alert.get('model', 'unknown')),
            'threshold_used': float(alert.get('threshold_used', 0.0)),
            'alert_reason': str(alert.get('alert_reason', 'ML Detection')),
            'alert_type': 'Flow-based Detection',
            'severity': get_alert_severity(float(alert.get('confidence', 0.0))),
            'packet_info': alert.get('packet_info', {}),
            'detection_method': 'ML-based'
        }
        
        # Validate processed alert
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
    """Validate that required model files exist"""
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
        # Check corrected model file
        model_path = os.path.join(model_dir, config.get('model_file', ''))
        validation_result['model_file_exists'] = os.path.exists(model_path)
        
        # Check scaler file
        scaler_path = os.path.join(model_dir, config.get('scaler_file', ''))
        validation_result['scaler_file_exists'] = os.path.exists(scaler_path)
        
        # Check fallback model
        fallback_path = os.path.join(model_dir, config.get('fallback_model', ''))
        validation_result['fallback_available'] = os.path.exists(fallback_path)
        
        # Determine status
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
    """Run network monitoring in background thread with corrected models"""
    global detector, recent_alerts, system_status
    
    try:
        logger.info(f"Starting {model_type} detection on {interface} (using corrected flow-based model)")
        
        # Validate model files first
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
            'model_version': 'corrected' if validation['status'] == 'corrected_ready' else 'fallback',
            'feature_extraction': 'flow_based',
            'model_validation': validation
        })

        # Initialize detector with corrected model
        detector = NetworkIntrusionDetector(interface=interface, model_type=model_type)
        logger.info("Model loaded successfully")

        # Run detection with flow-based feature extraction
        alerts = detector.detect_intrusions(duration=duration)
        
        # Process alerts for dashboard compatibility
        processed_alerts = process_flow_alerts(alerts)
        logger.info(f"Detection completed with {len(processed_alerts)} alerts")

        # Update alerts thread-safely
        with alerts_lock:
            recent_alerts = processed_alerts + recent_alerts[:100]

        # Get performance metrics from detector
        performance = detector.get_performance_metrics()
        
        # Update final status
        system_status.update({
            'status': 'completed',
            'total_packets': performance.get('total_packets_analyzed', 0),
            'alert_count': len(processed_alerts),
            'progress': 100,
            'processing_time': performance.get('processing_time', 0),
            'performance_metrics': performance
        })

    except Exception as e:
        logger.error(f"Detection failed: {str(e)}", exc_info=True)
        system_status.update({
            'status': 'error',
            'error': str(e),
            'error_type': 'detection_error'
        })
    finally:
        is_monitoring.clear()
        system_status.update({
            'is_monitoring': False,
            'last_updated': time.time()
        })

def background_pcap_analysis(pcap_path, model_type, max_packets):
    """Analyze PCAP file in background thread with corrected models"""
    global detector, recent_alerts, system_status
    
    try:
        logger.info(f"Starting PCAP analysis of {pcap_path} with {model_type} (corrected flow-based model)")
        
        # Validate model files first
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
            'model_version': 'corrected' if validation['status'] == 'corrected_ready' else 'fallback',
            'feature_extraction': 'flow_based',
            'pcap_file': os.path.basename(pcap_path),
            'max_packets': max_packets,
            'model_validation': validation
        })

        # Initialize detector with corrected model
        detector = NetworkIntrusionDetector(model_type=model_type)
        
        # Run analysis with flow-based feature extraction
        alerts = detector.detect_intrusions_from_file(pcap_path, max_packets=max_packets)
        
        # Process alerts for dashboard compatibility
        processed_alerts = process_flow_alerts(alerts)
        logger.info(f"PCAP analysis completed with {len(processed_alerts)} alerts")

        # Update alerts thread-safely
        with alerts_lock:
            recent_alerts = processed_alerts + recent_alerts[:100]

        # Get performance metrics from detector
        performance = detector.get_performance_metrics()

        # Update final status
        system_status.update({
            'status': 'completed',
            'total_packets': performance.get('total_packets_analyzed', 0),
            'alert_count': len(processed_alerts),
            'progress': 100,
            'processing_time': performance.get('processing_time', 0),
            'performance_metrics': performance
        })

    except Exception as e:
        logger.error(f"PCAP analysis failed: {str(e)}", exc_info=True)
        system_status.update({
            'status': 'error',
            'error': str(e),
            'error_type': 'pcap_analysis_error'
        })
    finally:
        is_monitoring.clear()
        system_status.update({
            'is_monitoring': False,
            'last_updated': time.time()
        })

@app.route('/')
def index():
    """Render main dashboard page"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """Get current system status with enhanced information"""
    # Update is_monitoring flag from Event object
    system_status['is_monitoring'] = is_monitoring.is_set()
    system_status['timestamp'] = time.time()
    
    # Add model status information
    if detector:
        try:
            performance = detector.get_performance_metrics()
            system_status['current_performance'] = performance
        except:
            pass
    
    return jsonify(system_status)

@app.route('/api/alerts')
def api_alerts():
    """Get recent alerts with enhanced filtering and validation"""
    try:
        limit = min(request.args.get('limit', 50, type=int), 1000)
        min_confidence = request.args.get('min_confidence', 0.0, type=float)
        
        with alerts_lock:
            # Load fresh alerts and combine with recent ones
            fresh_alerts = load_alerts(limit)
            all_alerts = fresh_alerts + recent_alerts
            
            # Remove duplicates based on timestamp and confidence
            seen = set()
            unique_alerts = []
            for alert in all_alerts:
                if isinstance(alert, dict):
                    key = (alert.get('timestamp', 0), alert.get('confidence', 0))
                    if key not in seen:
                        seen.add(key)
                        unique_alerts.append(alert)
            
            # Apply limit
            filtered_alerts = unique_alerts[:limit]
            
            # Apply confidence filter if specified
            if min_confidence > 0:
                filtered_alerts = [a for a in filtered_alerts if a.get('confidence', 0) >= min_confidence]
            
            return jsonify(filtered_alerts)
            
    except Exception as e:
        logger.error(f"Error in api_alerts: {e}")
        return jsonify([])

@app.route('/api/start_detection', methods=['POST'])
def start_detection():
    """Start live network monitoring with corrected models"""
    if is_monitoring.is_set():
        return jsonify({'status': 'error', 'message': 'Already monitoring'})

    try:
        data = request.json or {}
        interface = data.get('interface', 'lo')
        model_type = data.get('model_type', 'xgboost')
        duration = data.get('duration', 300)

        # Validate model type
        if model_type not in MODEL_CONFIG:
            return jsonify({'status': 'error', 'message': f'Invalid model type: {model_type}'})

        # Validate model files
        validation = validate_model_files(model_type)
        if validation['status'] == 'missing_files':
            return jsonify({
                'status': 'error', 
                'message': f'Model files missing for {model_type}',
                'validation': validation
            })

        logger.info(f"Starting detection with {model_type} on {interface} for {duration}s")
        is_monitoring.set()
        thread = threading.Thread(
            target=background_detection,
            args=(interface, model_type, duration)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'status': 'success', 
            'message': 'Monitoring started with corrected flow-based model',
            'model_validation': validation
        })

    except Exception as e:
        logger.error(f"Start detection failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """Start live network monitoring (original endpoint)"""
    return start_detection()

@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    """Stop ongoing detection"""
    if not is_monitoring.is_set():
        return jsonify({'status': 'error', 'message': 'No detection running'})
    
    # Set flag to stop
    is_monitoring.clear()
    system_status['status'] = 'stopping'
    
    return jsonify({'status': 'success', 'message': 'Stopping detection'})

@app.route('/api/analyze_pcap', methods=['POST'])
def analyze_pcap_endpoint():
    """Start PCAP file analysis with corrected models"""
    if is_monitoring.is_set():
        return jsonify({'status': 'error', 'message': 'Already busy'})

    try:
        data = request.json or {}
        pcap_path = data.get('pcap_path')
        model_type = data.get('model_type', 'xgboost')
        max_packets = data.get('max_packets', 5000)

        # Validate inputs
        if not pcap_path:
            return jsonify({'status': 'error', 'message': 'PCAP path required'})
        
        if not os.path.exists(pcap_path):
            return jsonify({'status': 'error', 'message': f'File not found: {pcap_path}'})

        if model_type not in MODEL_CONFIG:
            return jsonify({'status': 'error', 'message': f'Invalid model type: {model_type}'})

        # Validate model files
        validation = validate_model_files(model_type)
        if validation['status'] == 'missing_files':
            return jsonify({
                'status': 'error', 
                'message': f'Model files missing for {model_type}',
                'validation': validation
            })

        logger.info(f"Starting PCAP analysis of {pcap_path} with {model_type}")
        is_monitoring.set()
        thread = threading.Thread(
            target=background_pcap_analysis,
            args=(pcap_path, model_type, max_packets)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'status': 'success', 
            'message': 'Analysis started with corrected flow-based model',
            'model_validation': validation
        })

    except Exception as e:
        logger.error(f"PCAP analysis failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/analyze', methods=['POST'])
def analyze_pcap():
    """Start PCAP file analysis (original endpoint)"""
    return analyze_pcap_endpoint()

@app.route('/api/pcap_list')
def api_pcap_list():
    """List available PCAP files with enhanced information"""
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
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x['modified'], reverse=True)
        logger.info(f"Found {len(files)} PCAP files in {pcap_dir}")
        
    except Exception as e:
        logger.error(f"PCAP list failed: {str(e)}")

    return jsonify(files)

@app.route('/api/pcap_files')
def api_pcap_files():
    """List available PCAP files (original endpoint)"""
    return api_pcap_list()

@app.route('/api/stats')
def api_stats():
    """Get detection statistics with enhanced analytics and error handling"""
    try:
        alerts = load_alerts(1000)
        stats = {
            'total_alerts': len(alerts),
            'alert_types': {},
            'source_ips': {},
            'destination_ports': {},
            'hourly_distribution': {i: 0 for i in range(24)},
            'severity_distribution': {'High': 0, 'Medium': 0, 'Low': 0, 'Info': 0},
            'confidence_stats': {
                'min': 0,
                'max': 0,
                'mean': 0,
                'high_confidence_count': 0
            },
            'model_distribution': {}
        }

        confidence_values = []
        
        for alert in alerts:
            try:
                # Validate alert is a dictionary
                if not isinstance(alert, dict):
                    logger.warning(f"Skipping non-dict alert in stats: {type(alert)}")
                    continue
                
                # Protocol stats
                protocol = alert.get('protocol', 'Unknown')
                stats['alert_types'][protocol] = stats['alert_types'].get(protocol, 0) + 1
                
                # Source IP stats
                src_ip = alert.get('source_ip', 'Unknown')
                stats['source_ips'][src_ip] = stats['source_ips'].get(src_ip, 0) + 1
                
                # Destination port stats
                dst_port = str(alert.get('destination_port', 'Unknown'))
                stats['destination_ports'][dst_port] = stats['destination_ports'].get(dst_port, 0) + 1
                
                # Model distribution
                model = alert.get('model', 'unknown')
                stats['model_distribution'][model] = stats['model_distribution'].get(model, 0) + 1
                
                # Severity distribution
                severity = alert.get('severity', 'Info')
                if severity in stats['severity_distribution']:
                    stats['severity_distribution'][severity] += 1
                else:
                    stats['severity_distribution']['Info'] += 1
                
                # Confidence statistics
                confidence = float(alert.get('confidence', 0.0))
                confidence_values.append(confidence)
                if confidence >= 0.8:
                    stats['confidence_stats']['high_confidence_count'] += 1
                
                # Time distribution
                try:
                    timestamp = alert.get('timestamp', 0)
                    if timestamp:
                        hour = datetime.fromtimestamp(float(timestamp)).hour
                        stats['hourly_distribution'][hour] += 1
                except (ValueError, OSError, OverflowError):
                    pass

            except Exception as e:
                logger.error(f"Error processing alert in stats: {e}")
                continue

        # Calculate confidence statistics
        if confidence_values:
            stats['confidence_stats']['min'] = min(confidence_values)
            stats['confidence_stats']['max'] = max(confidence_values)
            stats['confidence_stats']['mean'] = sum(confidence_values) / len(confidence_values)

        # Sort and limit results
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
    """Get information about available models and their status"""
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
        
        if model_type not in MODEL_CONFIG:
            return jsonify({
                'status': 'error',
                'error': f'Invalid model type: {model_type}'
            })
        
        # Validate model files
        validation = validate_model_files(model_type)
        
        # Test detector initialization
        test_detector = NetworkIntrusionDetector(model_type=model_type)
        
        validation_result = {
            'model_type': model_type,
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
        'directories': {}
    }
    
    try:
        # Check model files for both types
        for model_type in MODEL_CONFIG.keys():
            validation = validate_model_files(model_type)
            health['models'][model_type] = validation['status']
        
        # Check directories
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        alerts_dir = os.path.join(os.path.dirname(__file__), '..', 'alerts')
        
        health['directories']['data'] = 'exists' if os.path.exists(data_dir) else 'missing'
        health['directories']['alerts'] = 'exists' if os.path.exists(alerts_dir) else 'missing'
        
        # Update component status
        health['components']['model_files'] = 'ready' if any(v == 'corrected_ready' for v in health['models'].values()) else 'degraded'
        health['components']['data_directory'] = health['directories']['data']
        health['components']['alerts_directory'] = health['directories']['alerts']
        
        # Overall status
        if all(status in ['running', 'ready', 'exists'] for status in health['components'].values()):
            health['status'] = 'healthy'
        else:
            health['status'] = 'degraded'
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        health['status'] = 'unhealthy'
        health['error'] = str(e)
    
    return jsonify(health)

if __name__ == '__main__':
    # Create required directories
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'alerts'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data'), exist_ok=True)
    
    # Print model configuration
    logger.info("Model Configuration:")
    for model_type, config in MODEL_CONFIG.items():
        logger.info(f"  {model_type}: {config['model_file']} (accuracy: {config['metrics']['accuracy']})")
    
    # Print all registered routes for debugging
    logger.info("Registered routes:")
    for rule in app.url_map.iter_rules():
        logger.info(f"  {rule.endpoint}: {rule}")
    
    # Validate model files on startup
    logger.info("Validating model files on startup:")
    for model_type in MODEL_CONFIG.keys():
        validation = validate_model_files(model_type)
        logger.info(f"  {model_type}: {validation['status']}")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )
