from flask import Blueprint, request, jsonify, current_app
import cv2
import numpy as np
import base64
import os
import csv
from datetime import datetime
from src.pose_comparison import PoseComparator

pose_bp = Blueprint('pose', __name__)

# Global variables to store reference pose
reference_landmarks = None
pose_comparator = PoseComparator()

@pose_bp.route('/upload_reference', methods=['POST'])
def upload_reference():
    global reference_landmarks
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save the uploaded file
        filename = 'reference_pose.jpg'
        filepath = os.path.join(current_app.static_folder, filename)
        file.save(filepath)
        
        # Process the reference image
        results, image = pose_comparator.detect_pose_image(filepath)
        reference_landmarks = pose_comparator.extract_landmarks(results)
        
        if reference_landmarks is None:
            return jsonify({'error': 'No pose detected in the reference image'}), 400
        
        return jsonify({'message': 'Reference pose uploaded successfully', 'landmarks_count': len(reference_landmarks)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@pose_bp.route('/compare_pose', methods=['POST'])
def compare_pose():
    global reference_landmarks
    
    try:
        if reference_landmarks is None:
            return jsonify({'error': 'No reference pose uploaded'}), 400
        
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect pose in current frame
        results = pose_comparator.detect_pose_frame(frame)
        current_landmarks = pose_comparator.extract_landmarks(results)
        
        if current_landmarks is None:
            return jsonify({'pose_detected': False, 'distance': 'inf'})
        
        # Compare poses
        distance = pose_comparator.compare_poses(reference_landmarks, current_landmarks)
        
        # Log to CSV
        log_pose_comparison(distance, current_landmarks is not None)
        
        return jsonify({
            'pose_detected': True,
            'distance': float(distance),
            'landmarks_count': len(current_landmarks)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@pose_bp.route('/get_csv_data', methods=['GET'])
def get_csv_data():
    try:
        csv_path = os.path.join(current_app.static_folder, 'pose_comparison_log.csv')
        
        if not os.path.exists(csv_path):
            return jsonify({'data': []})
        
        data = []
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        
        return jsonify({'data': data})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def log_pose_comparison(distance, pose_detected):
    """Log pose comparison data to CSV file"""
    csv_path = os.path.join(current_app.static_folder, 'pose_comparison_log.csv')
    
    # Check if file exists, if not create with headers
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'pose_detected', 'distance', 'accuracy_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Calculate accuracy score (inverse of distance, normalized)
        if distance == float('inf'):
            accuracy_score = 0
            distance_str = 'inf'
        else:
            accuracy_score = max(0, 100 - (distance * 10)) if pose_detected else 0
            distance_str = round(distance, 4)
        
        writer.writerow({
            'timestamp': datetime.now().isoformat(),
            'pose_detected': pose_detected,
            'distance': distance_str,
            'accuracy_score': round(accuracy_score, 2)
        })

