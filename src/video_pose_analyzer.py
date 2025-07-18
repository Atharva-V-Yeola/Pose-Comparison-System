import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
import math
import os

class VideoPoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key body angles to track
        self.angle_definitions = {
            'left_elbow': [11, 13, 15],  # left_shoulder, left_elbow, left_wrist
            'right_elbow': [12, 14, 16],  # right_shoulder, right_elbow, right_wrist
            'left_knee': [23, 25, 27],   # left_hip, left_knee, left_ankle
            'right_knee': [24, 26, 28],  # right_hip, right_knee, right_ankle
            'left_shoulder': [13, 11, 23],  # left_elbow, left_shoulder, left_hip
            'right_shoulder': [14, 12, 24], # right_elbow, right_shoulder, right_hip
            'spine': [11, 23, 25],       # left_shoulder, left_hip, left_knee (approximation)
        }
        
        self.analysis_data = []
        
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        try:
            # Convert to numpy arrays
            a = np.array([point1.x, point1.y])
            b = np.array([point2.x, point2.y])
            c = np.array([point3.x, point3.y])
            
            # Calculate vectors
            ba = a - b
            bc = c - b
            
            # Calculate angle
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            
            return np.degrees(angle)
        except:
            return 0.0
    
    def analyze_frame(self, frame, frame_number, timestamp):
        """Analyze a single frame for pose and angles"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        frame_data = {
            'frame_number': frame_number,
            'timestamp': timestamp,
            'pose_detected': False
        }
        
        if results.pose_landmarks:
            frame_data['pose_detected'] = True
            landmarks = results.pose_landmarks.landmark
            
            # Calculate all defined angles
            for angle_name, landmark_indices in self.angle_definitions.items():
                try:
                    point1 = landmarks[landmark_indices[0]]
                    point2 = landmarks[landmark_indices[1]]
                    point3 = landmarks[landmark_indices[2]]
                    
                    angle = self.calculate_angle(point1, point2, point3)
                    frame_data[f'{angle_name}_angle'] = round(angle, 2)
                except:
                    frame_data[f'{angle_name}_angle'] = 0.0
            
            # Calculate pose quality metrics
            frame_data['pose_confidence'] = self._calculate_pose_confidence(landmarks)
            frame_data['body_alignment'] = self._calculate_body_alignment(landmarks)
            
            # Draw pose on frame
            annotated_frame = frame.copy()
            self.mp_drawing.draw_landmarks(
                annotated_frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Add angle annotations
            self._draw_angle_annotations(annotated_frame, landmarks)
            
            return annotated_frame, frame_data
        
        return frame, frame_data
    
    def _calculate_pose_confidence(self, landmarks):
        """Calculate overall pose detection confidence"""
        visible_landmarks = sum(1 for lm in landmarks if lm.visibility > 0.5)
        total_landmarks = len(landmarks)
        return round((visible_landmarks / total_landmarks) * 100, 2)
    
    def _calculate_body_alignment(self, landmarks):
        """Calculate body alignment score based on key points"""
        try:
            # Check shoulder alignment
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            
            # Check hip alignment
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            hip_diff = abs(left_hip.y - right_hip.y)
            
            # Calculate alignment score (lower difference = better alignment)
            alignment_score = max(0, 100 - ((shoulder_diff + hip_diff) * 1000))
            return round(alignment_score, 2)
        except:
            return 0.0
    
    def _draw_angle_annotations(self, frame, landmarks):
        """Draw angle measurements on the frame"""
        height, width = frame.shape[:2]
        
        for angle_name, landmark_indices in self.angle_definitions.items():
            try:
                point2 = landmarks[landmark_indices[1]]  # Joint point
                x = int(point2.x * width)
                y = int(point2.y * height)
                
                # Get the angle value from the last calculated data
                if self.analysis_data:
                    angle_value = self.analysis_data[-1].get(f'{angle_name}_angle', 0)
                    cv2.putText(frame, f'{angle_name}: {angle_value:.1f}°', 
                              (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 1)
            except:
                continue
    
    def process_video(self, video_path, output_path=None):
        """Process entire video and return analysis results"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        self.analysis_data = []
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_number / fps
            annotated_frame, frame_data = self.analyze_frame(frame, frame_number, timestamp)
            
            self.analysis_data.append(frame_data)
            
            if output_path:
                out.write(annotated_frame)
            
            frame_number += 1
            
            # Progress indicator
            if frame_number % 30 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        cap.release()
        if output_path:
            out.release()
        
        print("Video processing completed!")
        return self.analysis_data
    
    def save_analysis_to_csv(self, csv_path):
        """Save analysis data to CSV file"""
        if not self.analysis_data:
            print("No analysis data to save")
            return
        
        df = pd.DataFrame(self.analysis_data)
        df.to_csv(csv_path, index=False)
        print(f"Analysis data saved to: {csv_path}")
        return csv_path
    
    def get_angle_statistics(self):
        """Get statistical summary of angles throughout the video"""
        if not self.analysis_data:
            return {}
        
        df = pd.DataFrame(self.analysis_data)
        angle_columns = [col for col in df.columns if col.endswith('_angle')]
        
        stats = {}
        for col in angle_columns:
            angle_name = col.replace('_angle', '')
            stats[angle_name] = {
                'mean': round(df[col].mean(), 2),
                'min': round(df[col].min(), 2),
                'max': round(df[col].max(), 2),
                'std': round(df[col].std(), 2)
            }
        
        return stats
    
    def generate_analysis_report(self):
        """Generate a comprehensive analysis report"""
        if not self.analysis_data:
            return "No analysis data available"
        
        df = pd.DataFrame(self.analysis_data)
        total_frames = len(df)
        frames_with_pose = df['pose_detected'].sum()
        detection_rate = (frames_with_pose / total_frames) * 100
        
        report = f"""
VIDEO POSE ANALYSIS REPORT
==========================

General Statistics:
- Total Frames Analyzed: {total_frames}
- Frames with Pose Detected: {frames_with_pose}
- Pose Detection Rate: {detection_rate:.1f}%
- Average Pose Confidence: {df['pose_confidence'].mean():.1f}%
- Average Body Alignment: {df['body_alignment'].mean():.1f}%

Angle Analysis:
"""
        
        angle_stats = self.get_angle_statistics()
        for angle_name, stats in angle_stats.items():
            report += f"""
{angle_name.replace('_', ' ').title()}:
  - Average: {stats['mean']}°
  - Range: {stats['min']}° - {stats['max']}°
  - Variation (Std Dev): {stats['std']}°
"""
        
        return report

# Example usage and testing
if __name__ == "__main__":
    analyzer = VideoPoseAnalyzer()
    
    # Test with a sample video (you would replace this with actual video path)
    # video_path = "sample_exercise_video.mp4"
    # output_path = "analyzed_video.mp4"
    # csv_path = "pose_analysis.csv"
    
    # analysis_data = analyzer.process_video(video_path, output_path)
    # analyzer.save_analysis_to_csv(csv_path)
    # print(analyzer.generate_analysis_report())
    
    print("VideoPoseAnalyzer class created successfully!")

