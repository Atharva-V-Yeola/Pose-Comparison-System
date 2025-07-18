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
        self.fps = 0

        # Activity Timing variables
        self.activity_events = {}

        # Hit/Miss variables
        self.hit_miss_events = {}

        # Distance variables
        self.distance_events = {}

        # Feedback messages
        self.feedback_messages = []

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

    def calculate_distance(self, point1, point2, frame_width, frame_height):
        """Calculate Euclidean distance between two points in pixels"""
        try:
            x1 = point1.x * frame_width
            y1 = point1.y * frame_height
            x2 = point2.x * frame_width
            y2 = point2.y * frame_height
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        except:
            return 0.0

    def _check_activity_start_condition(self, landmarks, frame_number, timestamp, activity_id):
        """Example: Detect activity start when left elbow angle is below 90 degrees"""
        # Example for 'Wall Target Pass': left elbow angle below 90
        left_elbow_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        if activity_id == 'wall_target_pass' and left_elbow_angle < 90:
            return True
        
        # Example for 'Balance Statue': right hip is significantly higher than left hip (indicating single leg stance)
        # This is a very basic heuristic and would need refinement
        right_hip_y = landmarks[24].y
        left_hip_y = landmarks[23].y
        if activity_id == 'balance_statue' and (right_hip_y - left_hip_y) > 0.05: # Threshold for hip difference
            return True

        return False

    def _check_activity_end_condition(self, landmarks, frame_number, timestamp, activity_id):
        """Example: Detect activity end when left elbow angle is above 160 degrees"""
        left_elbow_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        if activity_id == 'wall_target_pass' and left_elbow_angle > 160:
            return True

        # Example for 'Balance Statue': both hips are level again
        right_hip_y = landmarks[24].y
        left_hip_y = landmarks[23].y
        if activity_id == 'balance_statue' and abs(right_hip_y - left_hip_y) < 0.02: # Threshold for hip difference
            return True

        return False

    def _check_hit_condition(self, landmarks, frame_width, frame_height, hit_event_id):
        """Example: Detect a 'hit' for 'Wall Target Pass' (right wrist in target area)"""
        if hit_event_id == 'wall_target_hit':
            right_wrist = landmarks[16]
            # Define a target area (e.g., top-right quadrant for a wall target)
            target_x_min = 0.7 * frame_width
            target_y_max = 0.3 * frame_height

            if (right_wrist.x * frame_width > target_x_min and 
                right_wrist.y * frame_height < target_y_max):
                return True
        return False

    def _measure_specific_distance(self, landmarks, frame_width, frame_height, distance_event_id):
        """Example: Measure distance for 'Walk & Tap Relay' (distance covered by hip)"""
        if distance_event_id == 'walk_tap_relay_distance':
            # Track a key point, e.g., mid-hip (average of left and right hip)
            mid_hip_x = (landmarks[23].x + landmarks[24].x) / 2 * frame_width
            mid_hip_y = (landmarks[23].y + landmarks[24].y) / 2 * frame_height
            return (mid_hip_x, mid_hip_y)
        return None

    def analyze_frame(self, frame, frame_number, timestamp):
        """Analyze a single frame for pose and angles, and custom metrics"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        frame_data = {
            'frame_number': frame_number,
            'timestamp': timestamp,
            'pose_detected': False,
            'activity_status': 'none',
            'hit_status': 'none',
            'measured_distance': 0.0,
            'feedback_message': ''
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

            # --- Activity Timing --- 
            # This part needs to be configured based on the specific exercise being analyzed
            # For demonstration, let's assume we are looking for 'wall_target_pass' activity
            current_activity_id = 'wall_target_pass' # Or 'balance_statue', etc. - this would come from user input

            if current_activity_id not in self.activity_events:
                self.activity_events[current_activity_id] = {'start_frame': None, 'end_frame': None, 'duration': 0}

            if self.activity_events[current_activity_id]['start_frame'] is None and \
               self._check_activity_start_condition(landmarks, frame_number, timestamp, current_activity_id):
                self.activity_events[current_activity_id]['start_frame'] = frame_number
                frame_data['activity_status'] = 'start'
                frame_data['feedback_message'] = 'Activity started! Keep going.'
            elif self.activity_events[current_activity_id]['start_frame'] is not None and \
                 self.activity_events[current_activity_id]['end_frame'] is None and \
                 self._check_activity_end_condition(landmarks, frame_number, timestamp, current_activity_id):
                self.activity_events[current_activity_id]['end_frame'] = frame_number
                if self.fps > 0:
                    self.activity_events[current_activity_id]['duration'] = \
                        (self.activity_events[current_activity_id]['end_frame'] - self.activity_events[current_activity_id]['start_frame']) / self.fps
                frame_data['activity_status'] = 'end'
                frame_data['feedback_message'] = f'Activity completed in {self.activity_events[current_activity_id]["duration"]:.2f} seconds. Great job!'
            elif self.activity_events[current_activity_id]['start_frame'] is not None and \
                 self.activity_events[current_activity_id]['end_frame'] is None:
                frame_data['activity_status'] = 'in_progress'

            # --- Hit/Miss Detection ---
            current_hit_event_id = 'wall_target_hit' # This would also come from user input
            if current_hit_event_id not in self.hit_miss_events:
                self.hit_miss_events[current_hit_event_id] = {'hit_detected': False}

            if not self.hit_miss_events[current_hit_event_id]['hit_detected'] and \
               self._check_hit_condition(landmarks, frame.shape[1], frame.shape[0], current_hit_event_id):
                self.hit_miss_events[current_hit_event_id]['hit_detected'] = True
                frame_data['hit_status'] = 'hit'
                frame_data['feedback_message'] = 'Target hit! Excellent accuracy.'
            elif self.hit_miss_events[current_hit_event_id]['hit_detected']:
                frame_data['hit_status'] = 'already_hit'

            # --- Distance Measurement --- 
            current_distance_event_id = 'walk_tap_relay_distance' # This would also come from user input
            measured_point = self._measure_specific_distance(landmarks, frame.shape[1], frame.shape[0], current_distance_event_id)
            if measured_point:
                frame_data['measured_distance_x'] = measured_point[0]
                frame_data['measured_distance_y'] = measured_point[1]
                # For actual distance covered, you'd need to track the point over frames
                # For simplicity, we'll just record the current point's coordinates

            # Generate general feedback based on pose confidence or alignment
            if frame_data['pose_confidence'] < 70 and frame_data['feedback_message'] == '':
                frame_data['feedback_message'] = 'Adjust your posture for better detection.'
            elif frame_data['body_alignment'] < 70 and frame_data['feedback_message'] == '':
                frame_data['feedback_message'] = 'Try to keep your body more aligned.'
            
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
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        frame_number = 0
        self.analysis_data = []
        self.activity_events = {}
        self.hit_miss_events = {}
        self.distance_events = {}
        self.feedback_messages = [] # Reset feedback messages for new video
        
        print(f"Processing video: {total_frames} frames at {self.fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_number / self.fps
            annotated_frame, frame_data = self.analyze_frame(frame, frame_number, timestamp)
            
            self.analysis_data.append(frame_data)
            if frame_data['feedback_message']:
                self.feedback_messages.append({
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'message': frame_data['feedback_message']
                })
            
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

Activity Timing:
"""
        for activity_id, event_data in self.activity_events.items():
            if event_data['start_frame'] is not None and event_data['end_frame'] is not None:
                report += f"- Activity '{activity_id}' Duration: {event_data['duration']:.2f} seconds\n"
            else:
                report += f"- Activity '{activity_id}' not completed or detected.\n"

        report += "\nHit/Miss Detection:\n"
        for hit_event_id, event_data in self.hit_miss_events.items():
            report += f"- Hit Event '{hit_event_id}': {'Detected' if event_data['hit_detected'] else 'Not Detected'}\n"

        report += "\nDistance Measurement (Sample):\n"
        # Add more detailed distance reporting here if needed
        report += f"- Shoulder Distance (Avg): {df['shoulder_distance'].mean():.2f} pixels\n\n"

        report += "Angle Analysis:\n"
        
        angle_stats = self.get_angle_statistics()
        for angle_name, stats in angle_stats.items():
            report += f"""
{angle_name.replace('_', ' ').title()}:
  - Average: {stats['mean']}°
  - Range: {stats['min']}° - {stats['max']}°
  - Variation (Std Dev): {stats['std']}°
"""
        
        report += "\nFeedback Messages:\n"
        if self.feedback_messages:
            for msg in self.feedback_messages:
                report += f"- [Frame {msg['frame_number']:.0f}, {msg['timestamp']:.2f}s]: {msg['message']}\n"
        else:
            report += "No specific feedback generated.\n"

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


