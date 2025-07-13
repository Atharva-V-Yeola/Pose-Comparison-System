
import cv2
import mediapipe as mp
import numpy as np
import time

class PoseComparator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_pose_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None, None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results, image

    def detect_pose_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        return results

    def extract_landmarks(self, results):
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            return np.array(landmarks)
        return None

    def compare_poses(self, landmarks1, landmarks2):
        if landmarks1 is None or landmarks2 is None:
            return float("inf") # Return infinity if one of the poses is not detected

        # Simple Euclidean distance for now, can be improved with more sophisticated metrics
        # Only compare x and y coordinates for 2D pose comparison
        diff = landmarks1[:, :2] - landmarks2[:, :2]
        distance = np.linalg.norm(diff)
        return distance

    def draw_landmarks(self, image, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return image

if __name__ == '__main__':
    comparator = PoseComparator()

    # Load reference image
    reference_image_path = '/home/ubuntu/test_pose_image.png'
    ref_results, ref_image = comparator.detect_pose_image(reference_image_path)
    ref_landmarks = comparator.extract_landmarks(ref_results)

    if ref_landmarks is None:
        print("Error: Could not detect pose in reference image. Exiting.")
        exit()

    cap = cv2.VideoCapture(0) # 0 for default camera

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1) # Flip horizontally for selfie-view

        current_results = comparator.detect_pose_frame(frame)
        current_landmarks = comparator.extract_landmarks(current_results)

        if current_landmarks is not None:
            distance = comparator.compare_poses(ref_landmarks, current_landmarks)
            cv2.putText(frame, f'Distance: {distance:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            frame = comparator.draw_landmarks(frame, current_results)
        else:
            cv2.putText(frame, 'No pose detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Live Pose Comparison', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


