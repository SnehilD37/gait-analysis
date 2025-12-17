import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
VIDEO_PATH = r"C:\Users\91896\Desktop\Proj1\running_sample.mp4" 
IMU_PATH = r"C:\Users\91896\Desktop\Proj1\WISDM_ar_v1.1_raw.txt" 

CLIP_DURATION = 3
def load_imu_data(csv_path):
    print("Loading IMU Data...")
    if not os.path.exists(csv_path):
        print(f"Warning: IMU file not found at {csv_path}. Skipping Graph 2.")
        return None

    try:
        df = pd.read_csv(
            csv_path, 
            header=None, 
            names=['user', 'activity', 'time', 'x', 'y', 'z'],
            on_bad_lines='skip' 
        )
        df['z'] = df['z'].astype(str).str.replace(';', '', regex=False)
        df['z'] = pd.to_numeric(df['z'], errors='coerce')
        df.dropna(subset=['z'], inplace=True)
        return df
    except Exception as e:
        print(f"Error loading IMU: {e}")
        return None

def process_video_data(video_path):
    print("Processing Video...")
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found at {video_path}. Skipping Video Graphs.")
        return None, None

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_clip = int(CLIP_DURATION * fps)

    clip_counts = {"Normal": 0, "Moderate": 0, "Fatigue-like": 0}
    ankle_y_coords = []
    timestamps = []
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        progress = frame_idx / total_frames
        if progress < 0.25: label = "Normal"
        elif progress < 0.75: label = "Moderate"
        else: label = "Fatigue-like"
        
        if frame_idx > 0 and frame_idx % frames_per_clip == 0:
            clip_counts[label] += 1

        if frame_idx % 2 == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            if results.pose_landmarks:
                ankle_y = results.pose_landmarks.landmark[28].y
                ankle_y_coords.append(ankle_y)
                timestamps.append(frame_idx / fps)
        
        frame_idx += 1
        
    cap.release()
    pose.close()
    return clip_counts, (timestamps, ankle_y_coords)

if __name__ == "__main__":
    imu_data = load_imu_data(IMU_PATH)
    counts, trajectory = process_video_data(VIDEO_PATH)

    if counts:
        plt.figure(figsize=(8, 5))
        plt.bar(counts.keys(), counts.values(), color=['green', 'orange', 'red'])
        plt.title("Graph 1: Number of Samples in Each Class")
        plt.ylabel("Count")
        plt.grid(axis='y', alpha=0.5)

    if imu_data is not None and not imu_data.empty:
        plt.figure(figsize=(10, 4))
        subset = imu_data.iloc[:60]
        plt.plot(range(len(subset)), subset['z'], label='Z-Axis Acc', color='purple')
        plt.title("Graph 2: IMU Acceleration vs Time (3s Window)")
        plt.xlabel("Sample Index")
        plt.ylabel("Acceleration (m/s^2)")
        plt.legend()
        plt.grid(True)

    if trajectory and len(trajectory[1]) > 0:
        t, y = trajectory
        plt.figure(figsize=(10, 5))
        plt.plot(t, y, color='blue', alpha=0.7)
        plt.title("Graph 3: Pose Keypoint Trajectory (Right Ankle)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Vertical Position (Normalized)")
        plt.gca().invert_yaxis()
        plt.grid(True)

    print("Done! displaying graphs...")
    plt.show()