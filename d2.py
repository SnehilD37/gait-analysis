import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ==========================================
# 1. CONFIGURATION
# ==========================================
dataset_path = r"C:\Users\91896\Desktop\Proj1\dataset_videos"
OUTPUT_CSV = "gait_features.csv"
SHOW_GRAPHS_FOR_FIRST_VIDEO = True 

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def calculate_angle(a, b, c):
    """ Calculates angle 0-180 degrees at point b (Knee) """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def process_video(video_path, is_first_video=False):
    """ Extracts features from a video for BOTH legs. Plots graphs if is_first_video is True. """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 15.0 
    
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return []

    video_data = []
    frame_idx = 0
    
    # Plotting containers
    plot_data = {
        'timestamps': [],
        'r_knee_angles': [], 'l_knee_angles': [],
        'r_ankle_y': [], 'l_ankle_y': []
    }

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # --- EXTRACT RIGHT LEG ---
            r_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            r_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            r_angle = calculate_angle(r_hip, r_knee, r_ankle)

            # --- EXTRACT LEFT LEG ---
            l_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            l_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            l_angle = calculate_angle(l_hip, l_knee, l_ankle)
            
            current_time = frame_idx / fps
            
            video_data.append({
                'Time': current_time,
                'R_Knee_Angle': r_angle, 'L_Knee_Angle': l_angle,
                'R_Ankle_Y': r_ankle[1], 'L_Ankle_Y': l_ankle[1],
                'Source_Video': os.path.basename(video_path)
            })
            
            if is_first_video:
                plot_data['timestamps'].append(current_time)
                plot_data['r_knee_angles'].append(r_angle)
                plot_data['l_knee_angles'].append(l_angle)
                plot_data['r_ankle_y'].append(r_ankle[1])
                plot_data['l_ankle_y'].append(l_ankle[1])
            
        frame_idx += 1
    
    cap.release()
    pose.close()
    
    # --- POST-PROCESSING: Calculate Cadence (Avg of Both Legs) ---
    r_ankle_array = np.array([row['R_Ankle_Y'] for row in video_data])
    l_ankle_array = np.array([row['L_Ankle_Y'] for row in video_data])
    
    cadence = 0
    if len(r_ankle_array) > 10:
        # Get peaks for both
        peaks_r, _ = find_peaks(r_ankle_array, distance=int(fps/2))
        peaks_l, _ = find_peaks(l_ankle_array, distance=int(fps/2))
        
        # Total steps = R + L
        total_steps = len(peaks_r) + len(peaks_l)
        duration_min = (frame_idx/fps) / 60
        if duration_min > 0:
            cadence = total_steps / duration_min 

    for row in video_data:
        row['Clip_Cadence'] = cadence

    # --- PLOTTING ---
    if is_first_video and SHOW_GRAPHS_FOR_FIRST_VIDEO and len(plot_data['timestamps']) > 0:
        print(f"\nGenerating Graphs for: {os.path.basename(video_path)}")
        t = plot_data['timestamps']
        
        # Graph 1: Knee Angle
        plt.figure(figsize=(12, 5))
        plt.plot(t, plot_data['r_knee_angles'], label='Right Knee', color='blue')
        plt.plot(t, plot_data['l_knee_angles'], label='Left Knee', color='red', linestyle='--')
        plt.title(f"Bilateral Knee Angles")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (deg)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Graph 2: Stride Detection (BOTH LEGS)
        plt.figure(figsize=(12, 5))
        
        # Plot signals
        plt.plot(t, plot_data['r_ankle_y'], label='Right Ankle Y', color='blue', alpha=0.5)
        plt.plot(t, plot_data['l_ankle_y'], label='Left Ankle Y', color='red', alpha=0.5)
        
        # Detect & Plot Peaks for Graph
        y_r = np.array(plot_data['r_ankle_y'])
        y_l = np.array(plot_data['l_ankle_y'])
        
        p_r, _ = find_peaks(y_r, distance=int(fps/2))
        p_l, _ = find_peaks(y_l, distance=int(fps/2))
        
        if len(p_r) > 0:
             plt.plot(np.array(t)[p_r], y_r[p_r], "x", color='black', markersize=8, label='Step (Right)')
        if len(p_l) > 0:
             plt.plot(np.array(t)[p_l], y_l[p_l], "o", color='black', markerfacecolor='none', markersize=8, label='Step (Left)')

        plt.title(f"Stride Detection (Left vs Right) - Cadence: {int(cadence)} SPM")
        plt.xlabel("Time (s)")
        plt.gca().invert_yaxis() 
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return video_data

# ==========================================
# 3. MAIN LOOP
# ==========================================
def extract_normal_features():
    if not os.path.exists(dataset_path):
        print(f"Error: Folder '{dataset_path}' does not exist.")
        return

    all_features = []
    files = [f for f in os.listdir(dataset_path) if f.endswith((".mp4", ".avi", ".mov", ".gif"))]
    
    if not files:
        print("No video files found!")
        return

    print(f"Found {len(files)} videos. Processing...")

    for i, file in enumerate(files):
        path = os.path.join(dataset_path, file)
        features = process_video(path, is_first_video=(i == 0))
        all_features.extend(features)
        print(f"  [{i+1}/{len(files)}] Processed {file}")
        
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccess! Saved {len(df)} rows to {OUTPUT_CSV}")
    else:
        print("Failed to extract data.")

if __name__ == "__main__":
    extract_normal_features()