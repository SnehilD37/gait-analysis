import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import joblib

# ==========================================
# 1. CONFIGURATION
# ==========================================
VIDEO_CSV_PATH = r"C:\Users\91896\Desktop\Proj1\gait_features.csv" 
IMU_PATH = r"C:\Users\91896\Desktop\Proj1\WISDM_ar_v1.1_raw.txt" 
WINDOW_SIZE_SEC = 3 
VIDEO_FPS = 30 
IMU_HZ = 20 

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def get_video_windows(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except:
        return pd.DataFrame()
    
    duration = df['Time'].max() - df['Time'].min()
    fps = len(df) / duration if duration > 0 else 30
    samples_per_window = int(WINDOW_SIZE_SEC * fps)
    windows = []
    num_windows = int(len(df) / samples_per_window)

    for i in range(num_windows):
        start = i * samples_per_window
        end = (i + 1) * samples_per_window
        chunk = df.iloc[start:end]
        if len(chunk) < 2: continue
        
        rom_right = chunk['R_Knee_Angle'].max() - chunk['R_Knee_Angle'].min()
        r_y = chunk['R_Ankle_Y'].values
        peaks, _ = find_peaks(r_y, distance=int(fps/3)) 
        cadence = (len(peaks) / WINDOW_SIZE_SEC) * 60 
        
        windows.append({'Knee_ROM': rom_right, 'Cadence': cadence, 'Step_Count': len(peaks)})
    return pd.DataFrame(windows)

def get_imu_windows(txt_path, num_windows_needed):
    try:
        df = pd.read_csv(txt_path, header=None, names=['u','a','t','x','y','z'], on_bad_lines='skip')
        df['z'] = pd.to_numeric(df['z'].astype(str).str.replace(';', '', regex=False), errors='coerce')
        df.dropna(subset=['z'], inplace=True)
    except:
        return pd.DataFrame()

    samples_per_window = int(WINDOW_SIZE_SEC * IMU_HZ)
    windows = []
    for i in range(num_windows_needed):
        start = i * samples_per_window
        end = (i + 1) * samples_per_window
        if end > len(df): break
        chunk = df.iloc[start:end]['z'].values
        rms = np.sqrt(np.mean(chunk**2))
        peaks, _ = find_peaks(chunk, height=5, distance=10)
        variability = np.std(np.diff(peaks) / IMU_HZ) if len(peaks) > 1 else 0
        windows.append({'RMS_Acc': rms, 'Stride_Var': variability})
    return pd.DataFrame(windows)

# ==========================================
# 3. TRAINING & EVALUATION LOOP
# ==========================================
def train_and_evaluate():
    # --- Load Data ---
    vid_features = get_video_windows(VIDEO_CSV_PATH)
    if vid_features.empty: return
    imu_features = get_imu_windows(IMU_PATH, len(vid_features))
    
    if imu_features.empty: X = vid_features
    else:
        min_len = min(len(vid_features), len(imu_features))
        X = pd.concat([vid_features.iloc[:min_len].reset_index(drop=True), 
                       imu_features.iloc[:min_len].reset_index(drop=True)], axis=1)

    print(f"\nData Loaded: {len(X)} windows")

    # --- Define "Ground Truth" (Time-based Assumption) ---
    total_samples = len(X)
    train_limit = int(total_samples * 0.40)  
    fatigue_start = int(total_samples * 0.60) 
    
    y_true = np.zeros(total_samples) 
    y_true[:train_limit] = 1         # Normal
    y_true[fatigue_start:] = -1      # Fatigue

    # --- Train Model (Normal Only) ---
    X_train_normal = X.iloc[:train_limit]
    print(f"Training on first {train_limit} samples (Normal Data)...")
    
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train_normal)

    # --- Evaluate ---
    y_pred = model.predict(X)
    scores = model.decision_function(X)

    mask = y_true != 0 
    y_true_eval = y_true[mask]
    y_pred_eval = y_pred[mask]

    acc = accuracy_score(y_true_eval, y_pred_eval)
    prec = precision_score(y_true_eval, y_pred_eval, pos_label=-1)
    rec = recall_score(y_true_eval, y_pred_eval, pos_label=-1)

    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Fatigue Detection (Recall): {rec*100:.2f}%")

    # ==========================================
    # 4. PLOTTING GRAPHS (IN 3 WINDOWS)
    # ==========================================

    # --- Window 1: Anomaly Score Over Time ---
    plt.figure(figsize=(10, 6))
    plt.plot(scores, color='blue', label='Anomaly Score')
    plt.axhline(0, color='red', linestyle='--', label='Threshold (0.0)')
    plt.axvspan(0, train_limit, color='green', alpha=0.1, label='Training Phase (Normal)')
    plt.axvspan(fatigue_start, total_samples, color='red', alpha=0.1, label='Expected Fatigue')
    plt.title("Graph 1: Fatigue Detection Score (Drop = Fatigue)")
    plt.xlabel("Time (Windows)")
    plt.ylabel("Score")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    print("Displaying Graph 1...")
    plt.show()

    # --- Window 2: Confusion Matrix ---
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true_eval, y_pred_eval, labels=[1, -1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fatigue'], 
                yticklabels=['Normal', 'Fatigue'])
    plt.title(f"Graph 2: Confusion Matrix\n(Accuracy: {acc*100:.1f}%)")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label (Time-Based)")
    plt.tight_layout()
    print("Displaying Graph 2...")
    plt.show()

    # --- Window 3: Accuracy Metrics ---
    plt.figure(figsize=(8, 6))
    metrics = [acc, prec, rec]
    metric_names = ['Accuracy', 'Precision', 'Recall']
    bars = plt.bar(metric_names, metrics, color=['#4CAF50', '#2196F3', '#FF9800'])
    plt.ylim(0, 1.1)
    plt.title("Graph 3: Model Performance Metrics")
    plt.ylabel("Score (0-1)")
    
    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height*100:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    print("Displaying Graph 3...")
    plt.show()

    # Save Model
    joblib.dump(model, "anomaly_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()