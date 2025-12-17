import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.signal import find_peaks

# ==========================================
# 1. CONFIGURATION
# ==========================================
VIDEO_PATH = "gait_features.csv"
IMU_PATH = r"C:\Users\91896\Desktop\Proj1\WISDM_ar_v1.1_raw.txt"
WINDOW_SIZE = 3.0  # Seconds per window
VIDEO_FPS = 30     # Approx
IMU_HZ = 20        # WISDM standard

# ==========================================
# 2. FEATURE ENGINEERING (FUSION)
# ==========================================
def extract_fusion_features():
    print("--- STARTING PHASE 6: MULTIMODAL FUSION ---")
    
    # 1. Load Video Data
    try:
        df_vid = pd.read_csv(VIDEO_PATH)
        duration_v = df_vid['Time'].max()
    except:
        print("Error: gait_features.csv not found.")
        return pd.DataFrame()

    # 2. Load IMU Data
    try:
        df_imu = pd.read_csv(IMU_PATH, header=None, names=['u','a','t','x','y','z'], on_bad_lines='skip')
        df_imu['z'] = pd.to_numeric(df_imu['z'].astype(str).str.replace(';', '', regex=False), errors='coerce')
        df_imu.dropna(subset=['z'], inplace=True)
    except:
        print("Error loading IMU file.")
        return pd.DataFrame()

    # 3. Create Synchronized Windows
    num_windows = int(min(duration_v, len(df_imu)/IMU_HZ) / WINDOW_SIZE)
    print(f"Creating {num_windows} fused windows...")

    data_fusion = []

    for i in range(num_windows):
        t_start = i * WINDOW_SIZE
        t_end = (i + 1) * WINDOW_SIZE

        # A. Extract Video Features
        vid_chunk = df_vid[(df_vid['Time'] >= t_start) & (df_vid['Time'] < t_end)]
        if len(vid_chunk) < 5: continue 
        
        feat_knee_rom = vid_chunk['R_Knee_Angle'].max() - vid_chunk['R_Knee_Angle'].min()
        peaks_v, _ = find_peaks(vid_chunk['R_Ankle_Y'].values, distance=10)
        feat_cadence_v = len(peaks_v)

        # B. Extract IMU Features
        idx_start = int(t_start * IMU_HZ)
        idx_end = int(t_end * IMU_HZ)
        imu_chunk = df_imu.iloc[idx_start:idx_end]['z'].values
        
        if len(imu_chunk) < 5: continue

        feat_rms = np.sqrt(np.mean(imu_chunk**2))
        feat_std = np.std(imu_chunk)
        
        # C. Create Label (Simulated Fatigue)
        label = "Normal" if i < (num_windows/2) else "Fatigue"

        data_fusion.append({
            'Video_Knee_ROM': feat_knee_rom,
            'Video_Cadence': feat_cadence_v,
            'IMU_RMS': feat_rms,
            'IMU_Variability': feat_std,
            'Label': label
        })

    return pd.DataFrame(data_fusion)

# ==========================================
# 3. MODEL TRAINING & COMPARISON
# ==========================================
def compare_models(df):
    if df.empty: return

    X_vid = df[['Video_Knee_ROM', 'Video_Cadence']]
    X_imu = df[['IMU_RMS', 'IMU_Variability']]
    X_fusion = df.drop(columns=['Label']) 
    y = df['Label']

    ids_train, ids_test, y_train, y_test = train_test_split(range(len(df)), y, test_size=0.3, random_state=42)

    # 1. Train Video-Only
    clf_v = RandomForestClassifier(random_state=42)
    clf_v.fit(X_vid.iloc[ids_train], y_train)
    y_pred_v = clf_v.predict(X_vid.iloc[ids_test])
    acc_v = accuracy_score(y_test, y_pred_v)

    # 2. Train IMU-Only
    clf_i = RandomForestClassifier(random_state=42)
    clf_i.fit(X_imu.iloc[ids_train], y_train)
    y_pred_i = clf_i.predict(X_imu.iloc[ids_test])
    acc_i = accuracy_score(y_test, y_pred_i)

    # 3. Train FUSION
    clf_f = RandomForestClassifier(random_state=42)
    clf_f.fit(X_fusion.iloc[ids_train], y_train)
    y_pred_f = clf_f.predict(X_fusion.iloc[ids_test])
    acc_f = accuracy_score(y_test, y_pred_f)

    print("\n" + "="*30)
    print("PHASE 6: FUSION RESULTS")
    print("="*30)
    print(f"Video Accuracy: {acc_v*100:.1f}%")
    print(f"IMU Accuracy:   {acc_i*100:.1f}%")
    print(f"FUSION Accuracy:{acc_f*100:.1f}%")

    # ==========================================
    # 4. PLOTTING
    # ==========================================
    
    # --- Graph 1: Accuracy Bar Chart ---
    plt.figure(figsize=(8, 5))
    bars = plt.bar(['Video', 'IMU', 'Fusion'], [acc_v, acc_i, acc_f], color=['#3498db', '#e67e22', '#2ecc71'])
    plt.ylim(0, 1.1)
    plt.title("Graph 1: Accuracy Comparison")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height*100:.1f}%', ha='center', va='bottom')
    plt.show()

    # --- Graph 2: Feature Importance ---
    plt.figure(figsize=(8, 5))
    plt.barh(X_fusion.columns, clf_f.feature_importances_, color='teal')
    plt.title("Graph 2: Feature Importance (Fusion Model)")
    plt.tight_layout()
    plt.show()

    # --- Graph 3: Confusion Matrices Comparison (Requested) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Helper to plot one matrix
    def plot_cm(ax, y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred, labels=["Normal", "Fatigue"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["Normal", "Fatigue"], yticklabels=["Normal", "Fatigue"])
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plot_cm(axes[0], y_test, y_pred_v, f"Video Only (Acc: {acc_v*100:.0f}%)")
    plot_cm(axes[1], y_test, y_pred_i, f"IMU Only (Acc: {acc_i*100:.0f}%)")
    plot_cm(axes[2], y_test, y_pred_f, f"Fusion (Acc: {acc_f*100:.0f}%)")
    
    plt.suptitle("Graph 3: Confusion Matrices Comparison")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = extract_fusion_features()
    compare_models(df)