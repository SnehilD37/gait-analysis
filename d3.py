import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

IMU_PATH = r"C:\Users\91896\Desktop\Proj1\WISDM_ar_v1.1_raw.txt"

SAMPLE_RATE = 20
CUTOFF_FREQ = 5
def load_imu_data(csv_path):
    """ Loads WISDM data, skipping broken lines. """
    print("Loading IMU Data...")
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

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """ Applies a low-pass Butterworth filter to remove noise. """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def extract_data():
    df = load_imu_data(IMU_PATH)
    if df is None or df.empty:
        print("Failed to load data.")
        return

    data_slice = df.iloc[:2000].copy()
    
    raw_z = data_slice['z'].values
    filtered_z = butter_lowpass_filter(raw_z, CUTOFF_FREQ, SAMPLE_RATE)
    
    peaks, _ = find_peaks(filtered_z, height=5, distance=10)
    
    peak_times = peaks / SAMPLE_RATE
    step_intervals = np.diff(peak_times)
    
    avg_step_time = np.mean(step_intervals) if len(step_intervals) > 0 else 0
    stride_variability = np.std(step_intervals) if len(step_intervals) > 0 else 0
    
    rms_acc = np.sqrt(np.mean(filtered_z**2))

    print("\n" + "="*30)
    print("       IMU REPORT    ")
    print("="*30)
    print(f"Total Steps Detected: {len(peaks)}")
    print(f"Avg Step Time: {avg_step_time:.3f} sec")
    print(f"Stride Variability: {stride_variability:.4f} s")
    print(f"RMS Acceleration: {rms_acc:.2f} m/s^2")
    print("="*30)

    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(len(raw_z))/SAMPLE_RATE, raw_z, label='Raw Z-Acc', color='lightgray', alpha=0.6)
    plt.plot(np.arange(len(filtered_z))/SAMPLE_RATE, filtered_z, label='Filtered Z-Acc (10Hz)', color='purple')
    plt.plot(peaks/SAMPLE_RATE, filtered_z[peaks], "x", color='red', label='Steps Detected')
    plt.title("IMU Acceleration with Step Detection")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    if len(step_intervals) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(step_intervals, bins=15, color='orange', edgecolor='black', alpha=0.7)
        plt.title("Histogram of Step Times")
        plt.xlabel("Time between steps (seconds)")
        plt.ylabel("Frequency")
        plt.grid(axis='y')
        plt.show()

    rolling_rms = pd.Series(filtered_z).rolling(window=20).apply(lambda x: np.sqrt(np.mean(x**2)))
    
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(len(rolling_rms))/SAMPLE_RATE, rolling_rms, color='green')
    plt.title("RMS Acceleration Over Time (Intensity)")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS Value")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    extract_data()