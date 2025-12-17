import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ==========================================
# 1. CONFIGURATION
# ==========================================
VIDEO_CSV_PATH = r"C:\Users\91896\Desktop\Proj1\gait_features.csv" 

# ==========================================
# 2. DATA LOADING & PROCESSING
# ==========================================
def apply_smoothing(df, window=5):
    """
    Applies a rolling average to smooth out camera jitter.
    This drastically improves the separation between 'Normal' and 'Mild'.
    """
    # Select only numeric columns for smoothing
    cols_to_smooth = ['R_Knee_Angle', 'L_Knee_Angle']
    
    # We group by video source so we don't smooth across different files
    if 'Source_Video' in df.columns:
        df[cols_to_smooth] = df.groupby('Source_Video')[cols_to_smooth].transform(lambda x: x.rolling(window, min_periods=1).mean())
    else:
        df[cols_to_smooth] = df[cols_to_smooth].rolling(window, min_periods=1).mean()
        
    return df

def load_and_augment_data(path):
    print("Loading data...")
    try:
        df_normal = pd.read_csv(path)
    except:
        return pd.DataFrame()

    if 'L_Knee_Angle' not in df_normal.columns:
        print("Error: Left leg data missing.")
        return pd.DataFrame()

    # --- STEP 1: SMOOTH THE RAW DATA ---
    print(f"Applying smoothing (Window=5)...")
    df_normal = apply_smoothing(df_normal)

    # --- STEP 2: AUGMENTATION (SIMULATE INJURY) ---
    # Class 1: NORMAL
    df_normal['Condition'] = 'Normal'
    
    # Class 2: MILD ASYMMETRY (15% reduction)
    df_mild = df_normal.copy()
    df_mild['L_Knee_Angle'] = df_mild['L_Knee_Angle'] * 0.85 
    df_mild['Condition'] = 'Mild Asymmetry'
    
    # Class 3: SEVERE ASYMMETRY (40% reduction)
    df_severe = df_normal.copy()
    df_severe['L_Knee_Angle'] = df_severe['L_Knee_Angle'] * 0.60
    df_severe['Condition'] = 'Severe Asymmetry'
    
    # Combine
    df_combined = pd.concat([df_normal, df_mild, df_severe], axis=0).reset_index(drop=True)
    return df_combined

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def extract_asymmetry_features(df):
    # Absolute Difference
    df['Knee_Diff_Abs'] = abs(df['R_Knee_Angle'] - df['L_Knee_Angle'])
    
    # Symmetry Index
    # Adding larger epsilon (1.0) to stabilize division for small angles
    df['Symmetry_Index'] = (df['R_Knee_Angle'] - df['L_Knee_Angle']) / \
                           (0.5 * (df['R_Knee_Angle'] + df['L_Knee_Angle']) + 1.0) * 100
    return df

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def gait_analysis():
    # A. Prepare Data
    df_raw = load_and_augment_data(VIDEO_CSV_PATH)
    if df_raw.empty: return

    df = extract_asymmetry_features(df_raw)
    
    # --- CRITICAL CHANGE: FEATURE SELECTION ---
    # We DROP the raw angles. The model must learn pure asymmetry.
    # We only keep relative metrics.
    feature_cols = ['Knee_Diff_Abs', 'Symmetry_Index']
    
    X = df[feature_cols]
    y = df['Condition']
    
    # B. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # C. Train Classifier
    print("\nTraining Asymmetry Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # D. Evaluate
    y_pred = clf.predict(X_test)
    print("\n" + "="*30)
    print("PHASE 5 RESULTS")
    print("="*30)
    print(classification_report(y_test, y_pred))
    
    # ==========================================
    # 5. PLOTTING GRAPHS
    # ==========================================
    
    # --- Window 1: Asymmetry Distribution (Smoothed) ---
    plt.figure(figsize=(10, 5))
    subset = df.sample(n=300, random_state=42).sort_values(by='Symmetry_Index')
    colors = {'Normal': 'green', 'Mild Asymmetry': 'orange', 'Severe Asymmetry': 'red'}
    plt.scatter(range(len(subset)), subset['Symmetry_Index'], c=subset['Condition'].map(colors), alpha=0.6)
    plt.title("Graph 1: Smoothed Symmetry Index")
    plt.ylabel("Symmetry Index (%)")
    plt.xlabel("Samples")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Window 2: Feature Importance ---
    plt.figure(figsize=(8, 5))
    importances = clf.feature_importances_
    plt.barh(feature_cols, importances, color='teal')
    plt.title("Graph 2: Feature Importance")
    plt.tight_layout()
    plt.show()

    # --- Window 3: Confusion Matrix ---
    plt.figure(figsize=(6, 5))
    labels_order = ["Normal", "Mild Asymmetry", "Severe Asymmetry"]
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=labels_order, yticklabels=labels_order)
    plt.title("Graph 3: Accuracy Heatmap")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    
    joblib.dump(clf, "asymmetry_model.pkl")

if __name__ == "__main__":
    gait_analysis()