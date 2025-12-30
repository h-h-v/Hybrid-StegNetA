import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# --- CONFIGURATION ---
# Path to your data (Matches your screenshot)
BASE_PATH = os.path.join('Stegware_Project_Data', 'Stegware_Project_Data', 'data')

# DATA LIMITS (Kept small for speed, but enough for a demo)
MAX_IMAGES_PER_CLASS = 2500  
IMG_SIZE = 32
BATCH_SIZE = 32

# *** CHANGED TO 50 EPOCHS ***
EPOCHS = 50 

# --- 1. DATA LOADING ---
def load_subset_data(base_dir, img_size, max_samples):
    print(f"--- Loading Data from {base_dir} ---")
    
    clean_dir = os.path.join(base_dir, 'final_clean')
    stego_base = os.path.join(base_dir, 'stegware')
    
    data = []
    labels = []
    
    # Load Clean
    print(f"Loading CLEAN images...")
    clean_files = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    clean_files = clean_files[:max_samples] 
    
    for file in clean_files:
        path = os.path.join(clean_dir, file)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            data.append(img)
            labels.append(0)

    # Load Stego (Balanced)
    samples_per_family = max_samples // 4
    families = ['lsb1_stego', 'lsb3_stego', 'ppm_stego', 'parity_stego']
    family_data_indices = {} 
    
    for family in families:
        print(f"Loading {family}...")
        fam_dir = os.path.join(stego_base, family)
        if not os.path.exists(fam_dir):
            continue
            
        files = [f for f in os.listdir(fam_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        files = files[:samples_per_family]
        
        start_idx = len(data)
        for file in files:
            path = os.path.join(fam_dir, file)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                data.append(img)
                labels.append(1)
        
        family_data_indices[family] = list(range(start_idx, len(data)))

    X = np.array(data).reshape(-1, img_size, img_size, 1) / 255.0
    y = np.array(labels)
    
    return X, y, family_data_indices

# --- 2. MODEL DEFINITIONS ---

def build_hybrid_stegnet(input_shape):
    inputs = layers.Input(shape=input_shape)
    # High Pass Filter
    x = layers.Conv2D(16, 5, padding='same', activation=None)(inputs) 
    
    # Residual Block
    res = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Add()([x, res])
    x = layers.MaxPooling2D()(x)
    
    # Attention
    att = layers.Conv2D(1, 1, activation='sigmoid')(x)
    x = layers.Multiply()([x, att])
    
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs, outputs, name="Hybrid_StegNetA")

def build_baseline_cnn(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ], name="Baseline_CNN")
    return model

def build_resnet_lite(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    res = x
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Add()([x, res])
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs, outputs, name="ResNet_Lite")

# --- 3. EXECUTION ---

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Running on GPU: {gpus[0]}")

    # Load
    X, y, family_indices = load_subset_data(BASE_PATH, IMG_SIZE, MAX_IMAGES_PER_CLASS)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_shape = (IMG_SIZE, IMG_SIZE, 1)
    models_list = [
        build_hybrid_stegnet(input_shape),
        build_baseline_cnn(input_shape),
        build_resnet_lite(input_shape)
    ]
    
    results = []

    # Train
    for model in models_list:
        print(f"\n--- Training {model.name} (50 Epochs) ---")
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Training with progress bar
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)
        
        # Evaluate Overall
        pred_prob = model.predict(X_test, verbose=0)
        pred_class = (pred_prob > 0.5).astype(int)
        
        acc = accuracy_score(y_test, pred_class)
        f1 = f1_score(y_test, pred_class)
        auc = roc_auc_score(y_test, pred_prob)
        
        results.append({
            "Model": model.name,
            "Family": "OVERALL",
            "Accuracy": round(acc, 4),
            "F1": round(f1, 4),
            "AUC": round(auc, 4)
        })
        
        # Evaluate Families
        for fam_name, indices in family_indices.items():
            fam_X = X[indices]
            fam_y = y[indices]
            if len(fam_X) == 0: continue

            pred_fam = (model.predict(fam_X, verbose=0) > 0.5).astype(int)
            fam_acc = accuracy_score(fam_y, pred_fam)
            
            results.append({
                "Model": model.name,
                "Family": fam_name.replace('_stego', ''),
                "Accuracy": round(fam_acc, 4),
                "F1": "-", 
                "AUC": "-"
            })

    # Print
    print("\n" + "="*60)
    print("FINAL 50-EPOCH RESULTS")
    print("="*60)
    df = pd.DataFrame(results)
    print(df.to_string())
    print("="*60)