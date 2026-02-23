
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

def create_dummy_models():
    print("Creating compatible placeholder models...")
    
    os.makedirs("models", exist_ok=True)

    # === CAMERA MODEL ===
    # Features: [perclos, blink_rate, blink_duration_mean]
    X_cam = np.random.rand(10, 3)
    y_cam = np.random.randint(0, 3, size=10)
    
    scaler_cam = StandardScaler()
    X_cam_scaled = scaler_cam.fit_transform(X_cam)
    
    clf_cam = RandomForestClassifier(n_estimators=10, random_state=42)
    clf_cam.fit(X_cam_scaled, y_cam)
    
    joblib.dump(clf_cam, 'models/model_camera_rf.pkl')
    joblib.dump(scaler_cam, 'models/model_camera_rf_scaler.pkl')
    print("✅ Saved models/model_camera_rf.pkl (and scaler)")

    # === CARLA MODEL ===
    # Features: [entropy, steering_rate, sdlp]
    X_carla = np.random.rand(10, 3)
    y_carla = np.random.randint(0, 3, size=10)
    
    scaler_carla = StandardScaler()
    X_carla_scaled = scaler_carla.fit_transform(X_carla)
    
    clf_carla = RandomForestClassifier(n_estimators=10, random_state=42)
    clf_carla.fit(X_carla_scaled, y_carla)
    
    joblib.dump(clf_carla, 'models/model_carla_steering_rf.pkl')
    joblib.dump(scaler_carla, 'models/model_carla_steering_rf_scaler.pkl')
    print("✅ Saved models/model_carla_steering_rf.pkl (and scaler)")

if __name__ == "__main__":
    create_dummy_models()
