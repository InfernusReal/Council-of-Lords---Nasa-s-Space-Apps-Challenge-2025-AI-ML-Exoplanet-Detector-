"""
🔍 COSMIC CONDUCTOR DIAGNOSTIC 🔍
Analyze why Cosmic Conductor votes so low on real exoplanets
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import joblib
import glob

# Register custom functions
def harmonic_activation(x):
    return tf.sin(x) * tf.cos(x * 0.5) + tf.tanh(x)

def cosmic_conductor_nasa_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    fn_penalty = tf.where(tf.equal(y_true, 1), 
                         tf.sin(tf.square(1 - y_pred) * np.pi) * 4.0,
                         tf.zeros_like(y_pred))
    fp_penalty = tf.where(tf.equal(y_true, 0), 
                         tf.cos(tf.square(y_pred) * np.pi) * 2.2,
                         tf.zeros_like(y_pred))
    return bce + fn_penalty + fp_penalty

get_custom_objects().update({
    'harmonic_activation': harmonic_activation,
    'cosmic_conductor_nasa_loss': cosmic_conductor_nasa_loss
})

# Load Cosmic Conductor
print("🎼 Loading Cosmic Conductor...")
model_files = glob.glob("COSMIC_CONDUCTOR_NASA_2025-09-11*.h5")
scaler_files = glob.glob("COSMIC_CONDUCTOR_NASA_SCALER_2025-09-11*.pkl")

if not model_files or not scaler_files:
    print("❌ Files not found!")
    exit()

model = load_model(model_files[0])
scaler = joblib.load(scaler_files[0])

print("✅ Loaded successfully!")
print(f"Model: {model_files[0]}")
print(f"Scaler: {scaler_files[0]}")

# Test cases
test_cases = [
    {
        'name': 'Kepler-442b (should be HIGH)',
        'features': [112.31, 1.34, 4402, 0.54, 0.61, 370.46, 0.04, 2.3],
        'expected': 'HIGH'
    },
    {
        'name': 'TRAPPIST-1e (should be HIGH)', 
        'features': [6.099, 0.92, 2559, 0.121, 0.089, 12.43, 0.005, 0.692],
        'expected': 'HIGH'
    },
    {
        'name': 'Brown Dwarf (should be LOW)',
        'features': [100.0, 12.0, 5000, 1.2, 1.2, 150.0, 0.3, 30.0],
        'expected': 'LOW'
    },
    {
        'name': 'Binary Eclipse (should be LOW)',
        'features': [2.1, 15.0, 5500, 1.0, 1.0, 200.0, 0.0, 500.0],
        'expected': 'LOW'
    }
]

print("\n🧪 COSMIC CONDUCTOR DIAGNOSTIC:")
print("=" * 60)

for case in test_cases:
    features = np.array(case['features'])
    
    # Raw features
    print(f"\n🔍 {case['name']} (Expected: {case['expected']})")
    print(f"Raw features: {features}")
    
    # Scaled features  
    scaled = scaler.transform([features])
    print(f"Scaled features: {scaled[0]}")
    print(f"Scaling stats: min={scaled[0].min():.3f}, max={scaled[0].max():.3f}")
    
    # Prediction
    pred = model.predict(scaled, verbose=0)[0][0]
    print(f"Prediction: {pred:.6f}")
    
    # Analysis
    if case['expected'] == 'HIGH' and pred < 0.5:
        print("❌ PROBLEM: Should be HIGH but voted LOW!")
    elif case['expected'] == 'LOW' and pred > 0.5:
        print("❌ PROBLEM: Should be LOW but voted HIGH!")
    else:
        print("✅ Correct behavior")

# Feature analysis
print("\n📊 SCALING ANALYSIS:")
print("=" * 60)

# Check if scaling is extreme
real_exoplanet = [112.31, 1.34, 4402, 0.54, 0.61, 370.46, 0.04, 2.3]
scaled_real = scaler.transform([real_exoplanet])[0]

print(f"Real exoplanet scaled features:")
for i, (raw, scaled) in enumerate(zip(real_exoplanet, scaled_real)):
    print(f"  Feature {i}: {raw:8.3f} -> {scaled:8.3f}")

# Check scaler stats
print(f"\nScaler mean: {scaler.mean_}")
print(f"Scaler std:  {scaler.scale_}")

# Look for extreme scaling that might cause issues
extreme_indices = []
for i, scale in enumerate(scaler.scale_):
    if scale > 10 or scale < 0.001:
        extreme_indices.append(i)

if extreme_indices:
    print(f"\n⚠️ EXTREME SCALING detected in features: {extreme_indices}")
    print("This could cause the model to behave poorly!")
else:
    print("\n✅ Scaling looks reasonable")

print("\n🔧 RECOMMENDATIONS:")
if any(pred < 0.5 for case in test_cases[:2] if case['expected'] == 'HIGH'):
    print("1. Cosmic Conductor is TOO CONSERVATIVE on real exoplanets")
    print("2. Consider retraining with different loss weights")
    print("3. Or adjust the harmonic activation function")
    print("4. Check if training data had realistic NASA features")
else:
    print("Cosmic Conductor behavior looks correct!")