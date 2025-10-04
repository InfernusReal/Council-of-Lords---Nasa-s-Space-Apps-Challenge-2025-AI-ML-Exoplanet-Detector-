"""
🛠️ QUICK FIX: COSMIC CONDUCTOR RETRAIN 🛠️
Retrain Cosmic Conductor with proper NASA catalog feature scaling
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib
from datetime import datetime
from nasa_catalog_data_generator import NASACatalogDataGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom functions
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

def create_fixed_cosmic_conductor():
    """Create new Cosmic Conductor with proper architecture"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(8,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation=harmonic_activation),  # Signature harmonic layer
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(16, activation='tanh'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=cosmic_conductor_nasa_loss,
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def main():
    """Retrain Cosmic Conductor with proper NASA features"""
    logger.info("🎼 FIXING COSMIC CONDUCTOR...")
    
    # Generate proper training data
    generator = NASACatalogDataGenerator()
    
    # Large training set for stability
    logger.info("📊 Generating training data...")
    X_train, y_train = generator.generate_training_data(5000, positive_fraction=0.5)
    X_val, y_val = generator.generate_training_data(1000, positive_fraction=0.5)
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Training labels: {np.bincount(y_train)}")
    
    # Create proper scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    logger.info(f"Scaled training stats:")
    logger.info(f"  Mean: {X_train_scaled.mean(axis=0)}")
    logger.info(f"  Std: {X_train_scaled.std(axis=0)}")
    logger.info(f"  Min: {X_train_scaled.min(axis=0)}")
    logger.info(f"  Max: {X_train_scaled.max(axis=0)}")
    
    # Create and train model
    logger.info("🏗️ Creating fixed model...")
    model = create_fixed_cosmic_conductor()
    
    logger.info("🚀 Training...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    train_loss, train_acc, train_prec, train_rec = model.evaluate(X_train_scaled, y_train, verbose=0)
    val_loss, val_acc, val_prec, val_rec = model.evaluate(X_val_scaled, y_val, verbose=0)
    
    logger.info(f"📊 Training Results:")
    logger.info(f"  Train: Acc={train_acc:.3f}, Prec={train_prec:.3f}, Rec={train_rec:.3f}")
    logger.info(f"  Val:   Acc={val_acc:.3f}, Prec={val_prec:.3f}, Rec={val_rec:.3f}")
    
    # Test on the problematic cases
    logger.info("🧪 Testing on problematic cases...")
    
    test_cases = [
        {'name': 'Kepler-442b', 'features': [112.31, 1.34, 4402, 0.54, 0.61, 370.46, 0.04, 2.3], 'expected': 'HIGH'},
        {'name': 'TRAPPIST-1e', 'features': [6.099, 0.92, 2559, 0.121, 0.089, 12.43, 0.005, 0.692], 'expected': 'HIGH'},
        {'name': 'Brown Dwarf', 'features': [100.0, 12.0, 5000, 1.2, 1.2, 150.0, 0.3, 30.0], 'expected': 'LOW'},
        {'name': 'Binary Eclipse', 'features': [2.1, 15.0, 5500, 1.0, 1.0, 200.0, 0.0, 500.0], 'expected': 'LOW'}
    ]
    
    for case in test_cases:
        features_scaled = scaler.transform([case['features']])
        pred = model.predict(features_scaled, verbose=0)[0][0]
        
        status = "✅" if (case['expected'] == 'HIGH' and pred > 0.5) or (case['expected'] == 'LOW' and pred < 0.5) else "❌"
        logger.info(f"  {status} {case['name']}: {pred:.3f} (expected {case['expected']})")
    
    # Save if performance is good
    if val_acc > 0.85 and val_prec > 0.8 and val_rec > 0.8:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        model_path = f"COSMIC_CONDUCTOR_NASA_FIXED_{timestamp}.h5"
        scaler_path = f"COSMIC_CONDUCTOR_NASA_FIXED_SCALER_{timestamp}.pkl"
        
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"✅ SAVED FIXED MODEL:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Scaler: {scaler_path}")
        
        logger.info("🎯 This should fix the 80.8% -> 90%+ accuracy issue!")
        
    else:
        logger.warning("⚠️ Model performance not good enough - not saving")
        logger.info("Try adjusting hyperparameters or training longer")

if __name__ == "__main__":
    main()