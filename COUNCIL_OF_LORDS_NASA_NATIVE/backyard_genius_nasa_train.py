#!/usr/bin/env python3
"""
🏡🧠 BACKYARD GENIUS AI - NASA NATIVE VERSION 🧠🏡
Amateur astronomer's intuition meets professional NASA data analysis
GRASSROOTS DETECTION POWERED BY NASA CATALOG PARAMETERS

Features: [pl_orbper, pl_rade, st_teff, st_rad, st_mass, sy_dist, pl_orbeccen, pl_bmasse]
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Activation
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import joblib
from datetime import datetime
import sys
import os

# Add the parent directory to Python path to import the data generator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nasa_catalog_data_generator import NASACatalogDataGenerator

warnings.filterwarnings('ignore')

def backyard_genius_nasa_loss(y_true, y_pred):
    """
    🏡🔭 BACKYARD GENIUS NASA LOSS FUNCTION 🔭🏡
    Amateur astronomer's intuitive approach to NASA catalog analysis
    """
    # Base binary crossentropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # AMATEUR ASTRONOMER APPROACH
    # Moderate penalty for missing discoveries (amateur enthusiasm)
    fn_penalty = tf.where(tf.equal(y_true, 1), 
                         tf.square(1 - y_pred) * 3.5,  # Enthusiastic but not aggressive
                         tf.zeros_like(y_pred))
    
    # Careful penalty for false detections (amateur caution)
    fp_penalty = tf.where(tf.equal(y_true, 0), 
                         tf.square(y_pred) * 1.8,  # Cautious approach
                         tf.zeros_like(y_pred))
    
    # INTUITIVE CONFIDENCE requirement
    confidence = tf.abs(y_pred - 0.5) * 2.0
    uncertainty_penalty = tf.reduce_mean(tf.square(1 - confidence)) * 0.25
    
    # AMATEUR DISCOVERY BONUS (reward finding hard cases)
    discovery_bonus = tf.where(
        tf.logical_and(tf.equal(tf.round(y_pred), y_true), 
                      tf.logical_and(y_pred > 0.3, y_pred < 0.7)),  # Reward medium confidence correct predictions
        -tf.abs(y_pred - 0.5) * 0.1,
        tf.zeros_like(y_pred)
    )
    
    # CONSISTENCY penalty (amateur consistency matters)
    consistency_penalty = tf.reduce_mean(tf.abs(y_pred - tf.reduce_mean(y_pred))) * 0.1
    
    total_loss = (bce + 
                 tf.reduce_mean(fn_penalty) + 
                 tf.reduce_mean(fp_penalty) + 
                 uncertainty_penalty + 
                 tf.reduce_mean(discovery_bonus) +
                 consistency_penalty)
    
    return total_loss

def backyard_genius_nasa_metric(y_true, y_pred):
    """NASA amateur detection metric with intuitive balance"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)  # Standard threshold for amateur
    
    # Calculate balanced amateur performance
    tp = tf.reduce_sum(y_true * y_pred_binary)
    fn = tf.reduce_sum(y_true * (1 - y_pred_binary))
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    
    # Calculate precision with amateur care
    fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    
    # Balanced F1 for amateur astronomer
    f1_amateur = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    
    return f1_amateur

def create_backyard_genius_nasa_architecture():
    """
    🏡🔭 Build NASA-native BACKYARD GENIUS architecture
    Amateur astronomer's approach to professional NASA data
    """
    input_layer = Input(shape=(8,), name='nasa_amateur_input')
    
    # AMATEUR OBSERVATION LAYERS
    x = Dense(200, activation='relu', kernel_regularizer=l2(0.0003), name='amateur_observation_1')(input_layer)
    x = BatchNormalization(name='amateur_norm_1')(x)
    x = Dropout(0.2, name='amateur_dropout_1')(x)
    
    # INTUITIVE PATTERN RECOGNITION
    x = Dense(100, activation='relu', kernel_regularizer=l2(0.0003), name='pattern_recognition')(x)
    x = BatchNormalization(name='amateur_norm_2')(x)
    x = Dropout(0.18, name='pattern_dropout')(x)
    
    # AMATEUR ANALYSIS LAYERS
    x = Dense(50, activation='relu', kernel_regularizer=l2(0.0003), name='amateur_analysis_1')(x)
    x = BatchNormalization(name='amateur_norm_3')(x)
    x = Dropout(0.15, name='analysis_dropout_1')(x)
    
    x = Dense(25, activation='relu', kernel_regularizer=l2(0.0003), name='amateur_analysis_2')(x)
    x = BatchNormalization(name='amateur_norm_4')(x)
    x = Dropout(0.12, name='analysis_dropout_2')(x)
    
    # BACKYARD INTUITION LAYER
    x = Dense(12, activation='relu', kernel_regularizer=l2(0.0003), name='backyard_intuition')(x)
    
    # AMATEUR DISCOVERY VERDICT
    output = Dense(1, activation='sigmoid', name='amateur_discovery_verdict')(x)
    
    model = Model(inputs=input_layer, outputs=output, name='BACKYARD_GENIUS_NASA_AI')
    
    # Amateur-friendly optimizer (stable and reliable)
    optimizer = RMSprop(learning_rate=0.0008, rho=0.9, epsilon=1e-8)
    
    model.compile(
        optimizer=optimizer,
        loss=backyard_genius_nasa_loss,
        metrics=[backyard_genius_nasa_metric, 'accuracy', 'precision', 'recall']
    )
    
    return model

def train_backyard_genius_nasa():
    """Train the NASA-native BACKYARD GENIUS AI"""
    
    print("🏡🧠" + "="*80 + "🧠🏡")
    print("🔭 BACKYARD GENIUS AI - NASA NATIVE VERSION")
    print("🔭 AMATEUR ASTRONOMER'S INTUITION FOR NASA DATA")
    print("🔭 MISSION: GRASSROOTS EXOPLANET DISCOVERY")
    print("🏡🧠" + "="*80 + "🧠🏡")
    
    # Generate NASA catalog training data
    generator = NASACatalogDataGenerator()
    X, y = generator.generate_training_data(160000, positive_fraction=0.58)  # 58% confirmed exoplanets
    
    print(f"🔭📊 BACKYARD GENIUS NASA DATASET:")
    print(f"   Total observations: {len(X)}")
    print(f"   Amateur discoveries: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    print(f"   False alarms: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")
    
    # Split with stratification
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Amateur-grade standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print("🔭🏗️ SETTING UP BACKYARD OBSERVATORY...")
    model = create_backyard_genius_nasa_architecture()
    print(f"🔭⚙️ Amateur Equipment: {model.count_params():,} parameters")
    
    # Amateur astronomer callbacks (patient and steady)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=12, min_lr=1e-7, verbose=1),
        ModelCheckpoint('BACKYARD_GENIUS_NASA_checkpoint.h5', 
                       monitor='val_backyard_genius_nasa_metric', 
                       save_best_only=True, verbose=1, mode='max')
    ]
    
    print("🔭🌌 BACKYARD GENIUS SCANNING THE NASA CATALOG...")
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=120,
        batch_size=128,
        callbacks=callbacks,
        verbose=1,
        class_weight={0: 1.0, 1: 1.1}  # Slight amateur enthusiasm
    )
    
    # AMATEUR DISCOVERY EVALUATION
    print("\\n🔭📊 BACKYARD GENIUS NASA DISCOVERY LOG:")
    
    val_loss, val_metric, val_acc, val_prec, val_rec = model.evaluate(X_val_scaled, y_val, verbose=0)
    y_pred = model.predict(X_val_scaled, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()  # Amateur threshold
    
    accuracy = accuracy_score(y_val, y_pred_binary)
    precision = precision_score(y_val, y_pred_binary)
    recall = recall_score(y_val, y_pred_binary)
    f1 = f1_score(y_val, y_pred_binary)
    
    print(f"🏆 AMATEUR DISCOVERY PERFORMANCE:")
    print(f"   Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   Precision: {precision:.3f} ({precision*100:.1f}%)")
    print(f"   Recall:    {recall:.3f} ({recall*100:.1f}%)")
    print(f"   F1 Score:  {f1:.3f}")
    print(f"   Amateur Balance: Equal precision/recall focus")
    
    # Save the backyard genius and scaler
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    model_path = f"BACKYARD_GENIUS_NASA_{timestamp}.h5"
    scaler_path = f"BACKYARD_GENIUS_NASA_SCALER_{timestamp}.pkl"
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"💾 Saved NASA-native Backyard Genius: {model_path}")
    print(f"💾 Saved scaler: {scaler_path}")
    
    # Amateur achievement check
    if accuracy >= 0.87 and precision >= 0.85 and recall >= 0.85:
        print("🎉 AMATEUR SUCCESS! BACKYARD GENIUS READY FOR NASA!")
        print("🏆 Grassroots astronomy meets professional standards!")
    else:
        print("🔧 Backyard equipment needs tuning. Suggestions:")
        print("   - Increase observation time (more epochs)")
        print("   - Upgrade amateur equipment (adjust architecture)")
        print("   - Join amateur astronomy club (ensemble methods)")
    
    return model, scaler, history

if __name__ == "__main__":
    # Train the NASA-native Backyard Genius
    model, scaler, history = train_backyard_genius_nasa()
    
    print("\\n🔭🏡 BACKYARD GENIUS NASA TRAINING COMPLETE!")
    print("🌟 Amateur intuition ready for professional NASA discoveries!")
