#!/usr/bin/env python3
"""
🎼🌌 COSMIC CONDUCTOR AI - NASA NATIVE VERSION 🌌🎼
Harmonic analysis specialist trained on REAL NASA exoplanet catalog parameters
ORCHESTRATOR OF COSMIC RHYTHMS AND ORBITAL HARMONIES

Features: [pl_orbper, pl_rade, st_teff, st_rad, st_mass, sy_dist, pl_orbeccen, pl_bmasse]
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Activation, Lambda
from tensorflow.keras.optimizers import Adam, Adamax
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

def cosmic_conductor_nasa_loss(y_true, y_pred):
    """
    🎼🌌 COSMIC CONDUCTOR NASA LOSS FUNCTION 🌌🎼
    Harmonic loss function for orbital resonance detection in NASA data
    """
    # Base binary crossentropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # HARMONIC PENALTIES
    # SYMPHONIC penalty for missing orbital harmonies
    fn_penalty = tf.where(tf.equal(y_true, 1), 
                         tf.sin(tf.square(1 - y_pred) * np.pi) * 4.0,  # Sinusoidal FN penalty
                         tf.zeros_like(y_pred))
    
    # DISSONANCE penalty for false harmonies
    fp_penalty = tf.where(tf.equal(y_true, 0), 
                         tf.cos(tf.square(y_pred) * np.pi) * 2.2,  # Cosine FP penalty
                         tf.zeros_like(y_pred))
    
    # HARMONIC RESONANCE requirement
    confidence = tf.abs(y_pred - 0.5) * 2.0
    resonance_factor = tf.reduce_mean(tf.sin(confidence * np.pi * 2) * 0.2)
    
    # COSMIC SYMPHONY BONUS
    symphony_bonus = tf.where(
        tf.equal(tf.round(y_pred), y_true),
        -tf.sin(confidence * np.pi) * 0.18,  # Harmonic bonus
        tf.zeros_like(y_pred)
    )
    
    # ORBITAL RHYTHM CONSISTENCY
    rhythm_consistency = tf.reduce_mean(tf.cos(y_pred * np.pi * 4)) * 0.1
    
    # CONDUCTOR'S BATON (gradient harmony)
    baton_guidance = tf.reduce_mean(tf.sin((y_pred - tf.reduce_mean(y_pred)) * np.pi * 2)) * 0.05
    
    total_loss = (bce + 
                 tf.reduce_mean(fn_penalty) + 
                 tf.reduce_mean(fp_penalty) + 
                 resonance_factor + 
                 tf.reduce_mean(symphony_bonus) +
                 rhythm_consistency +
                 baton_guidance)
    
    return total_loss

def cosmic_conductor_nasa_metric(y_true, y_pred):
    """NASA harmonic detection metric with orbital resonance focus"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred_binary = tf.cast(y_pred > 0.52, tf.float32)  # Harmonic threshold
    
    # Calculate harmonic recall
    tp = tf.reduce_sum(y_true * y_pred_binary)
    fn = tf.reduce_sum(y_true * (1 - y_pred_binary))
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    
    # Calculate precision with harmonic weighting
    fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    
    # Harmonic F1 with symphony weighting
    harmonic_weight = tf.reduce_mean(tf.sin(y_pred * np.pi)) + 0.5
    f1_harmonic = (2 * precision * recall * harmonic_weight) / (precision + recall + harmonic_weight + tf.keras.backend.epsilon())
    
    return f1_harmonic

def harmonic_activation(x):
    """Custom harmonic activation function"""
    return tf.nn.sigmoid(x) * (1 + 0.1 * tf.sin(x * np.pi))

def create_cosmic_conductor_nasa_architecture():
    """
    🎼🌌 Build NASA-native COSMIC CONDUCTOR architecture
    Harmonic resonance network for orbital pattern detection
    """
    input_layer = Input(shape=(8,), name='nasa_harmonic_input')
    
    # ORCHESTRAL OPENING
    x = Dense(280, kernel_regularizer=l2(0.0004), name='orchestral_opening')(input_layer)
    x = Lambda(harmonic_activation, name='harmonic_activation_1')(x)
    x = BatchNormalization(name='harmonic_norm_1')(x)
    x = Dropout(0.22, name='orchestral_dropout_1')(x)
    
    # FIRST MOVEMENT - STELLAR HARMONICS
    x = Dense(140, kernel_regularizer=l2(0.0004), name='stellar_harmonics')(x)
    x = Lambda(harmonic_activation, name='harmonic_activation_2')(x)
    x = BatchNormalization(name='harmonic_norm_2')(x)
    x = Dropout(0.2, name='stellar_dropout')(x)
    
    # SECOND MOVEMENT - ORBITAL RESONANCE
    x = Dense(70, kernel_regularizer=l2(0.0004), name='orbital_resonance')(x)
    x = Lambda(harmonic_activation, name='harmonic_activation_3')(x)
    x = BatchNormalization(name='harmonic_norm_3')(x)
    x = Dropout(0.18, name='orbital_dropout')(x)
    
    # THIRD MOVEMENT - COSMIC SYMPHONY
    x = Dense(35, kernel_regularizer=l2(0.0004), name='cosmic_symphony_1')(x)
    x = Lambda(harmonic_activation, name='harmonic_activation_4')(x)
    x = BatchNormalization(name='harmonic_norm_4')(x)
    x = Dropout(0.15, name='symphony_dropout')(x)
    
    x = Dense(18, kernel_regularizer=l2(0.0004), name='cosmic_symphony_2')(x)
    x = Lambda(harmonic_activation, name='harmonic_activation_5')(x)
    
    # FINALE - CONDUCTOR'S VERDICT
    output = Dense(1, activation='sigmoid', name='conductor_finale')(x)
    
    model = Model(inputs=input_layer, outputs=output, name='COSMIC_CONDUCTOR_NASA_AI')
    
    # Harmonic optimizer (Adamax for adaptive learning)
    optimizer = Adamax(learning_rate=0.0012, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    
    model.compile(
        optimizer=optimizer,
        loss=cosmic_conductor_nasa_loss,
        metrics=[cosmic_conductor_nasa_metric, 'accuracy', 'precision', 'recall']
    )
    
    return model

def train_cosmic_conductor_nasa():
    """Train the NASA-native COSMIC CONDUCTOR AI"""
    
    print("🎼🌌" + "="*80 + "🌌🎼")
    print("🎼 COSMIC CONDUCTOR AI - NASA NATIVE VERSION")
    print("🎼 HARMONIC ANALYSIS SPECIALIST FOR NASA DATA")
    print("🎼 MISSION: ORCHESTRATE EXOPLANET DISCOVERY SYMPHONY")
    print("🎼🌌" + "="*80 + "🌌🎼")
    
    # Generate NASA catalog training data
    generator = NASACatalogDataGenerator()
    X, y = generator.generate_training_data(190000, positive_fraction=0.60)  # 60% confirmed - harmonic balance
    
    print(f"🎼📊 COSMIC CONDUCTOR NASA DATASET:")
    print(f"   Total symphonic samples: {len(X)}")
    print(f"   Harmonic targets: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    print(f"   Dissonant negatives: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")
    
    # Split with stratification
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Harmonic standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print("🎼🏗️ ASSEMBLING COSMIC ORCHESTRA...")
    model = create_cosmic_conductor_nasa_architecture()
    print(f"🎼⚙️ Orchestra Size: {model.count_params():,} parameters")
    
    # Symphonic training callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=22, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=11, min_lr=1e-7, verbose=1),
        ModelCheckpoint('COSMIC_CONDUCTOR_NASA_checkpoint.h5', 
                       monitor='val_cosmic_conductor_nasa_metric', 
                       save_best_only=True, verbose=1, mode='max')
    ]
    
    print("🎼🌌 COSMIC CONDUCTOR BEGINNING SYMPHONY...")
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=256,
        callbacks=callbacks,
        verbose=1,
        class_weight={0: 1.0, 1: 1.15}  # Slight harmonic bias
    )
    
    # SYMPHONIC EVALUATION
    print("\\n🎼📊 COSMIC CONDUCTOR NASA SYMPHONY PERFORMANCE:")
    
    val_loss, val_metric, val_acc, val_prec, val_rec = model.evaluate(X_val_scaled, y_val, verbose=0)
    y_pred = model.predict(X_val_scaled, verbose=0)
    y_pred_binary = (y_pred > 0.52).astype(int).flatten()  # Harmonic threshold
    
    accuracy = accuracy_score(y_val, y_pred_binary)
    precision = precision_score(y_val, y_pred_binary)
    recall = recall_score(y_val, y_pred_binary)
    f1 = f1_score(y_val, y_pred_binary)
    
    # Calculate harmonic metrics
    harmonic_variance = np.var(np.sin(y_pred * np.pi))
    resonance_strength = np.mean(np.abs(np.sin(y_pred * np.pi * 2)))
    
    print(f"🎼 SYMPHONIC PERFORMANCE:")
    print(f"   Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   Precision: {precision:.3f} ({precision*100:.1f}%)")
    print(f"   Recall:    {recall:.3f} ({recall*100:.1f}%)")
    print(f"   F1 Score:  {f1:.3f}")
    print(f"   Harmonic Variance: {harmonic_variance:.4f}")
    print(f"   Resonance Strength: {resonance_strength:.3f}")
    
    # Save the cosmic conductor and scaler
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    model_path = f"COSMIC_CONDUCTOR_NASA_{timestamp}.h5"
    scaler_path = f"COSMIC_CONDUCTOR_NASA_SCALER_{timestamp}.pkl"
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"💾 Saved NASA-native Cosmic Conductor: {model_path}")
    print(f"💾 Saved scaler: {scaler_path}")
    
    # Symphonic mastery check
    if accuracy >= 0.88 and precision >= 0.87 and recall >= 0.87 and resonance_strength > 0.3:
        print("🎉 SYMPHONY COMPLETE! COSMIC CONDUCTOR READY FOR NASA!")
        print("🏆 Harmonic analysis mastered for exoplanet detection!")
    else:
        print("🎵 Symphony needs fine-tuning. Recommendations:")
        print("   - Adjust harmonic frequencies")
        print("   - Increase orchestral complexity")
        print("   - Enhance resonance patterns")
    
    return model, scaler, history

if __name__ == "__main__":
    # Train the NASA-native Cosmic Conductor
    model, scaler, history = train_cosmic_conductor_nasa()
    
    print("\\n🎼🌌 COSMIC CONDUCTOR NASA TRAINING COMPLETE!")
    print("🎼 Harmonic analysis ready to conduct NASA's exoplanet symphony!")
