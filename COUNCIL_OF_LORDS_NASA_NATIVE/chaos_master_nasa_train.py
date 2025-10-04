#!/usr/bin/env python3
"""
🌪️💀 CHAOS MASTER AI - NASA NATIVE VERSION 💀🌪️
Unstable system specialist trained on REAL NASA exoplanet catalog parameters
EDGE CASE HUNTER - FINDS WHAT OTHERS MISS

Features: [pl_orbper, pl_rade, st_teff, st_rad, st_mass, sy_dist, pl_orbeccen, pl_bmasse]
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU, GaussianNoise
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2, l1_l2
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

def chaos_master_nasa_loss(y_true, y_pred):
    """
    🌪️💀 CHAOS MASTER NASA LOSS FUNCTION 💀🌪️
    Unstable, adaptive loss for edge case detection in NASA data
    """
    # Base binary crossentropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # CHAOS PENALTIES - UNSTABLE AND ADAPTIVE
    # EXTREME penalty for missing edge cases
    fn_penalty = tf.where(tf.equal(y_true, 1), 
                         tf.pow(1 - y_pred, 3.0) * 6.0,  # Cubic penalty for FN
                         tf.zeros_like(y_pred))
    
    # CHAOTIC penalty for false edge cases
    fp_penalty = tf.where(tf.equal(y_true, 0), 
                         tf.pow(y_pred, 2.5) * 3.0,  # Non-linear FP penalty
                         tf.zeros_like(y_pred))
    
    # CHAOS UNCERTAINTY EMBRACE (different from others)
    confidence = tf.abs(y_pred - 0.5) * 2.0
    chaos_uncertainty = tf.reduce_mean(tf.sin(confidence * np.pi) * 0.2)  # Sinusoidal uncertainty
    
    # EDGE CASE DISCOVERY BONUS
    edge_bonus = tf.where(
        tf.logical_and(tf.equal(tf.round(y_pred), y_true),
                      tf.logical_or(y_pred < 0.2, y_pred > 0.8)),  # Reward extreme confidence
        -tf.square(confidence) * 0.25,
        tf.zeros_like(y_pred)
    )
    
    # CHAOS INSTABILITY FACTOR
    batch_var = tf.math.reduce_variance(y_pred)
    instability_factor = tf.sin(batch_var * 10) * 0.1
    
    # ADAPTIVE CHAOS PENALTY
    adaptive_chaos = tf.reduce_mean(tf.abs(y_pred - tf.roll(y_pred, 1, 0))) * 0.15
    
    total_loss = (bce + 
                 tf.reduce_mean(fn_penalty) + 
                 tf.reduce_mean(fp_penalty) + 
                 chaos_uncertainty + 
                 tf.reduce_mean(edge_bonus) +
                 instability_factor +
                 adaptive_chaos)
    
    return total_loss

def chaos_master_nasa_metric(y_true, y_pred):
    """NASA chaos detection metric with edge case focus"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred_binary = tf.cast(y_pred > 0.42, tf.float32)  # Chaotic threshold
    
    # Calculate chaotic recall (must catch edge cases)
    tp = tf.reduce_sum(y_true * y_pred_binary)
    fn = tf.reduce_sum(y_true * (1 - y_pred_binary))
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    
    # Calculate precision with chaos tolerance
    fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    
    # Chaos F1 with edge case emphasis
    chaos_weight = tf.reduce_mean(tf.abs(y_pred - 0.5)) + 0.1  # Weight by uncertainty
    f1_chaos = (2 * precision * recall * chaos_weight) / (precision + recall + chaos_weight + tf.keras.backend.epsilon())
    
    return f1_chaos

def create_chaos_master_nasa_architecture():
    """
    🌪️💀 Build NASA-native CHAOS MASTER architecture
    Unstable, adaptive network for edge case detection
    """
    input_layer = Input(shape=(8,), name='nasa_chaos_input')
    
    # CHAOS INJECTION LAYER
    x = GaussianNoise(0.05, name='chaos_noise')(input_layer)
    
    # UNSTABLE CHAOS LAYERS
    x = Dense(400, kernel_regularizer=l1_l2(l1=0.0002, l2=0.0008), name='chaos_layer_1')(x)
    x = LeakyReLU(alpha=0.3, name='chaos_activation_1')(x)  # More aggressive leaky
    x = BatchNormalization(name='chaos_norm_1')(x)
    x = Dropout(0.4, name='chaos_dropout_1')(x)  # High dropout for chaos
    
    # EDGE CASE DETECTION LAYERS
    x = Dense(200, kernel_regularizer=l1_l2(l1=0.0002, l2=0.0008), name='edge_detection_1')(x)
    x = LeakyReLU(alpha=0.2, name='chaos_activation_2')(x)
    x = BatchNormalization(name='chaos_norm_2')(x)
    x = Dropout(0.35, name='edge_dropout_1')(x)
    
    # INSTABILITY LAYERS
    x = Dense(100, kernel_regularizer=l1_l2(l1=0.0002, l2=0.0008), name='instability_1')(x)
    x = LeakyReLU(alpha=0.25, name='chaos_activation_3')(x)
    x = BatchNormalization(name='chaos_norm_3')(x)
    x = Dropout(0.3, name='instability_dropout')(x)
    
    # CHAOS CONVERGENCE LAYERS
    x = Dense(50, kernel_regularizer=l2(0.0008), name='chaos_convergence_1')(x)
    x = LeakyReLU(alpha=0.15, name='chaos_activation_4')(x)
    x = BatchNormalization(name='chaos_norm_4')(x)
    x = Dropout(0.25, name='convergence_dropout')(x)
    
    x = Dense(25, kernel_regularizer=l2(0.0008), name='chaos_convergence_2')(x)
    x = LeakyReLU(alpha=0.1, name='chaos_activation_5')(x)
    
    # CHAOS MASTER FINAL VERDICT
    output = Dense(1, activation='sigmoid', name='chaos_master_verdict')(x)
    
    model = Model(inputs=input_layer, outputs=output, name='CHAOS_MASTER_NASA_AI')
    
    # Chaotic optimizer (Nadam for adaptive momentum)
    optimizer = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    
    model.compile(
        optimizer=optimizer,
        loss=chaos_master_nasa_loss,
        metrics=[chaos_master_nasa_metric, 'accuracy', 'precision', 'recall']
    )
    
    return model

def train_chaos_master_nasa():
    """Train the NASA-native CHAOS MASTER AI"""
    
    print("🌪️💀" + "="*80 + "💀🌪️")
    print("💀 CHAOS MASTER AI - NASA NATIVE VERSION")
    print("💀 UNSTABLE SYSTEM SPECIALIST FOR NASA DATA")
    print("💀 MISSION: HUNT EDGE CASES AND ANOMALIES")
    print("🌪️💀" + "="*80 + "💀🌪️")
    
    # Generate NASA catalog training data with chaos emphasis
    generator = NASACatalogDataGenerator()
    X, y = generator.generate_training_data(220000, positive_fraction=0.68)  # 68% confirmed - chaos loves imbalance
    
    print(f"💀📊 CHAOS MASTER NASA DATASET:")
    print(f"   Total chaos samples: {len(X)}")
    print(f"   Edge case targets: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    print(f"   Chaotic negatives: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")
    
    # Split with stratification
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Chaotic standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print("💀🌪️ SUMMONING CHAOS MASTER ARCHITECTURE...")
    model = create_chaos_master_nasa_architecture()
    print(f"💀⚙️ Chaos Arsenal: {model.count_params():,} parameters")
    
    # Chaotic training callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=7, min_lr=1e-7, verbose=1),
        ModelCheckpoint('CHAOS_MASTER_NASA_checkpoint.h5', 
                       monitor='val_chaos_master_nasa_metric', 
                       save_best_only=True, verbose=1, mode='max')
    ]
    
    print("💀🌪️ CHAOS MASTER UNLEASHING INSTABILITY...")
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=80,
        batch_size=64,  # Smaller batches for more chaos
        callbacks=callbacks,
        verbose=1,
        class_weight={0: 1.0, 1: 1.4}  # Chaotic bias toward edge cases
    )
    
    # CHAOS EVALUATION
    print("\\n💀📊 CHAOS MASTER NASA INSTABILITY REPORT:")
    
    val_loss, val_metric, val_acc, val_prec, val_rec = model.evaluate(X_val_scaled, y_val, verbose=0)
    y_pred = model.predict(X_val_scaled, verbose=0)
    y_pred_binary = (y_pred > 0.42).astype(int).flatten()  # Chaos threshold
    
    accuracy = accuracy_score(y_val, y_pred_binary)
    precision = precision_score(y_val, y_pred_binary)
    recall = recall_score(y_val, y_pred_binary)
    f1 = f1_score(y_val, y_pred_binary)
    
    # Calculate chaos metrics
    prediction_variance = np.var(y_pred)
    edge_case_confidence = np.mean(np.abs(y_pred - 0.5))
    
    print(f"🌪️ CHAOS MASTER PERFORMANCE:")
    print(f"   Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   Precision: {precision:.3f} ({precision*100:.1f}%)")
    print(f"   Recall:    {recall:.3f} ({recall*100:.1f}%)")
    print(f"   F1 Score:  {f1:.3f}")
    print(f"   Chaos Variance: {prediction_variance:.4f}")
    print(f"   Edge Confidence: {edge_case_confidence:.3f}")
    
    # Save the chaos master and scaler
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    model_path = f"CHAOS_MASTER_NASA_{timestamp}.h5"
    scaler_path = f"CHAOS_MASTER_NASA_SCALER_{timestamp}.pkl"
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"💾 Saved NASA-native Chaos Master: {model_path}")
    print(f"💾 Saved scaler: {scaler_path}")
    
    # Chaos mastery check
    if accuracy >= 0.85 and recall >= 0.90 and prediction_variance > 0.02:
        print("🎉 CHAOS ACHIEVED! CHAOS MASTER READY FOR NASA EDGE CASES!")
        print("🏆 Instability and edge case detection mastered!")
    else:
        print("⚡ Chaos needs more instability. Recommendations:")
        print("   - Increase chaos injection (more noise)")
        print("   - Amplify instability factors")
        print("   - Embrace more edge case diversity")
    
    return model, scaler, history

if __name__ == "__main__":
    # Train the NASA-native Chaos Master
    model, scaler, history = train_chaos_master_nasa()
    
    print("\\n🌪️💀 CHAOS MASTER NASA TRAINING COMPLETE!")
    print("💀 Unstable genius ready to find NASA's hidden exoplanets!")
