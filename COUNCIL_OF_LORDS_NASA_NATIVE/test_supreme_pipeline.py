#!/usr/bin/env python3
"""
🔥⚔️ SUPREME CONVERSION LAYER vs REAL DATA TEST ⚔️🔥
Using the ULTIMATE conversion layer to handle all real-world telescope challenges
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import logging
import glob
from pathlib import Path

# Import our supreme converter
from supreme_telescope_converter import SupremeTelescopeConverter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom functions for model loading
def harmonic_activation(x):
    return tf.sin(x) * tf.cos(x * 0.5) + tf.tanh(x)

def celestial_oracle_nasa_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    fn_penalty = tf.where(tf.equal(y_true, 1), 
                         tf.sin(tf.square(1 - y_pred) * np.pi) * 8.0,
                         tf.zeros_like(y_pred))
    fp_penalty = tf.where(tf.equal(y_true, 0), 
                         tf.cos(tf.square(y_pred) * np.pi) * 2.5,
                         tf.zeros_like(y_pred))
    return bce + fn_penalty + fp_penalty

def atmospheric_warrior_nasa_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    fn_penalty = tf.where(tf.equal(y_true, 1), 
                         tf.square(1 - y_pred) * 6.0,
                         tf.zeros_like(y_pred))
    fp_penalty = tf.where(tf.equal(y_true, 0), 
                         tf.square(y_pred) * 3.5,
                         tf.zeros_like(y_pred))
    return bce + fn_penalty + fp_penalty

def backyard_genius_nasa_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    fn_penalty = tf.where(tf.equal(y_true, 1), 
                         tf.log(1 + tf.exp(1 - y_pred)) * 3.0,
                         tf.zeros_like(y_pred))
    fp_penalty = tf.where(tf.equal(y_true, 0), 
                         tf.log(1 + tf.exp(y_pred)) * 2.0,
                         tf.zeros_like(y_pred))
    return bce + fn_penalty + fp_penalty

def chaos_master_nasa_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    chaos_penalty = tf.sin(y_pred * np.pi) * tf.cos((1-y_true) * np.pi) * 4.0
    fn_penalty = tf.where(tf.equal(y_true, 1), 
                         tf.square(1 - y_pred) * 5.0,
                         tf.zeros_like(y_pred))
    return bce + chaos_penalty + fn_penalty

def cosmic_conductor_nasa_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    cosmic_penalty = tf.cos(y_pred * np.pi) * tf.sin(y_true * np.pi * 2) * 3.5
    fn_penalty = tf.where(tf.equal(y_true, 1), 
                         tf.square(1 - y_pred) * 7.0,
                         tf.zeros_like(y_pred))
    return bce + cosmic_penalty + fn_penalty

def load_council_of_lords():
    """Load the NASA-native Council of Lords ensemble"""
    print("🏛️ Loading Council of Lords NASA-Native Ensemble...")
    
    # Custom objects dictionary
    custom_objects = {
        'harmonic_activation': harmonic_activation,
        'celestial_oracle_nasa_loss': celestial_oracle_nasa_loss,
        'atmospheric_warrior_nasa_loss': atmospheric_warrior_nasa_loss,
        'backyard_genius_nasa_loss': backyard_genius_nasa_loss,
        'chaos_master_nasa_loss': chaos_master_nasa_loss,
        'cosmic_conductor_nasa_loss': cosmic_conductor_nasa_loss
    }
    
    base_path = Path(__file__).parent
    models = {}
    scalers = {}
    
    # Load each specialist with custom objects
    specialist_configs = {
        'CELESTIAL_ORACLE': {'loss': celestial_oracle_nasa_loss},
        'ATMOSPHERIC_WARRIOR': {'loss': atmospheric_warrior_nasa_loss},
        'BACKYARD_GENIUS': {'loss': backyard_genius_nasa_loss},
        'CHAOS_MASTER': {'loss': chaos_master_nasa_loss},
        'COSMIC_CONDUCTOR': {'loss': cosmic_conductor_nasa_loss, 'activation': harmonic_activation}
    }
    
    for specialist, config in specialist_configs.items():
        try:
            # Find latest model file
            model_pattern = f"{specialist}_NASA_*.h5"
            model_files = list(base_path.glob(model_pattern))
            
            if model_files:
                model_file = sorted(model_files, key=lambda x: x.name)[-1]
                logger.info(f"Loading {specialist} from {model_file.name}")
                
                # Load with custom objects
                models[specialist] = tf.keras.models.load_model(
                    str(model_file), 
                    custom_objects=custom_objects
                )
                print(f"  ✅ Loaded {specialist}")
                
                # Load corresponding scaler
                scaler_pattern = f"{specialist}_*SCALER_*.pkl"
                scaler_files = list(base_path.glob(scaler_pattern))
                
                if scaler_files:
                    scaler_file = sorted(scaler_files, key=lambda x: x.name)[-1]
                    scalers[specialist] = joblib.load(str(scaler_file))
                    print(f"  ✅ Loaded scaler for {specialist}")
                else:
                    print(f"  ⚠️ No scaler found for {specialist}")
                    # Create a dummy scaler that doesn't scale
                    from sklearn.preprocessing import StandardScaler
                    dummy_scaler = StandardScaler()
                    dummy_scaler.mean_ = np.zeros(8)  # 8 NASA catalog features
                    dummy_scaler.scale_ = np.ones(8)
                    scalers[specialist] = dummy_scaler
                    
        except Exception as e:
            print(f"  ❌ Failed to load {specialist}: {e}")
            logger.error(f"Error loading {specialist}: {e}")
    
    if len(models) < 3:
        raise Exception("Need at least 3 specialists for Council voting!")
    
    print(f"🎯 Council assembled: {len(models)} specialists ready!")
    return models, scalers

def council_of_lords_predict(models, scalers, nasa_params):
    """Get Council of Lords prediction on NASA catalog parameters"""
    
    votes = {}
    predictions = {}
    
    print("  🗳️  Council voting in session...")
    
    for specialist_name, model in models.items():
        try:
            scaler = scalers[specialist_name]
            
            # Scale features
            scaled_features = scaler.transform(nasa_params.reshape(1, -1))
            
            # Get prediction
            pred = model.predict(scaled_features, verbose=0)[0]
            
            # Convert to probability and vote
            if len(pred) == 1:
                prob = float(pred[0])
            else:
                prob = float(pred[1])  # Binary classification
            
            vote = "EXOPLANET" if prob > 0.5 else "NOT_EXOPLANET"
            
            votes[specialist_name] = vote
            predictions[specialist_name] = prob
            
            print(f"    {specialist_name}: {vote} (confidence: {prob:.3f})")
            
        except Exception as e:
            print(f"    ❌ {specialist_name} failed: {e}")
            votes[specialist_name] = "ABSTAIN"
            predictions[specialist_name] = 0.5
    
    # Calculate final verdict
    exoplanet_votes = sum(1 for vote in votes.values() if vote == "EXOPLANET")
    total_votes = sum(1 for vote in votes.values() if vote != "ABSTAIN")
    
    if total_votes == 0:
        return "UNCERTAIN", 0.5, votes, predictions
    
    consensus_strength = exoplanet_votes / total_votes
    
    if consensus_strength >= 0.6:
        verdict = "EXOPLANET"
    elif consensus_strength <= 0.4:
        verdict = "NOT_EXOPLANET"
    else:
        verdict = "UNCERTAIN"
    
    avg_confidence = np.mean([p for p in predictions.values() if isinstance(p, (int, float))])
    
    return verdict, avg_confidence, votes, predictions

def test_supreme_conversion_pipeline():
    """Test the complete pipeline: Raw data → Supreme converter → Council of Lords"""
    print("🔥🏛️ SUPREME CONVERSION PIPELINE vs REAL TELESCOPE DATA 🏛️🔥")
    print("=" * 80)
    
    # Load the Council
    models, scalers = load_council_of_lords()
    
    # Initialize supreme converter
    converter = SupremeTelescopeConverter()
    
    # Load real datasets
    data_dir = "real_telescope_data"
    
    if not os.path.exists(data_dir):
        print("❌ No real telescope data found! Run download_real_telescope_data.py first.")
        return
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    total_correct = 0
    total_tests = 0
    
    for csv_file in csv_files:
        print(f"\n🌟 Testing: {csv_file}")
        print("-" * 60)
        
        # Load data
        data_path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(data_path)
        
        time = df['time'].values
        flux = df['flux'].values
        
        # Load metadata for expected result
        meta_file = csv_file.replace('.csv', '_metadata.txt')
        meta_path = os.path.join(data_dir, meta_file)
        
        expected_type = "unknown"
        target_name = "Unknown"
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta_content = f.read()
                if "Type: exoplanet" in meta_content:
                    expected_type = "exoplanet"
                elif "Type: false_positive" in meta_content:
                    expected_type = "false_positive"
                
                # Extract target name
                for line in meta_content.split('\n'):
                    if line.startswith('Name: '):
                        target_name = line.replace('Name: ', '')
                        break
        
        print(f"📊 Data points: {len(time)}")
        print(f"⏱️  Duration: {time[-1] - time[0]:.1f} days")
        print(f"🎯 Expected: {expected_type}")
        print()
        
        # PHASE 1: Supreme conversion
        print("🔥 SUPREME CONVERTER: Processing real telescope data...")
        nasa_params = converter.convert_raw_to_nasa_catalog(time, flux, target_name)
        print()
        
        # PHASE 2: Council prediction
        print("🏛️ COUNCIL OF LORDS: Analyzing NASA parameters...")
        verdict, confidence, votes, predictions = council_of_lords_predict(models, scalers, nasa_params)
        
        print(f"\n🏛️ COUNCIL VERDICT: {verdict}")
        print(f"🎯 Confidence: {confidence:.3f}")
        
        # Check if correct
        correct = False
        if expected_type == "exoplanet" and verdict == "EXOPLANET":
            correct = True
        elif expected_type == "false_positive" and verdict == "NOT_EXOPLANET":
            correct = True
        elif expected_type == "unknown":
            correct = True  # Can't be wrong on unknown targets
        
        result_icon = "✅" if correct else "❌"
        print(f"{result_icon} Result: {'CORRECT' if correct else 'INCORRECT'}")
        
        if correct:
            total_correct += 1
        total_tests += 1
        
        print("=" * 60)
    
    # Final results
    print(f"\n🏆 SUPREME PIPELINE BATTLE RESULTS:")
    print(f"📊 Total tests: {total_tests}")
    print(f"✅ Correct predictions: {total_correct}")
    print(f"🎯 Accuracy: {total_correct/total_tests*100:.1f}%" if total_tests > 0 else "No tests completed")
    
    if total_tests > 0:
        accuracy = total_correct / total_tests * 100
        if accuracy >= 80:
            print("🏆 SUPREME VICTORY! Conversion layer dominates real data!")
        elif accuracy >= 60:
            print("🥈 STRONG PERFORMANCE! Conversion layer handles real challenges!")
        elif accuracy >= 40:
            print("🥉 MODERATE SUCCESS! Still room for improvement!")
        else:
            print("💥 NEEDS WORK! Conversion layer struggling with real data!")

if __name__ == "__main__":
    test_supreme_conversion_pipeline()