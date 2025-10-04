#!/usr/bin/env python3
"""
🔧⚡ FALSE POSITIVE REJECTION ENHANCER ⚡🔧
Enhanced Council logic to better handle false positives

The massive reality test revealed:
- 100% exoplanet detection (PERFECT!)
- 0% false positive rejection (NEEDS FIXING!)

Problem: Cosmic Conductor consistently votes "NOT_EXOPLANET" but gets overruled
Solution: Enhanced voting logic with false positive awareness
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import joblib
from supreme_telescope_converter import SupremeTelescopeConverter

def load_enhanced_council():
    """Load Council with enhanced false positive awareness"""
    print("🏛️ Loading Enhanced Council of Lords...")
    
    models = {}
    scalers = {}
    
    # Load all 5 specialist models
    specialists = [
        'CELESTIAL_ORACLE',
        'ATMOSPHERIC_WARRIOR', 
        'BACKYARD_GENIUS',
        'CHAOS_MASTER',
        'COSMIC_CONDUCTOR'
    ]
    
    for specialist in specialists:
        try:
            model_path = f"{specialist}_NASA_checkpoint.h5"
            scaler_path = f"{specialist}_NASA_scaler.pkl"
            
            if os.path.exists(model_path):
                print(f"INFO: Loading {specialist} from {model_path}")
                model = tf.keras.models.load_model(model_path)
                models[specialist] = model
                print(f"✅ Loaded {specialist}")
                
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    scalers[specialist] = scaler
                    print(f"  ✅ Loaded scaler for {specialist}")
                else:
                    print(f"  ⚠️ No scaler found for {specialist}")
            else:
                print(f"  ❌ Model not found: {model_path}")
                
        except Exception as e:
            print(f"  💥 Error loading {specialist}: {e}")
    
    print(f"🎯 Enhanced Council assembled: {len(models)} specialists ready!")
    return models, scalers

def enhanced_council_predict(models, scalers, nasa_params):
    """Enhanced prediction with better false positive handling"""
    
    # Prepare input
    features = np.array([
        nasa_params['koi_period'],
        nasa_params['koi_prad'], 
        nasa_params['koi_teq'],
        nasa_params['koi_insol'],
        nasa_params['koi_dor'],
        nasa_params['koi_depth'],
        nasa_params['koi_duration'],
        nasa_params['koi_ingress'],
        nasa_params['koi_impact'],
        nasa_params['ra'],
        nasa_params['dec'],
        nasa_params['koi_kepmag']
    ]).reshape(1, -1)
    
    print("  🗳️  Enhanced Council voting in session...")
    
    votes = {}
    predictions = {}
    
    # Get individual votes
    for name, model in models.items():
        try:
            if name in scalers:
                scaled_features = scalers[name].transform(features)
            else:
                scaled_features = features
            
            pred_prob = model.predict(scaled_features, verbose=0)[0][0]
            pred_class = "EXOPLANET" if pred_prob > 0.5 else "NOT_EXOPLANET"
            
            votes[name] = pred_class
            predictions[name] = pred_prob
            
            print(f"    {name}: {pred_class} (confidence: {pred_prob:.3f})")
            
        except Exception as e:
            print(f"    💥 {name} failed: {e}")
            votes[name] = "ABSTAIN"
            predictions[name] = 0.5
    
    # ENHANCED VOTING LOGIC
    exoplanet_votes = sum(1 for vote in votes.values() if vote == "EXOPLANET")
    not_exoplanet_votes = sum(1 for vote in votes.values() if vote == "NOT_EXOPLANET")
    
    # False positive protection features
    fp_score = nasa_params.get('false_positive_score', 0.5)
    suspicious_period = nasa_params['koi_period'] < 1.0 or nasa_params['koi_period'] > 50.0
    suspicious_radius = nasa_params['koi_prad'] > 15.0  # Very large "planet"
    suspicious_depth = nasa_params['koi_depth'] > 0.02  # Very deep "transit"
    
    # Count red flags
    red_flags = 0
    if fp_score > 0.7:
        red_flags += 2  # High FP score is major red flag
    if suspicious_period:
        red_flags += 1
    if suspicious_radius:
        red_flags += 1  
    if suspicious_depth:
        red_flags += 1
    
    print(f"🚩 Red flags detected: {red_flags}")
    print(f"   FP Score: {fp_score:.3f}")
    print(f"   Suspicious period: {suspicious_period}")
    print(f"   Suspicious radius: {suspicious_radius}")
    print(f"   Suspicious depth: {suspicious_depth}")
    
    # ENHANCED DECISION LOGIC
    if red_flags >= 3:
        # Too many red flags - force NOT_EXOPLANET
        final_verdict = "NOT_EXOPLANET"
        confidence = 0.8
        print("🚨 TOO MANY RED FLAGS - Forcing NOT_EXOPLANET")
        
    elif red_flags >= 2 and not_exoplanet_votes >= 1:
        # Some red flags + at least one dissenting vote
        final_verdict = "NOT_EXOPLANET"
        confidence = 0.7
        print("⚠️ Red flags + dissenting vote - Leaning NOT_EXOPLANET")
        
    elif exoplanet_votes >= 4:
        # Strong consensus for exoplanet
        avg_confidence = np.mean([pred for pred in predictions.values() if pred > 0.5])
        final_verdict = "EXOPLANET"
        confidence = avg_confidence
        print("✨ Strong consensus - EXOPLANET")
        
    elif exoplanet_votes >= 3 and red_flags <= 1:
        # Majority vote with low red flags
        avg_confidence = np.mean([pred for pred in predictions.values() if pred > 0.5])
        final_verdict = "EXOPLANET"
        confidence = avg_confidence * 0.9  # Slight penalty for not unanimous
        print("🤝 Majority consensus - EXOPLANET")
        
    else:
        # Unclear case - be conservative
        final_verdict = "NOT_EXOPLANET"
        confidence = 0.6
        print("🤔 Unclear case - Conservative NOT_EXOPLANET")
    
    print(f"⚖️ ENHANCED COUNCIL VERDICT: {final_verdict} (confidence: {confidence:.3f})")
    
    return final_verdict, confidence, votes, predictions

def test_enhanced_council():
    """Test the enhanced council on some challenging cases"""
    print("🔧⚡ TESTING ENHANCED FALSE POSITIVE REJECTION ⚡🔧")
    print("=" * 70)
    
    # Load systems
    converter = SupremeTelescopeConverter()
    models, scalers = load_enhanced_council()
    
    # Test cases from massive reality test
    test_cases = [
        ("massive_reality_test/detached_binary.csv", "false_positive", "Detached Binary"),
        ("massive_reality_test/contact_binary.csv", "false_positive", "Contact Binary"),
        ("massive_reality_test/blended_binary.csv", "false_positive", "Blended Binary"),
        ("massive_reality_test/systematic_fp.csv", "false_positive", "Systematic Artifact"),
        ("massive_reality_test/stellar_activity_fp.csv", "false_positive", "Stellar Activity"),
        ("massive_reality_test/hot_jupiter_kepler.csv", "exoplanet", "Hot Jupiter"),
        ("massive_reality_test/super_earth_tess.csv", "exoplanet", "Super-Earth")
    ]
    
    total_correct = 0
    total_tests = 0
    
    for filepath, expected_type, name in test_cases:
        if not os.path.exists(filepath):
            print(f"⚠️ Skipping {name} - file not found")
            continue
            
        print(f"\n🎯 TESTING: {name}")
        print("-" * 50)
        
        # Load data
        df = pd.read_csv(filepath)
        time = df['time'].values
        flux = df['flux'].values
        
        print(f"🎯 Expected: {expected_type}")
        
        # Convert
        nasa_params = converter.convert_raw_to_nasa_catalog(time, flux, name)
        
        # Enhanced prediction
        verdict, confidence, votes, predictions = enhanced_council_predict(models, scalers, nasa_params)
        
        # Check correctness
        correct = False
        if expected_type == "exoplanet" and verdict == "EXOPLANET":
            correct = True
        elif expected_type == "false_positive" and verdict == "NOT_EXOPLANET":
            correct = True
        
        result_icon = "🏆" if correct else "💥"
        result_text = "CORRECT" if correct else "INCORRECT"
        print(f"{result_icon} Result: {result_text}")
        
        if correct:
            total_correct += 1
        total_tests += 1
    
    accuracy = total_correct / total_tests * 100 if total_tests > 0 else 0
    
    print(f"\n🏆 ENHANCED COUNCIL TEST RESULTS:")
    print(f"   Tests: {total_tests}")
    print(f"   Correct: {total_correct}")
    print(f"   Accuracy: {accuracy:.1f}%")
    
    return accuracy

if __name__ == "__main__":
    test_enhanced_council()