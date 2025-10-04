"""
🌌 REALISTIC OBSERVATIONAL DATA TEST 🌌
Test the ensemble on what users would ACTUALLY upload:
- Incomplete data
- Measurement errors  
- Different units
- Missing parameters
- Observational noise
- Real-world data challenges
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import joblib
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom functions (same as before)
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
    fn_penalty = tf.where(tf.equal(y_true, 1), 
                         tf.pow(1 - y_pred, 3) * 10.0,
                         tf.zeros_like(y_pred))
    fp_penalty = tf.where(tf.equal(y_true, 0), 
                         tf.pow(y_pred, 3) * 4.0,
                         tf.zeros_like(y_pred))
    return bce + fn_penalty + fp_penalty

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
    'celestial_oracle_nasa_loss': celestial_oracle_nasa_loss,
    'atmospheric_warrior_nasa_loss': atmospheric_warrior_nasa_loss,
    'backyard_genius_nasa_loss': backyard_genius_nasa_loss,
    'chaos_master_nasa_loss': chaos_master_nasa_loss,
    'cosmic_conductor_nasa_loss': cosmic_conductor_nasa_loss
})

class RealisticObservationalDataSimulator:
    """Simulate what users would ACTUALLY upload"""
    
    def __init__(self):
        # Real confirmed exoplanets with perfect NASA data
        self.perfect_exoplanets = [
            {'name': 'Kepler-442b', 'features': [112.31, 1.34, 4402, 0.54, 0.61, 370.46, 0.04, 2.3], 'type': 'exoplanet'},
            {'name': 'TRAPPIST-1e', 'features': [6.099, 0.92, 2559, 0.121, 0.089, 12.43, 0.005, 0.692], 'type': 'exoplanet'},
            {'name': 'TOI-715b', 'features': [19.3, 1.55, 3470, 0.476, 0.43, 42.0, 0.0, 3.02], 'type': 'exoplanet'},
            {'name': 'K2-18b', 'features': [32.94, 2.61, 3457, 0.411, 0.359, 34.0, 0.0, 8.63], 'type': 'exoplanet'},
            {'name': 'Proxima Centauri b', 'features': [11.186, 1.17, 3042, 0.154, 0.123, 1.295, 0.35, 1.27], 'type': 'exoplanet'}
        ]
        
        # Known false positives  
        self.perfect_false_positives = [
            {'name': 'Brown Dwarf', 'features': [100.0, 12.0, 5000, 1.2, 1.2, 150.0, 0.3, 30.0], 'type': 'false_positive'},
            {'name': 'Binary Eclipse', 'features': [2.1, 15.0, 5500, 1.0, 1.0, 200.0, 0.0, 500.0], 'type': 'false_positive'},
            {'name': 'Stellar Activity', 'features': [25.0, 1.5, 3200, 0.3, 0.3, 50.0, 0.8, 2.0], 'type': 'false_positive'}
        ]
    
    def add_measurement_errors(self, features, error_level='realistic'):
        """Add realistic measurement uncertainties"""
        noisy_features = features.copy()
        
        if error_level == 'realistic':
            # Typical observational uncertainties
            errors = [
                0.05,  # period: ±5%
                0.10,  # radius: ±10% 
                0.02,  # temperature: ±2%
                0.15,  # stellar mass: ±15%
                0.10,  # stellar radius: ±10%
                0.05,  # distance: ±5%
                0.20,  # eccentricity: ±20%
                0.25   # planet mass: ±25%
            ]
        elif error_level == 'poor':
            # Poor quality observations
            errors = [0.15, 0.25, 0.08, 0.30, 0.25, 0.15, 0.50, 0.50]
        else:  # 'excellent'
            errors = [0.01, 0.03, 0.005, 0.05, 0.03, 0.01, 0.05, 0.08]
        
        for i, (feature, error) in enumerate(zip(noisy_features, errors)):
            noise = np.random.normal(0, error * abs(feature))
            noisy_features[i] = max(0.001, feature + noise)  # Prevent negative values
            
        return noisy_features
    
    def introduce_missing_data(self, features, missing_rate=0.2):
        """Simulate missing/unknown parameters"""
        incomplete_features = features.copy()
        
        # Randomly set some features to NaN or estimated values
        for i in range(len(incomplete_features)):
            if random.random() < missing_rate:
                if random.random() < 0.5:
                    # Missing data - use population average
                    population_averages = [50.0, 2.0, 5000, 1.0, 1.0, 100.0, 0.1, 5.0]
                    incomplete_features[i] = population_averages[i]
                else:
                    # Very uncertain estimate
                    uncertainty_factor = random.uniform(0.5, 2.0)
                    incomplete_features[i] *= uncertainty_factor
        
        return incomplete_features
    
    def apply_unit_conversion_errors(self, features):
        """Simulate common unit conversion mistakes"""
        converted_features = features.copy()
        
        # Common mistakes users make:
        conversion_errors = [
            (0, 0.1),   # period: sometimes in wrong units
            (1, 0.05),  # radius: sometimes Earth vs Jupiter radii confusion
            (2, 0.02),  # temperature: usually OK
            (3, 0.1),   # stellar mass: solar mass confusion
            (4, 0.1),   # stellar radius: solar radius confusion  
            (5, 0.05),  # distance: parsec vs light-year confusion
            (6, 0.0),   # eccentricity: dimensionless
            (7, 0.15)   # planet mass: Earth vs Jupiter mass confusion
        ]
        
        for feature_idx, error_prob in conversion_errors:
            if random.random() < error_prob:
                # Apply random unit conversion factor
                factors = [0.1, 0.33, 3.0, 10.0, 11.2, 317.8]  # Common conversion factors
                factor = random.choice(factors)
                converted_features[feature_idx] *= factor
                
        return converted_features
    
    def add_systematic_biases(self, features):
        """Add systematic observational biases"""
        biased_features = features.copy()
        
        # Transit surveys have selection biases:
        # - Favor shorter periods (easier to detect)
        # - Favor larger planets (bigger signals)
        # - Favor brighter stars (better photometry)
        
        period = biased_features[0]
        radius = biased_features[1]
        
        # Period bias: short periods overrepresented
        if period < 10:
            period_bias = random.uniform(0.9, 1.1)
        else:
            period_bias = random.uniform(1.0, 1.3)  # Longer periods less certain
        
        # Radius bias: small planets harder to detect
        if radius < 1.5:
            radius_bias = random.uniform(1.1, 1.4)  # Overestimate small planets
        else:
            radius_bias = random.uniform(0.95, 1.05)
            
        biased_features[0] *= period_bias
        biased_features[1] *= radius_bias
        
        return biased_features
    
    def create_realistic_dataset(self, data_quality='mixed'):
        """Create realistic observational dataset"""
        realistic_data = []
        
        all_objects = self.perfect_exoplanets + self.perfect_false_positives
        
        for obj in all_objects:
            # Create multiple "observations" of the same object with different qualities
            n_observations = random.randint(1, 3)
            
            for obs in range(n_observations):
                features = obj['features'].copy()
                
                # Apply realistic data challenges
                if data_quality == 'mixed':
                    error_level = random.choice(['excellent', 'realistic', 'poor'])
                else:
                    error_level = data_quality
                
                # Apply various realistic issues
                features = self.add_measurement_errors(features, error_level)
                
                if random.random() < 0.3:  # 30% chance of missing data
                    features = self.introduce_missing_data(features, random.uniform(0.1, 0.4))
                
                if random.random() < 0.2:  # 20% chance of unit errors
                    features = self.apply_unit_conversion_errors(features)
                
                if random.random() < 0.4:  # 40% chance of systematic bias
                    features = self.add_systematic_biases(features)
                
                realistic_data.append({
                    'name': f"{obj['name']}_obs{obs+1}_{error_level}",
                    'features': features,
                    'true_type': obj['type'],
                    'quality': error_level,
                    'source_object': obj['name']
                })
        
        return realistic_data

class RealisticEnsembleTest:
    """Test ensemble on realistic observational data"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_names = [
            'CELESTIAL_ORACLE_NASA',
            'ATMOSPHERIC_WARRIOR_NASA', 
            'BACKYARD_GENIUS_NASA',
            'CHAOS_MASTER_NASA',
            'COSMIC_CONDUCTOR_NASA'
        ]
        
    def load_models(self):
        """Load all models"""
        for model_name in self.model_names:
            try:
                import glob
                model_files = glob.glob(f"{model_name}_2025-09-11*.h5")
                scaler_files = glob.glob(f"{model_name}_SCALER_2025-09-11*.pkl")
                
                if model_files and scaler_files:
                    self.models[model_name] = load_model(model_files[0])
                    self.scalers[model_name] = joblib.load(scaler_files[0])
                    logger.info(f"✅ {model_name}")
                    
            except Exception as e:
                logger.error(f"❌ {model_name}: {e}")
        
        return len(self.models) == 5 and len(self.scalers) == 5
    
    def smart_ensemble_predict(self, features):
        """Smart ensemble prediction with compensation"""
        if len(self.models) == 0:
            return None, "No models loaded"
            
        predictions = {}
        votes = []
        
        # Handle potential issues with realistic data
        try:
            # Check for invalid values
            if any(np.isnan(features)) or any(np.isinf(features)):
                return None, "Invalid feature values (NaN/Inf)"
            
            if any(f < 0 for f in features):
                return None, "Negative feature values"
                
            # Check for extreme values that would break the scalers
            if features[0] > 10000 or features[1] > 100:  # Unrealistic period/radius
                return None, "Extreme feature values"
                
        except Exception as e:
            return None, f"Feature validation failed: {e}"
        
        # Get all individual predictions
        for model_name in self.model_names:
            if model_name in self.models and model_name in self.scalers:
                try:
                    scaled_features = self.scalers[model_name].transform([features])
                    pred = self.models[model_name].predict(scaled_features, verbose=0)[0][0]
                    predictions[model_name] = pred
                    votes.append(pred)
                except Exception as e:
                    logger.error(f"❌ {model_name}: {e}")
                    return None, f"Prediction failed: {e}"
        
        if len(votes) < 5:
            return None, "Insufficient predictions"
        
        # SMART ENSEMBLE LOGIC (same as before)
        celestial = predictions['CELESTIAL_ORACLE_NASA']
        atmospheric = predictions['ATMOSPHERIC_WARRIOR_NASA']
        backyard = predictions['BACKYARD_GENIUS_NASA']
        chaos = predictions['CHAOS_MASTER_NASA']
        cosmic = predictions['COSMIC_CONDUCTOR_NASA']
        
        big_4_votes = [celestial, atmospheric, backyard, chaos]
        big_4_avg = np.mean(big_4_votes)
        big_4_consensus = sum(1 for v in big_4_votes if v > 0.5) / 4
        
        cosmic_is_outlier = (cosmic < 0.3) and (big_4_avg > 0.7)
        
        if cosmic_is_outlier:
            compensated_confidence = big_4_avg * 0.9
            consensus = big_4_consensus
        else:
            avg_confidence = np.mean(votes)
            consensus = sum(1 for v in votes if v > 0.5) / 5
            
            if consensus <= 0.2:
                compensated_confidence = avg_confidence * 0.4
            elif consensus >= 0.8:
                compensated_confidence = min(avg_confidence * 1.1, 0.99)
            else:
                compensated_confidence = avg_confidence
        
        # Enhanced chaos master veto and other logic (same as before)
        if chaos < 0.15 and compensated_confidence > 0.5:
            compensated_confidence *= 0.2
        elif chaos < 0.3 and compensated_confidence > 0.6:
            compensated_confidence *= 0.3
        elif chaos < 0.5 and compensated_confidence > 0.7:
            compensated_confidence *= 0.6
        
        max_individual_vote = max(big_4_votes)
        if max_individual_vote > 0.95 and chaos < 0.9 and compensated_confidence > 0.8:
            compensated_confidence *= 0.4
        
        if big_4_consensus >= 0.75 and big_4_avg > 0.8 and chaos > 0.3:
            compensated_confidence = min(compensated_confidence * 1.1, 0.99)
        
        vote_std = np.std(big_4_votes)
        if vote_std > 0.3 and 0.4 < big_4_avg < 0.8:
            compensated_confidence *= 0.7
            
        if 0.45 <= compensated_confidence <= 0.55:
            compensated_confidence *= 0.85
            
        detected = compensated_confidence > 0.5
        
        return {
            'detected': detected,
            'confidence': compensated_confidence,
            'consensus': consensus,
            'votes': predictions
        }, "Success"

def run_realistic_test():
    """Run the realistic observational data test"""
    print("🌌 REALISTIC OBSERVATIONAL DATA TEST 🌌")
    print("Testing what users would ACTUALLY upload!")
    print("=" * 80)
    
    # Initialize
    simulator = RealisticObservationalDataSimulator()
    tester = RealisticEnsembleTest()
    
    if not tester.load_models():
        print("❌ Failed to load models")
        return
    
    print("✅ All models loaded!")
    
    # Create realistic dataset
    print("\n📊 Generating realistic observational data...")
    realistic_data = simulator.create_realistic_dataset('mixed')
    
    print(f"Generated {len(realistic_data)} realistic observations")
    
    # Test on realistic data
    print("\n🧪 TESTING ON REALISTIC OBSERVATIONAL DATA:")
    print("=" * 80)
    
    results = {
        'total': 0,
        'correct': 0,
        'exoplanet_detected': 0,
        'exoplanet_total': 0,
        'false_pos_rejected': 0,
        'false_pos_total': 0,
        'failed_predictions': 0
    }
    
    quality_performance = {'excellent': [], 'realistic': [], 'poor': []}
    
    for obs in realistic_data:
        results['total'] += 1
        
        result, status = tester.smart_ensemble_predict(obs['features'])
        
        if result is None:
            results['failed_predictions'] += 1
            print(f"💥 {obs['name']:<35} | FAILED: {status}")
            continue
        
        detected = result['detected']
        confidence = result['confidence']
        true_type = obs['true_type']
        quality = obs['quality']
        
        # Determine if prediction is correct
        is_correct = (detected and true_type == 'exoplanet') or (not detected and true_type == 'false_positive')
        
        if is_correct:
            results['correct'] += 1
        
        # Track by type
        if true_type == 'exoplanet':
            results['exoplanet_total'] += 1
            if detected:
                results['exoplanet_detected'] += 1
        else:
            results['false_pos_total'] += 1
            if not detected:
                results['false_pos_rejected'] += 1
        
        # Track by quality
        quality_performance[quality].append(is_correct)
        
        # Display result
        status_icon = "✅" if is_correct else "❌"
        type_icon = "🌍" if true_type == 'exoplanet' else "🚫"
        quality_icon = {"excellent": "🎯", "realistic": "📊", "poor": "📉"}[quality]
        
        print(f"{status_icon} {obs['name']:<35} | {type_icon} | {quality_icon} | Conf: {confidence:.3f}")
    
    # Calculate performance metrics
    overall_accuracy = results['correct'] / (results['total'] - results['failed_predictions'])
    exoplanet_recall = results['exoplanet_detected'] / results['exoplanet_total'] if results['exoplanet_total'] > 0 else 0
    false_pos_precision = results['false_pos_rejected'] / results['false_pos_total'] if results['false_pos_total'] > 0 else 0
    
    print("\n" + "=" * 80)
    print("📊 REALISTIC OBSERVATIONAL DATA PERFORMANCE:")
    print("=" * 80)
    print(f"🎯 Overall Accuracy: {overall_accuracy:.1%} ({results['correct']}/{results['total'] - results['failed_predictions']})")
    print(f"🌍 Exoplanet Detection: {exoplanet_recall:.1%} ({results['exoplanet_detected']}/{results['exoplanet_total']})")
    print(f"🛡️ False Positive Rejection: {false_pos_precision:.1%} ({results['false_pos_rejected']}/{results['false_pos_total']})")
    print(f"💥 Failed Predictions: {results['failed_predictions']}/{results['total']} ({results['failed_predictions']/results['total']:.1%})")
    
    # Performance by data quality
    print(f"\n📈 PERFORMANCE BY DATA QUALITY:")
    for quality, outcomes in quality_performance.items():
        if outcomes:
            quality_acc = sum(outcomes) / len(outcomes)
            print(f"   {quality.capitalize()}: {quality_acc:.1%} ({sum(outcomes)}/{len(outcomes)})")
    
    # Reality check
    print(f"\n🔬 REALITY CHECK:")
    if overall_accuracy >= 0.8:
        print(f"✅ {overall_accuracy:.1%} accuracy on realistic data - EXCELLENT!")
        print("   The ensemble handles real-world observational challenges well.")
        if results['failed_predictions'] / results['total'] < 0.1:
            print("   Low failure rate - conversion layer may not be strictly necessary.")
        else:
            print("   Significant failure rate - conversion layer recommended for robustness.")
    elif overall_accuracy >= 0.6:
        print(f"⚠️ {overall_accuracy:.1%} accuracy on realistic data - NEEDS IMPROVEMENT")
        print("   The ensemble struggles with real observational data.")
        print("   A robust conversion layer is ESSENTIAL.")
    else:
        print(f"❌ {overall_accuracy:.1%} accuracy on realistic data - MAJOR ISSUES")
        print("   The ensemble fails on realistic data.")
        print("   Extensive preprocessing and conversion layer REQUIRED.")
    
    return overall_accuracy, results

if __name__ == "__main__":
    run_realistic_test()