"""
🔭🏛️ FINAL RAW TELESCOPE DATA TO COUNCIL OF LORDS TEST 🏛️🔭
Raw light curve → Simple converter → NASA parameters → Council of Lords → Results
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import logging
import glob
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom functions
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

# Register custom functions
tf.keras.utils.get_custom_objects().update({
    'harmonic_activation': harmonic_activation,
    'celestial_oracle_nasa_loss': celestial_oracle_nasa_loss,
    'atmospheric_warrior_nasa_loss': atmospheric_warrior_nasa_loss,
    'backyard_genius_nasa_loss': backyard_genius_nasa_loss,
    'chaos_master_nasa_loss': chaos_master_nasa_loss,
    'cosmic_conductor_nasa_loss': cosmic_conductor_nasa_loss
})

class SimpleRawToNASA:
    """Super simple raw data to NASA converter"""
    
    def convert(self, time, flux):
        """Convert raw lightcurve to NASA parameters"""
        # Simple period detection
        flux_norm = flux / np.median(flux)
        
        # Look for the deepest dips
        min_flux = np.min(flux_norm)
        depth = 1.0 - min_flux
        
        # Estimate period from time span
        time_span = np.max(time) - np.min(time)
        period = max(1.0, time_span / 3)  # Assume 3 transits minimum
        
        # Sun-like star defaults
        stellar_temp = 5778
        stellar_mass = 1.0
        stellar_radius = 1.0
        
        # Planet size from depth
        planet_radius = np.sqrt(max(0.001, depth)) * stellar_radius * 109.2
        
        # Simple mass estimate
        if planet_radius < 1.5:
            planet_mass = planet_radius ** 3.7
        else:
            planet_mass = planet_radius ** 2.0
        
        return np.array([
            period,         # pl_orbper
            planet_radius,  # pl_rade
            stellar_temp,   # st_teff
            stellar_radius, # st_rad
            stellar_mass,   # st_mass
            100.0,          # sy_dist
            0.0,            # pl_orbeccen
            planet_mass     # pl_bmasse
        ])

class CouncilOfLords:
    """NASA-native Council of Lords"""
    
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
        self.converter = SimpleRawToNASA()
        self.load_models()
    
    def load_models(self):
        """Load the Council models"""
        logger.info("🏛️ Loading Council of Lords...")
        
        for model_name in self.model_names:
            try:
                # Find latest model files
                model_files = glob.glob(f"{model_name}_2025-09-11*.h5")
                scaler_files = glob.glob(f"{model_name}_SCALER_2025-09-11*.pkl")
                
                if model_files and scaler_files:
                    self.models[model_name] = tf.keras.models.load_model(model_files[0], compile=False)
                    self.scalers[model_name] = joblib.load(scaler_files[0])
                    logger.info(f"✅ {model_name}")
                else:
                    logger.error(f"❌ Missing files for {model_name}")
                    
            except Exception as e:
                logger.error(f"❌ {model_name}: {e}")
        
        return len(self.models) == 5
    
    def predict_from_raw_data(self, time, flux):
        """Predict from raw telescope data"""
        # Convert to NASA parameters
        nasa_features = self.converter.convert(time, flux)
        
        # Get individual predictions
        predictions = {}
        votes = []
        
        for model_name in self.model_names:
            if model_name in self.models and model_name in self.scalers:
                try:
                    scaled = self.scalers[model_name].transform([nasa_features])
                    pred = self.models[model_name].predict(scaled, verbose=0)[0][0]
                    predictions[model_name] = pred
                    votes.append(pred)
                except Exception as e:
                    logger.error(f"❌ {model_name}: {e}")
                    predictions[model_name] = 0.5
                    votes.append(0.5)
        
        if len(votes) < 5:
            return {"detected": False, "confidence": 0.0, "reasoning": "Model loading failed"}
        
        # Smart ensemble logic
        celestial = predictions.get('CELESTIAL_ORACLE_NASA', 0.5)
        atmospheric = predictions.get('ATMOSPHERIC_WARRIOR_NASA', 0.5)
        backyard = predictions.get('BACKYARD_GENIUS_NASA', 0.5)
        chaos = predictions.get('CHAOS_MASTER_NASA', 0.5)
        cosmic = predictions.get('COSMIC_CONDUCTOR_NASA', 0.5)
        
        big_4_votes = [celestial, atmospheric, backyard, chaos]
        big_4_avg = np.mean(big_4_votes)
        big_4_consensus = sum(1 for v in big_4_votes if v > 0.5) / 4
        
        # Cosmic Conductor compensation
        cosmic_is_outlier = (cosmic < 0.3) and (big_4_avg > 0.7)
        
        if cosmic_is_outlier:
            confidence = big_4_avg * 0.9
            consensus = big_4_consensus
        else:
            avg_confidence = np.mean(votes)
            consensus = sum(1 for v in votes if v > 0.5) / 5
            
            if consensus <= 0.2:
                confidence = avg_confidence * 0.4
            elif consensus >= 0.8:
                confidence = min(avg_confidence * 1.1, 0.99)
            else:
                confidence = avg_confidence
        
        # Chaos master veto
        if chaos < 0.15 and confidence > 0.5:
            confidence *= 0.2
        elif chaos < 0.3 and confidence > 0.6:
            confidence *= 0.3
        
        detected = confidence > 0.5
        
        return {
            "detected": detected,
            "confidence": confidence,
            "consensus": consensus,
            "votes": predictions,
            "nasa_features": nasa_features.tolist(),
            "reasoning": f"Council verdict: {'EXOPLANET' if detected else 'NO EXOPLANET'} ({confidence:.1%})"
        }

def create_test_transit():
    """Create a clear transit signal"""
    # 10 days, every 30 minutes
    time = np.arange(0, 10, 30/60/24)
    
    # Clear transit parameters
    period = 2.5  # days
    depth = 0.015  # 1.5% depth
    duration = 3 / 24  # 3 hours
    
    # Clean baseline
    flux = np.ones(len(time)) + np.random.normal(0, 0.0005, len(time))
    
    # Add 4 clear transits
    for i in range(4):
        transit_time = i * period + 1.0
        if transit_time < time[-1]:
            in_transit = np.abs(time - transit_time) < duration / 2
            flux[in_transit] *= (1 - depth)
    
    return time, flux

def create_test_false_positive():
    """Create a binary eclipse (false positive)"""
    time = np.arange(0, 5, 20/60/24)
    
    # Binary parameters (suspicious)
    period = 1.2  # Very short
    depth = 0.05  # Very deep
    duration = 4 / 24  # Very long
    
    flux = np.ones(len(time)) + np.random.normal(0, 0.001, len(time))
    
    # Both primary and secondary eclipses
    for i in range(4):
        # Primary
        primary_time = i * period
        if primary_time < time[-1]:
            in_primary = np.abs(time - primary_time) < duration / 2
            flux[in_primary] *= (1 - depth)
        
        # Secondary (half period later)
        secondary_time = i * period + period / 2
        if secondary_time < time[-1]:
            in_secondary = np.abs(time - secondary_time) < duration / 2
            flux[in_secondary] *= (1 - depth * 0.4)
    
    return time, flux

def main():
    """Run the complete test"""
    print("🔭🏛️ FINAL RAW TELESCOPE DATA TO COUNCIL OF LORDS TEST 🏛️🔭")
    print("=" * 80)
    
    # Initialize Council
    council = CouncilOfLords()
    
    if len(council.models) < 5:
        print("❌ Failed to load all Council models!")
        return False
    
    print("✅ Council of Lords loaded successfully!")
    
    # Test 1: Real exoplanet transit
    print("\n🌍 TEST 1: Realistic Exoplanet Transit")
    print("-" * 50)
    
    time1, flux1 = create_test_transit()
    result1 = council.predict_from_raw_data(time1, flux1)
    
    print(f"Raw data points: {len(time1)}")
    print(f"NASA features: {[f'{x:.2f}' for x in result1['nasa_features']]}")
    print(f"Council votes: {[f'{k}: {v:.3f}' for k, v in result1['votes'].items()]}")
    print(f"Result: {result1['reasoning']}")
    
    correct1 = result1['detected']
    print(f"Expected: DETECTED | Got: {'DETECTED' if correct1 else 'NOT DETECTED'} | {'✅ CORRECT' if correct1 else '❌ WRONG'}")
    
    # Test 2: False positive (binary)
    print("\n🚫 TEST 2: Binary Eclipse False Positive")
    print("-" * 50)
    
    time2, flux2 = create_test_false_positive()
    result2 = council.predict_from_raw_data(time2, flux2)
    
    print(f"Raw data points: {len(time2)}")
    print(f"NASA features: {[f'{x:.2f}' for x in result2['nasa_features']]}")
    print(f"Council votes: {[f'{k}: {v:.3f}' for k, v in result2['votes'].items()]}")
    print(f"Result: {result2['reasoning']}")
    
    correct2 = not result2['detected']  # Should NOT detect
    print(f"Expected: NOT DETECTED | Got: {'DETECTED' if result2['detected'] else 'NOT DETECTED'} | {'✅ CORRECT' if correct2 else '❌ WRONG'}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS:")
    print("=" * 80)
    
    total_correct = sum([correct1, correct2])
    print(f"Exoplanet Detection: {'✅ PASS' if correct1 else '❌ FAIL'}")
    print(f"False Positive Rejection: {'✅ PASS' if correct2 else '❌ FAIL'}")
    print(f"Overall Score: {total_correct}/2 ({total_correct/2*100:.0f}%)")
    
    if total_correct == 2:
        print("🎉 PERFECT! Raw telescope data → Council of Lords pipeline WORKS!")
    else:
        print("⚠️ Pipeline needs improvement.")
    
    return total_correct == 2

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)