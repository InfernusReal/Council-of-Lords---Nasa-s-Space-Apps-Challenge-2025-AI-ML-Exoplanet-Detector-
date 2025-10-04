"""
🧪 NASA-NATIVE ENSEMBLE TEST 🧪
Tests the new NASA-native Council of Lords on real-world user data scenarios
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import joblib
import logging
from nasa_catalog_data_generator import NASACatalogDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CUSTOM FUNCTIONS USED BY NASA-NATIVE MODELS
def harmonic_activation(x):
    """Cosmic Conductor harmonic activation"""
    return tf.sin(x) * tf.cos(x * 0.5) + tf.tanh(x)

def celestial_oracle_nasa_loss(y_true, y_pred):
    """Celestial Oracle loss function"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    fn_penalty = tf.where(tf.equal(y_true, 1), 
                         tf.sin(tf.square(1 - y_pred) * np.pi) * 8.0,
                         tf.zeros_like(y_pred))
    fp_penalty = tf.where(tf.equal(y_true, 0), 
                         tf.cos(tf.square(y_pred) * np.pi) * 2.5,
                         tf.zeros_like(y_pred))
    return bce + fn_penalty + fp_penalty

def atmospheric_warrior_nasa_loss(y_true, y_pred):
    """Atmospheric Warrior loss function"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    fn_penalty = tf.where(tf.equal(y_true, 1), 
                         tf.square(1 - y_pred) * 6.0,
                         tf.zeros_like(y_pred))
    fp_penalty = tf.where(tf.equal(y_true, 0), 
                         tf.square(y_pred) * 3.5,
                         tf.zeros_like(y_pred))
    return bce + fn_penalty + fp_penalty

def backyard_genius_nasa_loss(y_true, y_pred):
    """Backyard Genius loss function"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    fn_penalty = tf.where(tf.equal(y_true, 1), 
                         tf.log(1 + tf.exp(1 - y_pred)) * 3.0,
                         tf.zeros_like(y_pred))
    fp_penalty = tf.where(tf.equal(y_true, 0), 
                         tf.log(1 + tf.exp(y_pred)) * 2.0,
                         tf.zeros_like(y_pred))
    return bce + fn_penalty + fp_penalty

def chaos_master_nasa_loss(y_true, y_pred):
    """Chaos Master loss function"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    fn_penalty = tf.where(tf.equal(y_true, 1), 
                         tf.pow(1 - y_pred, 3) * 10.0,
                         tf.zeros_like(y_pred))
    fp_penalty = tf.where(tf.equal(y_true, 0), 
                         tf.pow(y_pred, 3) * 4.0,
                         tf.zeros_like(y_pred))
    return bce + fn_penalty + fp_penalty

def cosmic_conductor_nasa_loss(y_true, y_pred):
    """Cosmic Conductor loss function"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    fn_penalty = tf.where(tf.equal(y_true, 1), 
                         tf.sin(tf.square(1 - y_pred) * np.pi) * 4.0,
                         tf.zeros_like(y_pred))
    fp_penalty = tf.where(tf.equal(y_true, 0), 
                         tf.cos(tf.square(y_pred) * np.pi) * 2.2,
                         tf.zeros_like(y_pred))
    return bce + fn_penalty + fp_penalty

# Register custom functions
get_custom_objects().update({
    'harmonic_activation': harmonic_activation,
    'celestial_oracle_nasa_loss': celestial_oracle_nasa_loss,
    'atmospheric_warrior_nasa_loss': atmospheric_warrior_nasa_loss,
    'backyard_genius_nasa_loss': backyard_genius_nasa_loss,
    'chaos_master_nasa_loss': chaos_master_nasa_loss,
    'cosmic_conductor_nasa_loss': cosmic_conductor_nasa_loss
})

class NASANativeEnsembleTester:
    """Test the NASA-native ensemble on realistic user data"""
    
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
        
    def load_nasa_native_models(self):
        """Load all NASA-native trained models"""
        logger.info("🚀 Loading NASA-native Council of Lords...")
        
        for model_name in self.model_names:
            try:
                # Load model
                model_path = f"{model_name}_2025-09-11*.h5"
                import glob
                model_files = glob.glob(model_path)
                if model_files:
                    model_path = model_files[0]
                    self.models[model_name] = load_model(model_path)
                    logger.info(f"✅ Loaded {model_name}")
                    
                    # Load scaler - correct pattern
                    scaler_path = f"{model_name}_SCALER_2025-09-11*.pkl"
                    scaler_files = glob.glob(scaler_path)
                    if scaler_files:
                        scaler_path = scaler_files[0]
                        self.scalers[model_name] = joblib.load(scaler_path)
                        logger.info(f"✅ Loaded {model_name} scaler")
                    else:
                        logger.warning(f"⚠️ Scaler not found for {model_name}: {scaler_path}")
                else:
                    logger.warning(f"⚠️ Model not found: {model_name}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name}: {e}")
        
        logger.info(f"🏛️ Council loaded: {len(self.models)}/5 models ready")
        logger.info(f"🔧 Scalers loaded: {len(self.scalers)}/5 scalers ready")
        return len(self.models) == 5 and len(self.scalers) == 5
    
    def predict_ensemble(self, features):
        """Run ensemble prediction on features"""
        if len(self.models) == 0:
            return None, "No models loaded"
            
        predictions = {}
        votes = []
        
        for model_name in self.model_names:
            if model_name in self.models and model_name in self.scalers:
                try:
                    # Scale features
                    scaled_features = self.scalers[model_name].transform([features])
                    
                    # Get prediction
                    pred = self.models[model_name].predict(scaled_features, verbose=0)[0][0]
                    predictions[model_name] = pred
                    votes.append(pred)
                    
                except Exception as e:
                    logger.error(f"❌ Prediction failed for {model_name}: {e}")
                    predictions[model_name] = 0.0
                    votes.append(0.0)
        
        if len(votes) == 0:
            return None, "No valid predictions"
            
        # Ensemble logic
        avg_confidence = np.mean(votes)
        consensus = sum(1 for v in votes if v > 0.5) / len(votes)
        agreement = 1.0 - np.std(votes)
        
        # Apply consensus adjustments
        if consensus <= 0.2:  # Low consensus
            final_confidence = avg_confidence * 0.4
        elif consensus >= 0.8:  # Strong consensus
            final_confidence = min(avg_confidence * 1.2, 0.99)
        else:
            final_confidence = avg_confidence
            
        # Chaos Master veto power
        chaos_vote = predictions.get('CHAOS_MASTER_NASA', 0.5)
        if chaos_vote < 0.1 and final_confidence > 0.7:
            final_confidence *= 0.5  # Chaos Master skepticism
            
        detected = final_confidence > 0.5
        
        return {
            'detected': detected,
            'confidence': final_confidence,
            'consensus': consensus,
            'agreement': agreement,
            'individual_votes': predictions,
            'ensemble_stats': {
                'yes_votes': sum(1 for v in votes if v > 0.5),
                'no_votes': sum(1 for v in votes if v <= 0.5),
                'avg_confidence': avg_confidence,
                'vote_range': max(votes) - min(votes)
            }
        }, "Success"
    
    def test_real_exoplanets(self):
        """Test on known confirmed exoplanets"""
        logger.info("🌍 Testing on CONFIRMED EXOPLANETS...")
        
        # Real confirmed exoplanets (NASA Exoplanet Archive data)
        confirmed_exoplanets = [
            {
                'name': 'Kepler-442b (Habitable Zone)',
                'features': [112.31, 1.34, 4402, 0.54, 0.61, 370.46, 0.04, 2.3]
            },
            {
                'name': 'TRAPPIST-1e (Ultra-cool dwarf)',
                'features': [6.099, 0.92, 2559, 0.121, 0.089, 12.43, 0.005, 0.692]
            },
            {
                'name': 'TOI-715b (Recent discovery)',
                'features': [19.3, 1.55, 3470, 0.476, 0.43, 42.0, 0.0, 3.02]
            },
            {
                'name': 'HD 209458b (Hot Jupiter)',
                'features': [3.52, 1.359, 6091, 1.155, 1.148, 47.0, 0.014, 220.5]
            },
            {
                'name': 'K2-18b (Sub-Neptune)',
                'features': [32.94, 2.61, 3457, 0.411, 0.359, 34.0, 0.0, 8.63]
            }
        ]
        
        correct_detections = 0
        total_confidence = 0
        
        for exoplanet in confirmed_exoplanets:
            result, status = self.predict_ensemble(exoplanet['features'])
            
            if result:
                detected = result['detected']
                confidence = result['confidence']
                consensus = result['consensus']
                
                print(f"\n🔍 {exoplanet['name']}:")
                print(f"   Input: {exoplanet['features']}")
                print(f"   Result: {'✅ DETECTED' if detected else '❌ NOT DETECTED'}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Consensus: {consensus:.1%}")
                print(f"   Votes: {result['individual_votes']}")
                
                if detected:
                    correct_detections += 1
                total_confidence += confidence
            else:
                print(f"\n❌ {exoplanet['name']}: {status}")
        
        accuracy = correct_detections / len(confirmed_exoplanets)
        avg_confidence = total_confidence / len(confirmed_exoplanets)
        
        print(f"\n📊 CONFIRMED EXOPLANET RESULTS:")
        print(f"   Accuracy: {accuracy:.1%} ({correct_detections}/{len(confirmed_exoplanets)})")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        
        return accuracy, avg_confidence
    
    def test_false_positives(self):
        """Test on known false positive scenarios"""
        logger.info("🚫 Testing on FALSE POSITIVES...")
        
        # Known false positive patterns
        false_positives = [
            {
                'name': 'Binary Eclipse (Large companion)',
                'features': [2.1, 12.0, 5500, 1.0, 1.0, 200.0, 0.0, 300.0]
            },
            {
                'name': 'Stellar Activity (M-dwarf spots)',
                'features': [25.0, 1.5, 3200, 0.3, 0.3, 50.0, 0.8, 2.0]
            },
            {
                'name': 'Brown Dwarf (Too massive)',
                'features': [100.0, 10.0, 5000, 1.2, 1.2, 150.0, 0.3, 25.0]
            },
            {
                'name': 'Instrumental Noise',
                'features': [0.1, 0.01, 5778, 1.0, 1.0, 150.0, 0.9, 0.001]
            }
        ]
        
        correct_rejections = 0
        total_confidence = 0
        
        for false_pos in false_positives:
            result, status = self.predict_ensemble(false_pos['features'])
            
            if result:
                detected = result['detected']
                confidence = result['confidence']
                consensus = result['consensus']
                chaos_vote = result['individual_votes'].get('CHAOS_MASTER_NASA', 0.5)
                
                print(f"\n🔍 {false_pos['name']}:")
                print(f"   Input: {false_pos['features']}")
                print(f"   Result: {'❌ INCORRECTLY DETECTED' if detected else '✅ CORRECTLY REJECTED'}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Chaos Master: {chaos_vote:.3f}")
                print(f"   Consensus: {consensus:.1%}")
                
                if not detected:
                    correct_rejections += 1
                total_confidence += confidence
            else:
                print(f"\n❌ {false_pos['name']}: {status}")
        
        accuracy = correct_rejections / len(false_positives)
        avg_confidence = total_confidence / len(false_positives)
        
        print(f"\n📊 FALSE POSITIVE RESULTS:")
        print(f"   Accuracy: {accuracy:.1%} ({correct_rejections}/{len(false_positives)})")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        
        return accuracy, avg_confidence
    
    def test_user_data_simulation(self):
        """Simulate typical user data upload scenarios"""
        logger.info("👤 Testing SIMULATED USER DATA...")
        
        # Generate realistic user data
        generator = NASACatalogDataGenerator()
        
        # Test on mixed data (like users would upload)
        n_samples = 20
        X, y = generator.generate_training_data(n_samples, positive_fraction=0.6)
        
        correct_predictions = 0
        
        for i in range(n_samples):
            features = X[i]
            true_label = y[i]  # 1 = exoplanet, 0 = false positive
            
            result, status = self.predict_ensemble(features)
            
            if result:
                predicted = result['detected']
                confidence = result['confidence']
                
                correct = (predicted and true_label == 1) or (not predicted and true_label == 0)
                if correct:
                    correct_predictions += 1
                
                print(f"Sample {i+1}: {'✅' if correct else '❌'} "
                      f"True={'Exoplanet' if true_label else 'FalsePos'} "
                      f"Pred={'Exoplanet' if predicted else 'FalsePos'} "
                      f"Conf={confidence:.3f}")
        
        accuracy = correct_predictions / n_samples
        print(f"\n📊 USER DATA SIMULATION:")
        print(f"   Accuracy: {accuracy:.1%} ({correct_predictions}/{n_samples})")
        
        return accuracy

def main():
    """Run comprehensive NASA-native ensemble test"""
    print("🧪 TESTING NASA-NATIVE COUNCIL OF LORDS 🧪")
    print("=" * 60)
    
    tester = NASANativeEnsembleTester()
    
    # Load models
    success = tester.load_nasa_native_models()
    if not success:
        print("❌ Failed to load all models. Test aborted.")
        return
    
    print("\n🚀 ALL MODELS LOADED SUCCESSFULLY!")
    print("=" * 60)
    
    # Run tests
    try:
        # Test 1: Confirmed exoplanets
        exo_accuracy, exo_confidence = tester.test_real_exoplanets()
        
        # Test 2: False positives
        fp_accuracy, fp_confidence = tester.test_false_positives()
        
        # Test 3: Simulated user data
        sim_accuracy = tester.test_user_data_simulation()
        
        # Overall results
        print("\n" + "=" * 60)
        print("🏆 FINAL ENSEMBLE PERFORMANCE SUMMARY:")
        print("=" * 60)
        print(f"📈 Confirmed Exoplanet Detection: {exo_accuracy:.1%}")
        print(f"🛡️ False Positive Rejection: {fp_accuracy:.1%}")
        print(f"👤 Simulated User Data: {sim_accuracy:.1%}")
        print(f"🎯 Overall Performance: {(exo_accuracy + fp_accuracy + sim_accuracy)/3:.1%}")
        
        # Performance assessment
        overall = (exo_accuracy + fp_accuracy + sim_accuracy) / 3
        if overall >= 0.9:
            print("\n🎉 OUTSTANDING! 90%+ accuracy achieved!")
        elif overall >= 0.8:
            print("\n🚀 EXCELLENT! 80%+ accuracy achieved!")
        elif overall >= 0.7:
            print("\n✅ GOOD! 70%+ accuracy achieved!")
        else:
            print("\n⚠️ NEEDS IMPROVEMENT. Below 70% accuracy.")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
