#!/usr/bin/env python3
"""
🔭⚔️ REAL TELESCOPE DATA vs COUNCIL OF LORDS BATTLE ARENA ⚔️🔭

This script takes the REAL messy telescope data with all its challenges and 
throws it at our Council of Lords to see how they perform against reality!

Real challenges included:
- Instrumental systematic trends  
- Stellar variability and rotation
- Correlated noise patterns
- Data gaps and bad quality flags
- Complex transit shapes (not perfect boxes)
- Secondary eclipses in binaries
- Position-dependent systematics
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import logging
import glob
from pathlib import Path

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
                    dummy_scaler.mean_ = np.zeros(15)
                    dummy_scaler.scale_ = np.ones(15)
                    scalers[specialist] = dummy_scaler
                    
        except Exception as e:
            print(f"  ❌ Failed to load {specialist}: {e}")
            logger.error(f"Error loading {specialist}: {e}")
    
    if len(models) < 3:
        raise Exception("Need at least 3 specialists for Council voting!")
    
    print(f"🎯 Council assembled: {len(models)} specialists ready!")
    return models, scalers

def advanced_feature_extraction(time, flux):
    """Extract features from REAL messy telescope data"""
    
    # Handle data gaps and bad points
    finite_mask = np.isfinite(time) & np.isfinite(flux)
    time = time[finite_mask]
    flux = flux[finite_mask]
    
    if len(time) < 50:
        return None  # Not enough data
    
    # Detrend systematic effects (this is critical for real data!)
    # Method 1: Polynomial detrending
    poly_coeffs = np.polyfit(time, flux, deg=3)
    systematic_trend = np.polyval(poly_coeffs, time)
    detrended_flux = flux - systematic_trend + np.median(flux)
    
    # Method 2: Moving median filter for stellar variability
    window_size = max(int(len(flux) * 0.1), 10)
    if window_size % 2 == 0:
        window_size += 1
    
    stellar_baseline = ndimage.median_filter(detrended_flux, size=window_size)
    cleaned_flux = detrended_flux - stellar_baseline + np.median(detrended_flux)
    
    # Now extract features from the cleaned light curve
    features = {}
    
    # Basic statistics
    features['period'] = estimate_period(time, cleaned_flux)
    features['transit_depth'] = estimate_transit_depth(cleaned_flux)
    features['duration'] = estimate_duration(time, cleaned_flux, features['period'])
    features['rp_rs'] = np.sqrt(features['transit_depth']) if features['transit_depth'] > 0 else 0
    features['stellar_mass'] = 1.0  # Default assumption
    features['stellar_radius'] = 1.0  # Default assumption
    features['semi_major_axis'] = estimate_semi_major_axis(features['period'], features['stellar_mass'])
    features['inclination'] = estimate_inclination(features['duration'], features['period'], 
                                                 features['semi_major_axis'], features['stellar_radius'])
    features['eccentricity'] = 0.0  # Assume circular for now
    features['impact_parameter'] = features['semi_major_axis'] * np.cos(np.radians(features['inclination'])) / features['stellar_radius']
    
    # Real-data specific features
    features['noise_level'] = np.std(cleaned_flux)
    features['systematic_amplitude'] = np.std(flux - detrended_flux)
    features['data_quality'] = assess_data_quality(time, flux)
    features['stellar_variability'] = assess_stellar_variability(time, flux)
    features['secondary_eclipse'] = detect_secondary_eclipse(time, cleaned_flux, features['period'])
    
    # Validation
    if features['period'] <= 0 or features['period'] > 100:
        features['period'] = 10.0  # Default
    if features['transit_depth'] <= 0 or features['transit_depth'] > 0.1:
        features['transit_depth'] = 0.01  # Default
    if features['duration'] <= 0 or features['duration'] > 1:
        features['duration'] = 0.1  # Default
    
    return features

def estimate_period(time, flux):
    """Estimate orbital period from real noisy data"""
    try:
        # Use autocorrelation for period finding
        dt = np.median(np.diff(time))
        max_period_days = min(30, (time[-1] - time[0]) / 3)
        max_lag = int(max_period_days / dt)
        
        if max_lag > len(flux) // 2:
            max_lag = len(flux) // 2
        
        autocorr = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode='full')
        autocorr = autocorr[len(autocorr)//2:][:max_lag]
        
        # Find peaks in autocorrelation
        peaks, _ = find_peaks(autocorr, height=np.max(autocorr) * 0.3, distance=int(1.0/dt))
        
        if len(peaks) > 0:
            period_samples = peaks[0]
            period = period_samples * dt
            
            # Validate period range
            if 0.5 <= period <= max_period_days:
                return period
        
        # Fallback: look for periodic dips
        # Find negative outliers
        threshold = np.mean(flux) - 2 * np.std(flux)
        dips = time[flux < threshold]
        
        if len(dips) >= 2:
            dip_intervals = np.diff(dips)
            # Find most common interval
            if len(dip_intervals) > 0:
                period = np.median(dip_intervals[dip_intervals > 0.5])
                if 0.5 <= period <= max_period_days:
                    return period
        
        return 5.0  # Default fallback
        
    except:
        return 5.0

def estimate_transit_depth(flux):
    """Estimate transit depth from real noisy data"""
    try:
        # Find the deepest consistent dips
        median_flux = np.median(flux)
        
        # Look for points significantly below median
        threshold = median_flux - 2 * np.std(flux)
        in_transit = flux < threshold
        
        if np.sum(in_transit) > 3:  # Need multiple points
            transit_depth = median_flux - np.median(flux[in_transit])
            return max(0, min(0.1, transit_depth))  # Clamp to reasonable range
        
        return 0.005  # Default small depth
        
    except:
        return 0.005

def estimate_duration(time, flux, period):
    """Estimate transit duration"""
    try:
        if period <= 0:
            return 0.1
        
        # Find transit points
        median_flux = np.median(flux)
        threshold = median_flux - 1.5 * np.std(flux)
        in_transit = flux < threshold
        
        if np.sum(in_transit) > 1:
            # Find longest continuous sequence
            transit_times = time[in_transit]
            if len(transit_times) > 1:
                duration = np.max(transit_times) - np.min(transit_times)
                # Make sure it's reasonable compared to period
                max_duration = period * 0.2  # Max 20% of period
                return min(duration, max_duration)
        
        # Default: 3% of period
        return period * 0.03
        
    except:
        return 0.1

def estimate_semi_major_axis(period, stellar_mass):
    """Estimate semi-major axis using Kepler's third law"""
    try:
        # a^3 = (G * M * P^2) / (4 * pi^2)
        # In solar units: a = (P^2 * M)^(1/3) where P in years, M in solar masses
        period_years = period / 365.25
        a_au = (period_years**2 * stellar_mass)**(1/3)
        a_solar_radii = a_au * 215.0  # 1 AU = ~215 solar radii
        return a_solar_radii
    except:
        return 10.0

def estimate_inclination(duration, period, semi_major_axis, stellar_radius):
    """Estimate orbital inclination"""
    try:
        if duration <= 0 or period <= 0 or semi_major_axis <= 0:
            return 89.0  # Default near edge-on
        
        # Duration relates to chord length across star
        # duration/period = chord_length / (2*pi*a)
        chord_fraction = (duration / period) * 2 * np.pi
        chord_length = chord_fraction * semi_major_axis
        
        # For grazing transit: chord_length ≈ 2*R_star
        # cos(i) = sqrt(1 - (chord_length/(2*R_star))^2) * R_star/a
        if chord_length > 0 and chord_length < 4 * stellar_radius:
            cos_i = np.sqrt(1 - (chord_length/(2*stellar_radius))**2) * stellar_radius/semi_major_axis
            cos_i = np.clip(cos_i, 0, 1)
            inclination = np.degrees(np.arccos(cos_i))
            return np.clip(inclination, 80, 90)  # Transit requires high inclination
        
        return 89.0
        
    except:
        return 89.0

def assess_data_quality(time, flux):
    """Assess overall data quality (0-1 scale)"""
    try:
        # Factor in data gaps, noise level, systematic trends
        
        # Data completeness
        expected_points = (time[-1] - time[0]) / np.median(np.diff(time))
        completeness = len(time) / expected_points
        
        # Noise assessment
        noise_level = np.std(flux) / np.median(flux)
        noise_score = np.exp(-noise_level * 1000)  # Lower noise = higher score
        
        # Systematic trends
        poly_fit = np.polyfit(time, flux, deg=2)
        systematic_trend = np.std(np.polyval(poly_fit, time) - np.median(flux))
        systematic_score = np.exp(-systematic_trend * 500)
        
        # Combined quality score
        quality = (completeness * 0.4 + noise_score * 0.3 + systematic_score * 0.3)
        return np.clip(quality, 0, 1)
        
    except:
        return 0.5

def assess_stellar_variability(time, flux):
    """Assess stellar variability level"""
    try:
        # Remove any obvious transits first
        median_flux = np.median(flux)
        threshold = median_flux - 3 * np.std(flux)
        non_transit = flux > threshold
        
        if np.sum(non_transit) < len(flux) * 0.8:
            clean_flux = flux[non_transit]
        else:
            clean_flux = flux
        
        # Measure variability in the "clean" light curve
        variability = np.std(clean_flux) / np.median(clean_flux)
        return np.clip(variability * 1000, 0, 0.1)  # Scale to reasonable range
        
    except:
        return 0.01

def detect_secondary_eclipse(time, flux, period):
    """Detect secondary eclipse (sign of eclipsing binary)"""
    try:
        if period <= 0:
            return 0
        
        # Phase the data
        phase = ((time - time[0]) % period) / period
        
        # Look for dip around phase 0.5 (secondary eclipse)
        secondary_phase_mask = (phase > 0.4) & (phase < 0.6)
        
        if np.sum(secondary_phase_mask) > 5:
            secondary_flux = flux[secondary_phase_mask]
            baseline_flux = np.median(flux)
            secondary_depth = baseline_flux - np.median(secondary_flux)
            
            # Return secondary depth relative to primary
            primary_depth = estimate_transit_depth(flux)
            if primary_depth > 0:
                return secondary_depth / primary_depth
        
        return 0
        
    except:
        return 0

def convert_to_nasa_params(features):
    """Convert extracted features to NASA catalog format (8 features only)"""
    
    # Map our extracted features to the 8 NASA catalog features:
    # 1. pl_orbper - Orbital period (days)
    # 2. pl_rade - Planet radius (Earth radii) 
    # 3. st_teff - Stellar temperature (Kelvin)
    # 4. st_rad - Stellar radius (solar radii)
    # 5. st_mass - Stellar mass (solar masses)
    # 6. sy_dist - Distance (parsecs)
    # 7. pl_orbeccen - Orbital eccentricity
    # 8. pl_bmasse - Planet mass (Earth masses)
    
    # Extract the 8 NASA catalog parameters
    nasa_params = np.array([
        features['period'],                    # pl_orbper
        features['rp_rs'] * 11.0,             # pl_rade (convert to Earth radii, assuming solar radius)
        5778.0,                               # st_teff (assume Sun-like star)
        features['stellar_radius'],           # st_rad
        features['stellar_mass'],             # st_mass
        100.0,                                # sy_dist (assume 100 parsecs default)
        features['eccentricity'],             # pl_orbeccen
        features['rp_rs']**3 * 317.8          # pl_bmasse (rough mass estimate)
    ])
    
    # Ensure no NaN or inf values
    nasa_params = np.nan_to_num(nasa_params, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Validate ranges to match NASA catalog distributions
    nasa_params[0] = np.clip(nasa_params[0], 0.1, 10000)      # period
    nasa_params[1] = np.clip(nasa_params[1], 0.1, 100)        # planet radius
    nasa_params[2] = np.clip(nasa_params[2], 3000, 8000)      # stellar temp
    nasa_params[3] = np.clip(nasa_params[3], 0.5, 3.0)        # stellar radius
    nasa_params[4] = np.clip(nasa_params[4], 0.5, 3.0)        # stellar mass
    nasa_params[5] = np.clip(nasa_params[5], 10, 1000)        # distance
    nasa_params[6] = np.clip(nasa_params[6], 0.0, 0.8)        # eccentricity
    nasa_params[7] = np.clip(nasa_params[7], 0.1, 1000)       # planet mass
    
    return nasa_params.reshape(1, -1)

def council_of_lords_predict(models, scalers, features):
    """Get Council of Lords prediction on real data"""
    
    nasa_params = convert_to_nasa_params(features)
    
    votes = {}
    predictions = {}
    
    print("  🗳️  Council voting in session...")
    
    for specialist_name, model in models.items():
        try:
            scaler = scalers[specialist_name]
            
            # Scale features
            scaled_features = scaler.transform(nasa_params)
            
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

def test_real_data():
    """Test Council of Lords on real telescope data"""
    print("🔭⚔️ COUNCIL OF LORDS vs REAL TELESCOPE DATA ARENA ⚔️🔭")
    print("=" * 70)
    
    # Load the Council
    models, scalers = load_council_of_lords()
    
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
        print("-" * 50)
        
        # Load data
        data_path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(data_path)
        
        time = df['time'].values
        flux = df['flux'].values
        
        # Load metadata for expected result
        meta_file = csv_file.replace('.csv', '_metadata.txt')
        meta_path = os.path.join(data_dir, meta_file)
        
        expected_type = "unknown"
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta_content = f.read()
                if "Type: exoplanet" in meta_content:
                    expected_type = "exoplanet"
                elif "Type: false_positive" in meta_content:
                    expected_type = "false_positive"
        
        print(f"📊 Data points: {len(time)}")
        print(f"⏱️  Duration: {time[-1] - time[0]:.1f} days")
        print(f"🎯 Expected: {expected_type}")
        print()
        
        # Extract features from real messy data
        print("🔧 Extracting features from real telescope data...")
        features = advanced_feature_extraction(time, flux)
        
        if features is None:
            print("❌ Failed to extract features - data too poor quality")
            continue
        
        print(f"  ✅ Period: {features['period']:.2f} days")
        print(f"  ✅ Transit depth: {features['transit_depth']:.4f}")
        print(f"  ✅ Duration: {features['duration']:.3f} days") 
        print(f"  ✅ Data quality: {features['data_quality']:.2f}")
        print(f"  ✅ Secondary eclipse: {features['secondary_eclipse']:.3f}")
        print()
        
        # Get Council verdict
        verdict, confidence, votes, predictions = council_of_lords_predict(models, scalers, features)
        
        print(f"\n🏛️ COUNCIL VERDICT: {verdict}")
        print(f"🎯 Confidence: {confidence:.3f}")
        
        # Check if correct
        correct = False
        if expected_type == "exoplanet" and verdict == "EXOPLANET":
            correct = True
        elif expected_type == "false_positive" and verdict == "NOT_EXOPLANET":
            correct = True
        
        result_icon = "✅" if correct else "❌"
        print(f"{result_icon} Result: {'CORRECT' if correct else 'INCORRECT'}")
        
        if correct:
            total_correct += 1
        total_tests += 1
        
        print("=" * 50)
    
    # Final results
    print(f"\n🏆 FINAL BATTLE RESULTS:")
    print(f"📊 Total tests: {total_tests}")
    print(f"✅ Correct predictions: {total_correct}")
    print(f"🎯 Accuracy: {total_correct/total_tests*100:.1f}%" if total_tests > 0 else "No tests completed")
    
    if total_tests > 0:
        accuracy = total_correct / total_tests * 100
        if accuracy >= 80:
            print("🏆 EXCELLENT! Council of Lords dominates real data!")
        elif accuracy >= 60:
            print("🥈 GOOD! Council holds their ground against real challenges!")
        elif accuracy >= 40:
            print("🥉 STRUGGLING! Real data is much harder than synthetic!")
        else:
            print("💥 DEFEAT! Council needs more training on real data!")

if __name__ == "__main__":
    test_real_data()