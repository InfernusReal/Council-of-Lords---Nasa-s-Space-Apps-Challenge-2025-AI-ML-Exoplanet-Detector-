#!/usr/bin/env python3
"""
🌟🔭 ULTIMATE MASSIVE REALITY CHECK - CLEAN VERSION 🔭🌟
Fixed all bugs - Enhanced false positive rejection + diverse test scenarios
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from supreme_telescope_converter import SupremeTelescopeConverter
from test_supreme_pipeline import load_council_of_lords

def enhanced_council_predict(models, scalers, nasa_params_list):
    """Enhanced prediction with better false positive handling"""
    
    # nasa_params_list is the output from the converter (list format)
    features = np.array(nasa_params_list).reshape(1, -1)
    
    # Extract key parameters for red flag analysis
    koi_period = nasa_params_list[0]
    koi_prad = nasa_params_list[1] 
    koi_depth = nasa_params_list[5] if len(nasa_params_list) > 5 else 0.001
    
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
    suspicious_period = koi_period < 0.5 or koi_period > 100.0
    suspicious_radius = koi_prad > 12.0  # Very large "planet"
    suspicious_depth = koi_depth > 0.015  # Very deep "transit"
    very_short_period = koi_period < 1.0  # Contact binary territory
    
    # Count red flags
    red_flags = 0
    flag_details = []
    
    if suspicious_period:
        red_flags += 1
        flag_details.append(f"Suspicious period: {koi_period:.3f}")
        
    if suspicious_radius:
        red_flags += 1  
        flag_details.append(f"Suspicious radius: {koi_prad:.3f}")
        
    if suspicious_depth:
        red_flags += 1
        flag_details.append(f"Suspicious depth: {koi_depth:.4f}")
        
    if very_short_period:
        red_flags += 1
        flag_details.append(f"Very short period: {koi_period:.3f}")
    
    print(f"🚩 Red flags detected: {red_flags}")
    for flag in flag_details:
        print(f"   - {flag}")
    
    # ENHANCED DECISION LOGIC WITH BETTER THRESHOLDS
    if red_flags >= 3:
        # Too many red flags - force NOT_EXOPLANET
        final_verdict = "NOT_EXOPLANET"
        confidence = 0.85
        print("🚨 TOO MANY RED FLAGS - Forcing NOT_EXOPLANET")
        
    elif red_flags >= 2 and not_exoplanet_votes >= 1:
        # Multiple red flags + at least one dissenting vote
        final_verdict = "NOT_EXOPLANET"
        confidence = 0.80
        print("⚠️ Multiple red flags + dissenting vote - NOT_EXOPLANET")
        
    elif red_flags >= 2 and exoplanet_votes <= 3:
        # Red flags + weak consensus
        final_verdict = "NOT_EXOPLANET"
        confidence = 0.75
        print("⚖️ Red flags + weak consensus - Conservative NOT_EXOPLANET")
        
    elif exoplanet_votes >= 4 and red_flags <= 1:
        # Strong consensus for exoplanet with low red flags
        avg_confidence = np.mean([pred for pred in predictions.values() if pred > 0.5])
        final_verdict = "EXOPLANET"
        confidence = avg_confidence
        print("✨ Strong consensus + low red flags - EXOPLANET")
        
    elif exoplanet_votes >= 3 and red_flags <= 1:
        # Majority vote with low red flags
        avg_confidence = np.mean([pred for pred in predictions.values() if pred > 0.5])
        final_verdict = "EXOPLANET"
        confidence = avg_confidence * 0.9
        print("🤝 Majority consensus + low red flags - EXOPLANET")
        
    else:
        # Unclear case - be conservative
        final_verdict = "NOT_EXOPLANET"
        confidence = 0.65
        print("🤔 Unclear case - Conservative NOT_EXOPLANET")
    
    print(f"⚖️ ENHANCED COUNCIL VERDICT: {final_verdict} (confidence: {confidence:.3f})")
    
    return final_verdict, confidence, votes, predictions

class CleanUltimateTester:
    """Clean version of the ultimate tester"""
    
    def __init__(self):
        self.test_dir = Path("clean_ultimate_test")
        self.test_dir.mkdir(exist_ok=True)
        self.datasets = []
    
    def create_good_exoplanet(self):
        """Clean exoplanet example"""
        print("🌍 Creating Good Exoplanet...")
        
        time = np.arange(0, 80, 30/60/24)
        flux = np.ones(len(time))
        
        # Good planet parameters
        period = 8.5  # Reasonable period
        depth = 0.003  # Reasonable depth
        duration = 3.2 / 24
        
        for i in range(int(80 / period)):
            t_center = i * period + 2.1
            if t_center < time[-1]:
                in_transit = np.abs(time - t_center) < duration / 2
                if np.sum(in_transit) > 0:
                    flux[in_transit] -= depth
        
        flux += np.random.normal(0, 0.0005, len(time))
        return self._save_dataset(time, flux, "good_exoplanet", "exoplanet", "Good Exoplanet")
    
    def create_bad_false_positive(self):
        """Obvious false positive"""
        print("🚨 Creating Bad False Positive...")
        
        time = np.arange(0, 20, 30/60/24)
        flux = np.ones(len(time))
        
        # BAD false positive parameters
        period = 0.4  # Way too short!
        depth = 0.025  # Way too deep!
        duration = 4.0 / 24
        
        for i in range(int(20 / period)):
            t_center = i * period + 0.1
            if t_center < time[-1]:
                in_transit = np.abs(time - t_center) < duration / 2
                if np.sum(in_transit) > 0:
                    flux[in_transit] -= depth
        
        flux += np.random.normal(0, 0.001, len(time))
        return self._save_dataset(time, flux, "bad_false_positive", "false_positive", "Bad False Positive")
    
    def create_contact_binary(self):
        """Contact binary - should be rejected"""
        print("💫 Creating Contact Binary...")
        
        time = np.arange(0, 10, 30/60/24)
        flux = np.ones(len(time))
        
        # Contact binary parameters
        period = 0.6  # Very short
        depth = 0.02  # Deep
        duration = 2.0 / 24
        
        for i in range(int(10 / period)):
            t_center = i * period + 0.2
            if t_center < time[-1]:
                in_transit = np.abs(time - t_center) < duration / 2
                if np.sum(in_transit) > 0:
                    flux[in_transit] -= depth
        
        flux += np.random.normal(0, 0.0008, len(time))
        return self._save_dataset(time, flux, "contact_binary", "false_positive", "Contact Binary")
    
    def create_hot_jupiter(self):
        """Hot Jupiter - should be detected"""
        print("🔥 Creating Hot Jupiter...")
        
        time = np.arange(0, 60, 30/60/24)
        flux = np.ones(len(time))
        
        # Hot Jupiter parameters
        period = 3.2  # Short but reasonable
        depth = 0.008  # Deep but reasonable
        duration = 2.8 / 24
        
        for i in range(int(60 / period)):
            t_center = i * period + 1.5
            if t_center < time[-1]:
                in_transit = np.abs(time - t_center) < duration / 2
                if np.sum(in_transit) > 0:
                    flux[in_transit] -= depth
        
        flux += np.random.normal(0, 0.0006, len(time))
        return self._save_dataset(time, flux, "hot_jupiter", "exoplanet", "Hot Jupiter")
    
    def create_giant_false_positive(self):
        """Giant false positive - multiple red flags"""
        print("🚨 Creating Giant False Positive...")
        
        time = np.arange(0, 15, 30/60/24)
        flux = np.ones(len(time))
        
        # EXTREME false positive
        period = 0.3  # Ridiculously short
        depth = 0.04  # Ridiculously deep
        duration = 3.0 / 24  # Long for such short period
        
        for i in range(int(15 / period)):
            t_center = i * period + 0.1
            if t_center < time[-1]:
                in_transit = np.abs(time - t_center) < duration / 2
                if np.sum(in_transit) > 0:
                    flux[in_transit] -= depth
        
        flux += np.random.normal(0, 0.001, len(time))
        return self._save_dataset(time, flux, "giant_fp", "false_positive", "Giant False Positive")
    
    def _save_dataset(self, time, flux, filename, expected_type, description):
        """Save dataset and return info"""
        filepath = self.test_dir / f"{filename}.csv"
        df = pd.DataFrame({'time': time, 'flux': flux})
        df.to_csv(filepath, index=False)
        
        dataset_info = {
            'file': str(filepath),
            'type': expected_type,
            'name': description,
            'points': len(time),
            'duration': time[-1] - time[0]
        }
        
        self.datasets.append(dataset_info)
        print(f"   ✅ Saved: {len(time)} points, {time[-1]-time[0]:.1f} days")
        return dataset_info
    
    def run_clean_test(self):
        """Run the clean test"""
        print("🌟🔭 CLEAN ULTIMATE REALITY CHECK 🔭🌟")
        print("=" * 70)
        
        # Create test datasets
        print("\n📊 CREATING CLEAN TEST DATASETS...")
        print("-" * 50)
        
        self.create_good_exoplanet()
        self.create_hot_jupiter()
        self.create_bad_false_positive()
        self.create_contact_binary()
        self.create_giant_false_positive()
        
        print(f"\n✅ Created {len(self.datasets)} test scenarios!")
        
        # Initialize testing systems
        print("\n🔥 INITIALIZING ENHANCED SUPREME COUNCIL...")
        converter = SupremeTelescopeConverter()
        models, scalers = load_council_of_lords()
        
        # Test all datasets
        print("\n🏛️ TESTING WITH ENHANCED LOGIC...")
        print("=" * 70)
        
        total_correct = 0
        total_tests = 0
        exoplanet_correct = 0
        exoplanet_total = 0
        fp_correct = 0
        fp_total = 0
        
        results = []
        
        for i, dataset in enumerate(self.datasets, 1):
            print(f"\n[{i}/{len(self.datasets)}] 🎯 TESTING: {dataset['name']}")
            print("-" * 50)
            
            # Load data
            df = pd.read_csv(dataset['file'])
            time = df['time'].values
            flux = df['flux'].values
            
            print(f"🎯 Expected: {dataset['type']}")
            
            # Process with supreme converter
            nasa_params = converter.convert_raw_to_nasa_catalog(time, flux, dataset['name'])
            
            # Get ENHANCED Council verdict
            verdict, confidence, votes, predictions = enhanced_council_predict(models, scalers, nasa_params)
            
            # Check correctness
            correct = False
            if dataset['type'] == "exoplanet" and verdict == "EXOPLANET":
                correct = True
                exoplanet_correct += 1
            elif dataset['type'] == "false_positive" and verdict == "NOT_EXOPLANET":
                correct = True
                fp_correct += 1
            
            if dataset['type'] == "exoplanet":
                exoplanet_total += 1
            else:
                fp_total += 1
            
            result_icon = "🏆" if correct else "💥"
            result_text = "CORRECT" if correct else "INCORRECT"
            print(f"{result_icon} Result: {result_text}")
            
            if correct:
                total_correct += 1
            total_tests += 1
            
            results.append({
                'name': dataset['name'],
                'type': dataset['type'],
                'verdict': verdict,
                'confidence': confidence,
                'correct': correct
            })
        
        # Final results
        print("\n" + "=" * 70)
        print("🏆 CLEAN ULTIMATE TEST RESULTS")
        print("=" * 70)
        
        overall_accuracy = total_correct / total_tests * 100
        print(f"📊 OVERALL: {total_correct}/{total_tests} = {overall_accuracy:.1f}%")
        
        if exoplanet_total > 0:
            exoplanet_accuracy = exoplanet_correct / exoplanet_total * 100
            print(f"🌍 EXOPLANETS: {exoplanet_correct}/{exoplanet_total} = {exoplanet_accuracy:.1f}%")
        
        if fp_total > 0:
            fp_accuracy = fp_correct / fp_total * 100
            print(f"🚫 FALSE POSITIVES: {fp_correct}/{fp_total} = {fp_accuracy:.1f}%")
        
        # Assessment
        if overall_accuracy >= 80:
            print("\n🌟 EXCELLENT - Enhanced logic working!")
        elif overall_accuracy >= 60:
            print("\n🥉 GOOD - Improvement over baseline!")
        else:
            print("\n📈 NEEDS WORK - More tuning required!")
        
        print("\n📋 DETAILED RESULTS:")
        for result in results:
            icon = "✅" if result['correct'] else "❌"
            print(f"{icon} {result['name']:25} | {result['verdict']:12} | {result['confidence']:.3f}")
        
        return results, overall_accuracy

if __name__ == "__main__":
    tester = CleanUltimateTester()
    results, accuracy = tester.run_clean_test()