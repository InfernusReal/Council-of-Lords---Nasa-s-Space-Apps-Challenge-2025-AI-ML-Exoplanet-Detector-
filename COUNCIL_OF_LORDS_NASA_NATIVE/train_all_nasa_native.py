#!/usr/bin/env python3
"""
🚀🌌 COUNCIL OF LORDS NASA NATIVE - COMPLETE TRAINING PIPELINE 🌌🚀
Trains all 5 specialist models + Supreme Calibrated Ensemble on real NASA data
MISSION: ACHIEVE 90%+ ACCURACY ON REAL NASA EXOPLANET CATALOG

This script will:
1. Train all 5 NASA-native specialist models
2. Train the NASA-native Supreme Calibrated Ensemble
3. Generate performance reports and model evaluations
4. Create a complete NASA-native Council of Lords ensemble
"""

import sys
import os
import subprocess
import time
from datetime import datetime
import json

def run_training_script(script_name, model_name):
    """Run a training script and capture results"""
    print(f"\\n🚀 STARTING {model_name} TRAINING...")
    print("="*100)
    
    start_time = time.time()
    
    try:
        # Run the training script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        end_time = time.time()
        training_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {model_name} TRAINING COMPLETED SUCCESSFULLY!")
            print(f"⏱️ Training time: {training_time:.1f} seconds")
            return True, training_time, result.stdout
        else:
            print(f"❌ {model_name} TRAINING FAILED!")
            print(f"Error: {result.stderr}")
            return False, training_time, result.stderr
            
    except Exception as e:
        end_time = time.time()
        training_time = end_time - start_time
        print(f"❌ {model_name} TRAINING CRASHED!")
        print(f"Exception: {str(e)}")
        return False, training_time, str(e)

def main():
    """Main training pipeline for NASA-native Council of Lords"""
    
    print("🚀🌌" + "="*80 + "🌌🚀")
    print("🚀 COUNCIL OF LORDS NASA NATIVE - COMPLETE TRAINING PIPELINE")
    print("🚀 REAL NASA CATALOG DATA - NO MORE SYNTHETIC FEATURES")
    print("🚀 TARGET: 90%+ ACCURACY ON REAL EXOPLANET DETECTION")
    print("🚀🌌" + "="*80 + "🌌🚀")
    
    # Training pipeline configuration
    training_pipeline = [
        ("nasa_catalog_data_generator.py", "NASA CATALOG DATA GENERATOR"),
        ("celestial_oracle_nasa_train.py", "CELESTIAL ORACLE NASA AI"),
        ("atmospheric_warrior_nasa_train.py", "ATMOSPHERIC WARRIOR NASA AI"),
        ("backyard_genius_nasa_train.py", "BACKYARD GENIUS NASA AI"),
        ("chaos_master_nasa_train.py", "CHAOS MASTER NASA AI"),
        ("cosmic_conductor_nasa_train.py", "COSMIC CONDUCTOR NASA AI"),
        ("supreme_ensemble_nasa_train.py", "SUPREME CALIBRATED ENSEMBLE NASA AI")
    ]
    
    # Training results tracking
    training_results = {
        "pipeline_start": datetime.now().isoformat(),
        "models": {},
        "summary": {}
    }
    
    total_start_time = time.time()
    successful_models = 0
    failed_models = 0
    
    print(f"\\n📋 TRAINING PIPELINE: {len(training_pipeline)} models to train")
    
    # Execute training pipeline
    for script_name, model_name in training_pipeline:
        
        # Check if script exists
        if not os.path.exists(script_name):
            print(f"⚠️ WARNING: {script_name} not found, skipping {model_name}")
            training_results["models"][model_name] = {
                "status": "SKIPPED",
                "reason": "Script not found",
                "training_time": 0
            }
            continue
        
        # Run training
        success, training_time, output = run_training_script(script_name, model_name)
        
        if success:
            successful_models += 1
            status = "SUCCESS"
        else:
            failed_models += 1
            status = "FAILED"
        
        # Record results
        training_results["models"][model_name] = {
            "status": status,
            "training_time": training_time,
            "script": script_name,
            "output_length": len(output),
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"📊 Progress: {successful_models + failed_models}/{len(training_pipeline)} models processed")
        
        # Small delay between models to prevent resource conflicts
        time.sleep(2)
    
    # Calculate final results
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    
    training_results["pipeline_end"] = datetime.now().isoformat()
    training_results["summary"] = {
        "total_training_time": total_training_time,
        "successful_models": successful_models,
        "failed_models": failed_models,
        "success_rate": successful_models / len(training_pipeline) * 100
    }
    
    # Generate final report
    print("\\n" + "="*100)
    print("🏆 COUNCIL OF LORDS NASA NATIVE TRAINING PIPELINE COMPLETE!")
    print("="*100)
    
    print(f"\\n📊 FINAL TRAINING SUMMARY:")
    print(f"   Total Training Time: {total_training_time:.1f} seconds ({total_training_time/60:.1f} minutes)")
    print(f"   Successful Models: {successful_models}/{len(training_pipeline)}")
    print(f"   Failed Models: {failed_models}/{len(training_pipeline)}")
    print(f"   Success Rate: {training_results['summary']['success_rate']:.1f}%")
    
    print(f"\\n📋 MODEL-BY-MODEL RESULTS:")
    for model_name, results in training_results["models"].items():
        status_emoji = "✅" if results["status"] == "SUCCESS" else "❌" if results["status"] == "FAILED" else "⚠️"
        print(f"   {status_emoji} {model_name}: {results['status']} ({results['training_time']:.1f}s)")
    
    # Save training report
    report_filename = f"NASA_NATIVE_TRAINING_REPORT_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
    with open(report_filename, 'w') as f:
        json.dump(training_results, f, indent=2)
    
    print(f"\\n💾 Training report saved: {report_filename}")
    
    # Final assessment
    if successful_models == len(training_pipeline):
        print("\\n🎉 COMPLETE SUCCESS! ALL NASA-NATIVE MODELS TRAINED!")
        print("🏆 Council of Lords NASA Native is ready for deployment!")
        print("🌟 Ready to achieve 90%+ accuracy on real NASA data!")
    elif successful_models >= len(training_pipeline) * 0.8:  # 80% success
        print("\\n✅ MOSTLY SUCCESSFUL! NASA-native ensemble ready with some limitations.")
        print("🔧 Consider retraining failed models for optimal performance.")
    else:
        print("\\n⚠️ TRAINING PIPELINE HAD SIGNIFICANT ISSUES!")
        print("🔧 Review failed models and training environment.")
        print("💡 Consider running individual training scripts to debug issues.")
    
    print("\\n🚀 NASA-NATIVE COUNCIL OF LORDS TRAINING PIPELINE COMPLETE! 🚀")
    
    return training_results

if __name__ == "__main__":
    # Configure Python environment
    print("🔧 Configuring training environment...")
    
    # Set TensorFlow to not use GPU if not available (prevents crashes)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} available")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🎮 GPU available: {len(gpus)} GPU(s) detected")
        else:
            print("💻 Using CPU training (GPU not detected)")
            
    except ImportError:
        print("⚠️ TensorFlow not available - training may fail")
    
    # Run the complete training pipeline
    results = main()
