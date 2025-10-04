#!/usr/bin/env python3
"""
🚀 ROBUST NASA RAW DATA DOWNLOADER 🚀
Download 100% authentic NASA telescope data with error handling
"""

import lightkurve as lk
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def download_kepler_raw():
    """Download raw Kepler data with robust error handling"""
    print("🔭 Searching for Kepler raw data...")
    
    # Try multiple confirmed exoplanet targets - EXPANDED TO 15+ TARGETS
    targets = [
        "Kepler-442",   # Super-Earth in habitable zone
        "Kepler-452",   # Earth's cousin
        "Kepler-186",   # First Earth-size in habitable zone
        "Kepler-16",    # Circumbinary planet
        "Kepler-22",    # First confirmed planet in habitable zone
        "Kepler-62",    # Multi-planet system with habitable zone planets
        "Kepler-69",    # Another habitable zone candidate
        "Kepler-78",    # Earth-size with known mass
        "Kepler-296",   # Multi-planet system
        "Kepler-438",   # Most Earth-like planet found
        "Kepler-1649", # Recent Earth-size discovery
        "Kepler-1410", # Hot Jupiter
        "Kepler-1708", # Potential exomoon host
        "Kepler-444",   # Ancient planetary system
        "Kepler-11",    # Tightly packed system
        "KIC 8462852",  # Tabby's Star (famous for anomalous dips)
        "Kepler-37",    # Smallest known exoplanet
        "Kepler-80",    # Resonant chain system
        "Kepler-90",    # 8-planet system (like our solar system)
        "Kepler-1606",  # Recent discovery
    ]
    
    for target in targets:
        try:
            print(f"  🎯 Trying target: {target}")
            
            # Search for light curves
            search_result = lk.search_lightcurve(target, mission='Kepler')
            
            if len(search_result) == 0:
                print(f"    ❌ No data found for {target}")
                continue
                
            print(f"    ✅ Found {len(search_result)} datasets")
            
            # Download the first available quarter
            try:
                lc = search_result[0].download()
                print(f"    📡 Downloaded {len(lc.time)} data points")
                
                # Clean the data
                lc = lc.remove_nans().remove_outliers()
                
                if len(lc.time) < 1000:
                    print(f"    ⚠️  Too few points after cleaning: {len(lc.time)}")
                    continue
                
                # Convert to our format
                time_days = lc.time.value
                flux_normalized = lc.flux.value / np.median(lc.flux.value)
                
                # Save as CSV
                df = pd.DataFrame({
                    'time': time_days,
                    'flux': flux_normalized
                })
                
                filename = f"raw_kepler_{target.replace(' ', '_').replace('-', '_')}.csv"
                df.to_csv(filename, index=False)
                
                print(f"    💾 Saved: {filename}")
                print(f"    📊 {len(df)} authentic NASA data points")
                print(f"    ⏱️  {time_days[-1] - time_days[0]:.1f} days duration")
                
                return df, target
                
            except Exception as e:
                print(f"    ❌ Download failed: {e}")
                continue
                
        except Exception as e:
            print(f"    ❌ Search failed for {target}: {e}")
            continue
    
    return None, None

def download_multiple_kepler_targets():
    """Download multiple Kepler targets and save all successful ones"""
    print("🔭 DOWNLOADING MULTIPLE KEPLER TARGETS...")
    
    successful_downloads = []
    
    # Try multiple confirmed exoplanet targets
    targets = [
        "Kepler-442", "Kepler-452", "Kepler-186", "Kepler-16", "Kepler-22",
        "Kepler-62", "Kepler-69", "Kepler-78", "Kepler-296", "Kepler-438", 
        "Kepler-1649", "Kepler-1410", "Kepler-444", "Kepler-11", "KIC 8462852",
        "Kepler-37", "Kepler-80", "Kepler-90", "Kepler-1606", "Kepler-1708"
    ]
    
    for i, target in enumerate(targets, 1):
        try:
            print(f"\n[{i}/{len(targets)}] 🎯 Trying target: {target}")
            
            # Search for light curves
            search_result = lk.search_lightcurve(target, mission='Kepler')
            
            if len(search_result) == 0:
                print(f"    ❌ No data found for {target}")
                continue
                
            print(f"    ✅ Found {len(search_result)} datasets")
            
            # Download the first available quarter
            try:
                lc = search_result[0].download()
                print(f"    📡 Downloaded {len(lc.time)} data points")
                
                # Clean the data
                lc = lc.remove_nans().remove_outliers()
                
                if len(lc.time) < 1000:
                    print(f"    ⚠️  Too few points after cleaning: {len(lc.time)}")
                    continue
                
                # Convert to our format
                time_days = lc.time.value
                flux_normalized = lc.flux.value / np.median(lc.flux.value)
                
                # Save as CSV
                df = pd.DataFrame({
                    'time': time_days,
                    'flux': flux_normalized
                })
                
                filename = f"raw_kepler_{target.replace(' ', '_').replace('-', '_')}.csv"
                df.to_csv(filename, index=False)
                
                print(f"    💾 Saved: {filename}")
                print(f"    📊 {len(df)} authentic NASA data points")
                print(f"    ⏱️  {time_days[-1] - time_days[0]:.1f} days duration")
                
                successful_downloads.append((target, filename, len(df)))
                
            except Exception as e:
                print(f"    ❌ Download failed: {e}")
                continue
                
        except Exception as e:
            print(f"    ❌ Search failed for {target}: {e}")
            continue
    
    return successful_downloads

def download_multiple_tess_targets():
    """Download multiple TESS targets and save all successful ones"""
    print("🛰️ DOWNLOADING MULTIPLE TESS TARGETS...")
    
    successful_downloads = []
    
    # Try multiple TESS targets
    targets = [
        "TOI-715", "TOI-270", "TOI-178", "TOI-849", "TOI-1338",
        "TOI-2109", "TOI-561", "TOI-125", "TOI-216", "TOI-402",
        "TOI-824", "TOI-1695", "TOI-2180", "TOI-674", "TOI-1728",
        "HD 21749", "LHS 3844", "L 98-59", "GJ 357", "LP 791-18"
    ]
    
    for i, target in enumerate(targets, 1):
        try:
            print(f"\n[{i}/{len(targets)}] 🎯 Trying target: {target}")
            
            # Search for TESS light curves
            search_result = lk.search_lightcurve(target, mission='TESS')
            
            if len(search_result) == 0:
                print(f"    ❌ No TESS data found for {target}")
                continue
                
            print(f"    ✅ Found {len(search_result)} TESS sectors")
            
            # Download the first available sector
            try:
                lc = search_result[0].download()
                print(f"    📡 Downloaded {len(lc.time)} raw TESS points")
                
                # Minimal cleaning (keep it raw!)
                lc = lc.remove_nans()
                
                if len(lc.time) < 1000:
                    print(f"    ⚠️  Too few points: {len(lc.time)}")
                    continue
                
                # Convert to our format
                time_days = lc.time.value
                flux_normalized = lc.flux.value / np.median(lc.flux.value)
                
                # Save as CSV
                df = pd.DataFrame({
                    'time': time_days,
                    'flux': flux_normalized
                })
                
                filename = f"raw_tess_{target.replace(' ', '_').replace('-', '_')}.csv"
                df.to_csv(filename, index=False)
                
                print(f"    💾 Saved: {filename}")
                print(f"    📊 {len(df)} authentic TESS data points")
                print(f"    ⏱️  {time_days[-1] - time_days[0]:.1f} days duration")
                
                successful_downloads.append((target, filename, len(df)))
                
            except Exception as e:
                print(f"    ❌ TESS download failed: {e}")
                continue
                
        except Exception as e:
            print(f"    ❌ TESS search failed for {target}: {e}")
            continue
    
    return successful_downloads

def download_tess_raw():
    """Download raw TESS data with robust error handling"""
    print("🛰️ Searching for TESS raw data...")
    
    # Try multiple TESS targets - EXPANDED TO 15+ TARGETS
    targets = [
        "TOI-715",      # Recent Earth-size discovery
        "TOI-270",      # Multi-planet system
        "TOI-178",      # Resonant chain system
        "TOI-849",      # Exposed planetary core
        "TOI-1338",     # Circumbinary planet
        "TOI-2109",     # Ultra-hot Jupiter
        "TOI-561",      # Ancient planetary system
        "TOI-125",      # Sub-Neptune
        "TOI-216",      # Super-Earth
        "TOI-402",      # Hot Jupiter
        "TOI-824",      # Multi-planet system
        "TOI-1695",     # Recent discovery
        "TOI-2180",     # Long-period Jupiter analog
        "TOI-674",      # Sub-Neptune in gap
        "TOI-1728",     # Mini-Neptune
        "HD 21749",     # TESS-confirmed planet
        "LHS 3844",     # Ultra-short period planet
        "L 98-59",      # Multi-planet system
        "GJ 357",       # Nearby system with habitable zone planet
        "LP 791-18",    # Recently discovered system
    ]
    
    for target in targets:
        try:
            print(f"  🎯 Trying target: {target}")
            
            # Search for TESS light curves
            search_result = lk.search_lightcurve(target, mission='TESS')
            
            if len(search_result) == 0:
                print(f"    ❌ No TESS data found for {target}")
                continue
                
            print(f"    ✅ Found {len(search_result)} TESS sectors")
            
            # Download the first available sector
            try:
                lc = search_result[0].download()
                print(f"    📡 Downloaded {len(lc.time)} raw TESS points")
                
                # Minimal cleaning (keep it raw!)
                lc = lc.remove_nans()
                
                if len(lc.time) < 1000:
                    print(f"    ⚠️  Too few points: {len(lc.time)}")
                    continue
                
                # Convert to our format
                time_days = lc.time.value
                flux_normalized = lc.flux.value / np.median(lc.flux.value)
                
                # Save as CSV
                df = pd.DataFrame({
                    'time': time_days,
                    'flux': flux_normalized
                })
                
                filename = f"raw_tess_{target.replace(' ', '_').replace('-', '_')}.csv"
                df.to_csv(filename, index=False)
                
                print(f"    💾 Saved: {filename}")
                print(f"    📊 {len(df)} authentic TESS data points")
                print(f"    ⏱️  {time_days[-1] - time_days[0]:.1f} days duration")
                
                return df, target
                
            except Exception as e:
                print(f"    ❌ TESS download failed: {e}")
                continue
                
        except Exception as e:
            print(f"    ❌ TESS search failed for {target}: {e}")
            continue
    
    return None, None

def main():
    print("🚀🔭 MASSIVE NASA RAW DATA DOWNLOADER 🔭🚀")
    print("=" * 60)
    print("DOWNLOADING 20+ CONFIRMED EXOPLANETS!")
    print("=" * 60)
    
    # Download multiple Kepler targets
    kepler_downloads = download_multiple_kepler_targets()
    
    # Download multiple TESS targets  
    tess_downloads = download_multiple_tess_targets()
    
    print("\n" + "=" * 60)
    print("📋 MASSIVE DOWNLOAD SUMMARY:")
    print("=" * 60)
    
    total_downloads = len(kepler_downloads) + len(tess_downloads)
    total_points = sum(points for _, _, points in kepler_downloads + tess_downloads)
    
    print(f"✅ KEPLER SUCCESSES: {len(kepler_downloads)}")
    for target, filename, points in kepler_downloads:
        print(f"   🔭 {target}: {points:,} points -> {filename}")
    
    print(f"\n✅ TESS SUCCESSES: {len(tess_downloads)}")
    for target, filename, points in tess_downloads:
        print(f"   🛰️ {target}: {points:,} points -> {filename}")
    
    print(f"\n🎯 GRAND TOTAL:")
    print(f"   📂 Files downloaded: {total_downloads}")
    print(f"   📊 Total data points: {total_points:,}")
    print(f"   🏆 Success rate: {total_downloads}/40 targets")
    
    if total_downloads >= 10:
        print("\n� MASSIVE SUCCESS! The Council now has abundant real NASA data!")
    elif total_downloads >= 5:
        print("\n⚡ GOOD SUCCESS! Multiple planets downloaded!")
    else:
        print("\n🚨 CREATING ULTRA-REALISTIC FALLBACK DATA...")
        
        # Create multiple ultra-realistic fallback datasets
        for i in range(20):
            time = np.linspace(0, np.random.uniform(20, 60), np.random.randint(3000, 8000))
            flux = np.ones(len(time))
            
            # Add real Kepler/TESS-like noise
            flux += np.random.normal(0, 0.0001, len(time))  # Shot noise
            flux += 0.002 * np.sin(2 * np.pi * time / 12.5)  # Systematic trend
            
            # Add a subtle exoplanet signal
            period = np.random.uniform(2, 50)
            depth = np.random.uniform(0.0003, 0.003)
            for j in range(int(time[-1]/period)):
                transit_time = j * period + np.random.uniform(0, period)
                if transit_time < time[-1]:
                    transit_mask = np.abs(time - transit_time) < 0.05
                    flux[transit_mask] -= depth
            
            df = pd.DataFrame({'time': time, 'flux': flux})
            filename = f"fallback_realistic_planet_{i+1:02d}.csv"
            df.to_csv(filename, index=False)
            print(f"💾 Created: {filename} ({len(df)} points)")
        
        print("🎯 Created 20 ultra-realistic fallback datasets!")

if __name__ == "__main__":
    main()