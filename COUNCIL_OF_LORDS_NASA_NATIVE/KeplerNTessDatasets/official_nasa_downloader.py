#!/usr/bin/env python3
"""
🚀 OFFICIAL NASA RAW DATA DOWNLOADER 🚀
Using lightkurve - NASA's official Python package for telescope data
"""

import lightkurve as lk
import pandas as pd
import numpy as np
import os

def download_authentic_kepler_data():
    """Download 100% authentic Kepler data using NASA's official tool"""
    print("🔭 DOWNLOADING 100% AUTHENTIC KEPLER DATA...")
    
    targets = [
        "Kepler-452",  # Earth's cousin
        "Kepler-22",   # First confirmed habitable zone planet  
        "Kepler-186",  # Earth-sized planet
        "Kepler-442"   # Super-Earth
    ]
    
    for target in targets:
        try:
            print(f"\n🌟 Downloading {target}...")
            
            # Search for lightcurves
            search_result = lk.search_lightcurve(target, mission='Kepler')
            
            if len(search_result) > 0:
                print(f"   ✅ Found {len(search_result)} quarters of data")
                
                # Download all available quarters
                lc_collection = search_result.download_all()
                
                if lc_collection:
                    # Stitch all quarters together
                    lc = lc_collection.stitch()
                    
                    # Convert to pandas DataFrame
                    df = lc.to_pandas()
                    
                    # Clean the data
                    df = df.dropna()
                    
                    # Save as CSV
                    filename = f"100_percent_authentic_{target.lower().replace('-', '_')}_raw.csv"
                    
                    # Rename columns to match our format
                    if 'time' in df.columns and 'flux' in df.columns:
                        clean_df = pd.DataFrame({
                            'time': df['time'],
                            'flux': df['flux']
                        })
                    else:
                        # Try different column names
                        time_col = [col for col in df.columns if 'time' in col.lower()][0]
                        flux_col = [col for col in df.columns if 'flux' in col.lower()][0]
                        clean_df = pd.DataFrame({
                            'time': df[time_col],
                            'flux': df[flux_col]
                        })
                    
                    clean_df.to_csv(filename, index=False)
                    
                    print(f"   🎯 SAVED: {filename}")
                    print(f"   📊 {len(clean_df)} authentic NASA data points!")
                    
                    return filename
            else:
                print(f"   ❌ No data found for {target}")
                
        except Exception as e:
            print(f"   ❌ Failed {target}: {e}")
            continue
    
    return None

def download_authentic_tess_data():
    """Download 100% authentic TESS data"""
    print("\n🛰️ DOWNLOADING 100% AUTHENTIC TESS DATA...")
    
    targets = [
        "TOI-715",     # Recent super-Earth discovery
        "TOI-700",     # Earth-sized in habitable zone
        "TOI-849",     # Unusual planet
        "TOI-1338",    # Circumbinary planet
    ]
    
    for target in targets:
        try:
            print(f"\n🌟 Downloading {target}...")
            
            # Search for TESS lightcurves
            search_result = lk.search_lightcurve(target, mission='TESS')
            
            if len(search_result) > 0:
                print(f"   ✅ Found {len(search_result)} sectors of data")
                
                # Download all available sectors
                lc_collection = search_result.download_all()
                
                if lc_collection:
                    # Stitch all sectors together
                    lc = lc_collection.stitch()
                    
                    # Convert to pandas DataFrame
                    df = lc.to_pandas()
                    
                    # Clean the data
                    df = df.dropna()
                    
                    # Save as CSV
                    filename = f"100_percent_authentic_{target.lower().replace('-', '_')}_tess_raw.csv"
                    
                    # Rename columns to match our format
                    if 'time' in df.columns and 'flux' in df.columns:
                        clean_df = pd.DataFrame({
                            'time': df['time'],
                            'flux': df['flux']
                        })
                    else:
                        # Try different column names
                        time_col = [col for col in df.columns if 'time' in col.lower()][0]
                        flux_col = [col for col in df.columns if 'flux' in col.lower()][0]
                        clean_df = pd.DataFrame({
                            'time': df[time_col],
                            'flux': df[flux_col]
                        })
                    
                    clean_df.to_csv(filename, index=False)
                    
                    print(f"   🎯 SAVED: {filename}")
                    print(f"   📊 {len(clean_df)} authentic TESS data points!")
                    
                    return filename
            else:
                print(f"   ❌ No data found for {target}")
                
        except Exception as e:
            print(f"   ❌ Failed {target}: {e}")
            continue
    
    return None

def main():
    """Download 100% authentic NASA telescope data"""
    print("🚀" + "="*60 + "🚀")
    print("   100% AUTHENTIC NASA DATA DOWNLOADER")
    print("   Using lightkurve - NASA's Official Python Package")
    print("🚀" + "="*60 + "🚀")
    
    # Download Kepler data
    kepler_file = download_authentic_kepler_data()
    
    # Download TESS data  
    tess_file = download_authentic_tess_data()
    
    print("\n🎯 DOWNLOAD SUMMARY:")
    if kepler_file:
        print(f"   ✅ Kepler: {kepler_file}")
    
    if tess_file:
        print(f"   ✅ TESS: {tess_file}")
    
    if kepler_file or tess_file:
        print("\n🔥 SUCCESS! You now have 100% authentic NASA telescope data!")
        print("   This is the REAL DEAL - actual telescope measurements!")
        print("   Feed this to the Council of Lords for ultimate testing!")
    else:
        print("\n⚠️  No data downloaded. Check your internet connection.")

if __name__ == "__main__":
    main()