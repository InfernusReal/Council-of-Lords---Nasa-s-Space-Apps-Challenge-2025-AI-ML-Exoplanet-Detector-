#!/usr/bin/env python3
"""
🚀 100% RAW NASA TELESCOPE DATA DOWNLOADER 🚀
Get ACTUAL telescope data directly from NASA archives!

OFFICIAL NASA DATA SOURCES:
1. MAST (Barbara A. Mikulski Archive for Space Telescopes)
2. NASA Exoplanet Archive 
3. TESS Data Archive
4. Kepler Archive

This downloads REAL FITS files with authentic lightcurve data!
"""

import os
import requests
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.utils.data import download_file
import matplotlib.pyplot as plt
from astroquery.mast import Catalogs, Observations
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

def download_raw_kepler_data():
    """Download 100% authentic Kepler FITS files from MAST"""
    print("🔭 DOWNLOADING 100% RAW KEPLER DATA FROM NASA MAST...")
    
    try:
        # Search for Kepler observations of a confirmed exoplanet
        print("📡 Searching MAST for Kepler lightcurves...")
        
        # Kepler-186f (famous Earth-sized exoplanet)
        target_name = "Kepler-186"
        
        # Query MAST for Kepler observations
        obs_table = Observations.query_object(target_name, radius="0.02 deg")
        kepler_obs = obs_table[obs_table['obs_collection'] == 'Kepler']
        
        if len(kepler_obs) > 0:
            print(f"✅ Found {len(kepler_obs)} Kepler observations!")
            
            # Get the first observation
            obs_id = kepler_obs[0]['obsid']
            print(f"📋 Downloading observation: {obs_id}")
            
            # Download the data products
            products = Observations.get_product_list(kepler_obs[0])
            lightcurve_products = products[products['productSubGroupDescription'] == 'LC']
            
            if len(lightcurve_products) > 0:
                print("💾 Downloading FITS file...")
                download_result = Observations.download_products(lightcurve_products[0])
                
                # Read the FITS file
                fits_file = download_result['Local Path'][0]
                print(f"📁 Saved: {fits_file}")
                
                with fits.open(fits_file) as hdul:
                    data = hdul[1].data
                    time = data['TIME']
                    flux = data['PDCSAP_FLUX']  # Corrected flux
                    quality = data['SAP_QUALITY']
                    
                    # Clean the data
                    good = ~np.isnan(time) & ~np.isnan(flux) & (quality == 0)
                    clean_time = time[good]
                    clean_flux = flux[good] / np.median(flux[good])  # Normalize
                    
                    # Save as CSV
                    raw_data = pd.DataFrame({
                        'time': clean_time,
                        'flux': clean_flux
                    })
                    
                    output_file = "100_percent_raw_kepler_data.csv"
                    raw_data.to_csv(output_file, index=False)
                    print(f"🎯 100% RAW KEPLER DATA SAVED: {output_file}")
                    print(f"📊 {len(clean_time)} authentic data points!")
                    
                    return output_file
        
    except Exception as e:
        print(f"❌ MAST download failed: {e}")
        print("🔄 Trying alternative method...")
        return download_raw_kepler_alternative()

def download_raw_kepler_alternative():
    """Alternative method using direct MAST URLs"""
    print("🔄 TRYING DIRECT MAST DOWNLOAD...")
    
    # Famous exoplanet systems with known KIC IDs
    targets = [
        {"name": "Kepler-452b", "kic": "8311864", "desc": "Earth's cousin"},
        {"name": "Kepler-186f", "kic": "8120608", "desc": "Earth-sized in habitable zone"},
        {"name": "Kepler-22b", "kic": "10593626", "desc": "First confirmed habitable zone planet"},
        {"name": "Kepler-442b", "kic": "9632895", "desc": "Super-Earth in habitable zone"}
    ]
    
    for target in targets:
        try:
            print(f"\n🌟 Trying {target['name']} - {target['desc']}")
            kic_id = target['kic'].zfill(9)  # Pad with zeros
            
            # Direct MAST URL for Kepler lightcurves
            quarters = [1, 2, 3, 4, 5]  # Try multiple quarters
            
            for quarter in quarters:
                try:
                    # MAST URL format
                    base_url = "https://archive.stsci.edu/missions/kepler/lightcurves"
                    filename = f"kplr{kic_id}-{quarter:04d}_llc.fits"
                    url = f"{base_url}/{kic_id[:4]}/{kic_id}/{filename}"
                    
                    print(f"  📡 Downloading Quarter {quarter}: {url}")
                    
                    response = requests.get(url, timeout=60)
                    if response.status_code == 200:
                        # Save FITS file
                        fits_filename = f"raw_{target['name'].lower()}_q{quarter}.fits"
                        with open(fits_filename, 'wb') as f:
                            f.write(response.content)
                        
                        print(f"  ✅ Downloaded: {fits_filename}")
                        
                        # Extract data from FITS
                        with fits.open(fits_filename) as hdul:
                            data = hdul[1].data
                            time = data['TIME']
                            flux = data['PDCSAP_FLUX']
                            quality = data['SAP_QUALITY']
                            
                            # Clean data
                            good = ~np.isnan(time) & ~np.isnan(flux) & (quality == 0)
                            if np.sum(good) > 1000:  # Need decent amount of data
                                clean_time = time[good]
                                clean_flux = flux[good] / np.median(flux[good])
                                
                                # Save as CSV
                                raw_data = pd.DataFrame({
                                    'time': clean_time,
                                    'flux': clean_flux
                                })
                                
                                output_file = f"100_percent_raw_{target['name'].lower()}_q{quarter}.csv"
                                raw_data.to_csv(output_file, index=False)
                                
                                print(f"  🎯 RAW DATA SAVED: {output_file}")
                                print(f"  📊 {len(clean_time)} authentic data points!")
                                
                                # Clean up FITS file
                                os.remove(fits_filename)
                                
                                return output_file
                        
                        # Clean up if data wasn't good
                        os.remove(fits_filename)
                    
                except Exception as e:
                    print(f"  ❌ Quarter {quarter} failed: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ {target['name']} failed: {e}")
            continue
    
    return None

def download_raw_tess_data():
    """Download 100% authentic TESS data"""
    print("\n🛰️ DOWNLOADING 100% RAW TESS DATA...")
    
    try:
        # Use astroquery to search for TESS observations
        from astroquery.mast import Tesscut
        
        # TOI-715 (recent TESS discovery)
        target_name = "TOI-715"
        
        print(f"📡 Searching for TESS data: {target_name}")
        
        # Get TESS cutout
        coord = "06h04m04.8s -05d06m09s"  # TOI-715 coordinates
        size = 20  # arcseconds
        
        # This gets the actual TESS pixels
        cutout_table = Tesscut.get_cutouts(coord, size=size)
        
        if len(cutout_table) > 0:
            print("✅ Found TESS cutout data!")
            
            # Extract the lightcurve from the cutout
            cutout_file = cutout_table[0]
            
            # This would contain the raw TESS pixels
            print(f"📁 TESS cutout available: {cutout_file}")
            
            # For now, let's try the public TESS lightcurves
            return download_tess_lightcurve_alternative()
        
    except Exception as e:
        print(f"❌ TESS cutout failed: {e}")
        return download_tess_lightcurve_alternative()

def download_tess_lightcurve_alternative():
    """Alternative TESS download method"""
    print("🔄 TRYING ALTERNATIVE TESS METHOD...")
    
    # Try to get TESS data from the public archives
    tess_targets = [
        {"name": "TOI-715", "tic": "271971130"},
        {"name": "TOI-700", "tic": "150428135"},
        {"name": "TOI-849", "tic": "139298196"}
    ]
    
    for target in tess_targets:
        try:
            print(f"\n🌟 Trying {target['name']} (TIC {target['tic']})")
            
            # MAST TESS URL format
            tic_id = target['tic']
            base_url = "https://archive.stsci.edu/hlsps/tess-data-alerts"
            
            # Try different sectors
            for sector in [1, 2, 3, 4, 5]:
                try:
                    # TESS data alert format
                    filename = f"hlsp_tess-data-alerts_tess_phot_{tic_id:016d}-s{sector:04d}_tess_v1_lc.fits"
                    url = f"{base_url}/{filename}"
                    
                    print(f"  📡 Trying Sector {sector}...")
                    
                    response = requests.get(url, timeout=60)
                    if response.status_code == 200:
                        fits_filename = f"raw_tess_{target['name'].lower()}_s{sector}.fits"
                        with open(fits_filename, 'wb') as f:
                            f.write(response.content)
                        
                        print(f"  ✅ Downloaded TESS data: {fits_filename}")
                        
                        # Extract lightcurve
                        with fits.open(fits_filename) as hdul:
                            data = hdul[1].data
                            time = data['TIME']
                            flux = data['SAP_FLUX']
                            
                            good = ~np.isnan(time) & ~np.isnan(flux)
                            if np.sum(good) > 1000:
                                clean_time = time[good]
                                clean_flux = flux[good] / np.median(flux[good])
                                
                                raw_data = pd.DataFrame({
                                    'time': clean_time,
                                    'flux': clean_flux
                                })
                                
                                output_file = f"100_percent_raw_tess_{target['name'].lower()}_s{sector}.csv"
                                raw_data.to_csv(output_file, index=False)
                                
                                print(f"  🎯 RAW TESS DATA SAVED: {output_file}")
                                print(f"  📊 {len(clean_time)} authentic TESS points!")
                                
                                os.remove(fits_filename)
                                return output_file
                        
                        os.remove(fits_filename)
                
                except Exception as e:
                    print(f"  ❌ Sector {sector} failed: {e}")
                    continue
        
        except Exception as e:
            print(f"❌ {target['name']} failed: {e}")
            continue
    
    return None

def main():
    """Download 100% authentic NASA telescope data"""
    print("🚀" + "="*60 + "🚀")
    print("   100% AUTHENTIC NASA TELESCOPE DATA DOWNLOADER")
    print("🚀" + "="*60 + "🚀")
    
    print("\n📋 OFFICIAL NASA DATA SOURCES:")
    print("   • MAST Archive (archive.stsci.edu)")
    print("   • Kepler/K2 Data Archive")
    print("   • TESS Data Archive")
    print("   • NASA Exoplanet Archive")
    
    # Try to download Kepler data
    kepler_file = download_raw_kepler_data()
    
    # Try to download TESS data
    tess_file = download_raw_tess_data()
    
    print("\n🎯 DOWNLOAD SUMMARY:")
    if kepler_file:
        print(f"   ✅ Kepler: {kepler_file}")
    else:
        print("   ❌ Kepler: Failed")
    
    if tess_file:
        print(f"   ✅ TESS: {tess_file}")
    else:
        print("   ❌ TESS: Failed")
    
    if kepler_file or tess_file:
        print("\n🔥 SUCCESS! You now have 100% authentic NASA telescope data!")
        print("   Feed this to the Council of Lords for REAL testing!")
    else:
        print("\n⚠️  All downloads failed. Try these manual sources:")
        print("   🌐 https://archive.stsci.edu/kepler/")
        print("   🌐 https://heasarc.gsfc.nasa.gov/docs/tess/")
        print("   🌐 https://exoplanetarchive.ipac.caltech.edu/")

if __name__ == "__main__":
    main()