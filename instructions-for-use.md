# Instructions for Use

## Overview
This project contains an exoplanet detection system with both backend and frontend components, along with comprehensive datasets for training and testing.

🌟 If you're here from NASA judging: Please begin by reading the [README.md](../README.md) file first. It contains the manifesto, feature breakdown, and scientific vision of this system. This is not a demo — this is real exoplanet detection built from scratch using 5 AI models and pure physics intelligence.


## 📡 Real Kepler and TESS Datasets

**Note**: If you want to use real Kepler and TESS datasets, kindly go to the `KeplerNTessDatasets` folder inside the `COUNCIL_OF_LORDS_NASA_NATIVE` folder where you'll find **36 authentic Kepler and TESS datasets** from confirmed exoplanet observations. You'll find them at the top.

**Location**: `COUNCIL_OF_LORDS_NASA_NATIVE/kepplerntess/`  
**Contents**: Real telescope data from NASA missions including Kepler-442, TOI-715, and other confirmed exoplanets

## Backend Setup and Execution

### Running the Backend Server
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend server:
   ```bash
   python main.py
   ```

The backend server will start and be ready to process exoplanet detection requests.

## Frontend Setup and Execution

### Running the Frontend
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies (if not already installed):
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

The frontend will be available in your web browser at the provided local development URL.

## Datasets

The project includes several datasets for training and testing:

### Primary Training Dataset
- **COUNCIL_OF_LORDS_NASA_NATIVE/**: Contains NASA-native datasets and trained models
  - Multiple trained models with .h5 and .pkl files
  - Training scripts for various detectors (Atmospheric Warrior, Backyard Genius, Celestial Oracle, etc.)
  - Real telescope data and catalog generators

### Testing Datasets

#### Comprehensive Testing
- **brutal_reality_test/**: Contains challenging real-world test cases
  - `ground_hell.csv`
  - `heartbreak_binary.csv`
  - `instrumental_demon.csv`
  - `kepler_disaster.csv`
  - `stellar_demon.csv`
  - `tess_nightmare.csv`
  - `tiny_earth_analog.csv`
  - `ultra_contact_binary.csv`

#### Clean Testing
- **clean_ultimate_test/**: Contains cleaner test datasets
  - `bad_false_positive.csv`
  - `contact_binary.csv`
  - `giant_fp.csv`
  - `good_exoplanet.csv`
  - `hot_jupiter.csv`

## Usage Workflow

1. **Start the Backend**: Follow the backend setup instructions to get your detection models running
2. **Launch the Frontend**: Start the web interface for user interactions
3. **Load Test Data**: Use the datasets from the specified folders to test detection accuracy
4. **Analyze Results**: Compare performance across different dataset categories (brutal reality vs clean tests)

## Notes

- The `COUNCIL_OF_LORDS_NASA_NATIVE` folder contains the most comprehensive and up-to-date models
- Use `brutal_reality_test` datasets to stress-test your detection algorithms
- Use `clean_ultimate_test` datasets for baseline performance verification
- Ensure both backend and frontend are running simultaneously for full functionality
