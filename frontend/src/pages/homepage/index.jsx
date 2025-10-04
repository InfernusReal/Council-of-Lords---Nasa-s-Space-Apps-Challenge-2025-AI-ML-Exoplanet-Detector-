import React, { useState, useEffect } from 'react';
import Navbar from '../../components/Navbar';
import HeroSection from './components/HeroSection';
import UploadSection from './components/UploadSection';
import ResultsSection from './components/ResultsSection';
import PlanetVisualization from './components/PlanetVisualization';
import { councilAPI } from '../../services/api';

const HomePage = () => {
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [councilStatus, setCouncilStatus] = useState(null);

  useEffect(() => {
    // Check Council status on page load
    const checkCouncilStatus = async () => {
      try {
        const health = await councilAPI.getHealth();
        const status = await councilAPI.getStatus();
        setCouncilStatus({ ...health, ...status });
      } catch (error) {
        console.error('Failed to check Council status:', error);
      }
    };

    checkCouncilStatus();
  }, []);

  const handleAnalysisStart = () => {
    setIsAnalyzing(true);
    setAnalysisResults(null);
  };

  const handleAnalysisComplete = (results) => {
    setIsAnalyzing(false);
    setAnalysisResults(results);
    
    // Scroll to results section
    setTimeout(() => {
      const resultsSection = document.getElementById('results-section');
      if (resultsSection) {
        resultsSection.scrollIntoView({ behavior: 'smooth' });
      }
    }, 100);
  };

  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <HeroSection councilStatus={councilStatus} />
      <UploadSection 
        onAnalysisStart={handleAnalysisStart}
        onAnalysisComplete={handleAnalysisComplete}
      />
      <div id="results-section">
        <ResultsSection results={analysisResults} />
      </div>
      <PlanetVisualization results={analysisResults} />
    </div>
  );
};

export default HomePage;