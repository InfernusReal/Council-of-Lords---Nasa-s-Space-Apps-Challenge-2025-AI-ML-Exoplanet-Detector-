import React, { useState } from 'react';import React, { useState } from 'react';import React, { useState } from 'react';import React from 'react';import React, { useState } from 'react';import React, { useState } from 'react';import React, { useState } from 'react'

import Navbar from '../../components/Navbar';

import HeroSection from './components/HeroSection';import Navbar from '../../components/Navbar';

import UploadSection from './components/UploadSection';

import ResultsSection from './components/ResultsSection';import HeroSection from './components/HeroSection';import Navbar from '../../components/Navbar';



const HomePage = () => {import UploadSection from './components/UploadSection';

  const [analysisResults, setAnalysisResults] = useState(null);

  const [isAnalyzing, setIsAnalyzing] = useState(false);import ResultsSection from './components/ResultsSection';import HeroSection from './components/HeroSection';import Navbar from '../../components/Navbar';



  const handleUpload = async () => {

    setIsAnalyzing(true);

    const HomePage = () => {import UploadSection from './components/UploadSection';

    // Simulate analysis

    await new Promise(resolve => setTimeout(resolve, 3000));  const [analysisResults, setAnalysisResults] = useState(null);

    

    setAnalysisResults({  const [isAnalyzing, setIsAnalyzing] = useState(false);import HeroSection from './components/HeroSection';import Navbar from '../../components/Navbar';

      totalExoplanets: 8,

      confidence: 92.4,

      analysisTime: "2.1s",

      detectionMethods: ['Transit', 'Radial Velocity'],  const handleUpload = async () => {const HomePage = () => {

      discoveries: [

        {    setIsAnalyzing(true);

          id: 1,

          name: "WASP-121b",      const [uploadedFile, setUploadedFile] = useState(null);import UploadSection from './components/UploadSection';

          confidence: 97.2,

          method: "Transit",    // Simulate analysis

          type: "Hot Jupiter",

          radius: "1.865 R⊕",    await new Promise(resolve => setTimeout(resolve, 3000));  const [analysisResults, setAnalysisResults] = useState(null);

          period: "1.27 days"

        }    

      ]

    });    setAnalysisResults({import HeroSection from './components/HeroSection';import Navbar from '../../components/Navbar';import { motion, AnimatePresence } from 'framer-motion'

    

    setIsAnalyzing(false);      totalExoplanets: 8,

  };

      confidence: 92.4,  const handleUpload = () => {

  return (

    <div className="min-h-screen bg-white">      analysisTime: "2.1s",

      <Navbar />

      <HeroSection />      detectionMethods: ['Transit', 'Radial Velocity'],    setUploadedFile({ name: "sample_lightcurve.csv", size: "2.5 MB" });const HomePage = () => {

      <UploadSection onUpload={handleUpload} />

      <ResultsSection results={analysisResults} />      discoveries: [

    </div>

  );        {    

};

          id: 1,

export default HomePage;
          name: "WASP-121b",    setTimeout(() => {  return (import UploadSection from './components/UploadSection';

          confidence: 97.2,

          method: "Transit",      setAnalysisResults({

          type: "Hot Jupiter",

          radius: "1.865 R⊕",        verdict: "EXOPLANET DETECTED",    <div className="min-h-screen bg-gray-50">

          period: "1.27 days"

        }        confidence: 94.7,

      ]

    });        planetType: "Hot Jupiter",      <Navbar />import ResultsSection from './components/ResultsSection';import HeroSection from './components/HeroSection';import Header from './components/Header'

    

    setIsAnalyzing(false);        orbitalPeriod: "3.2 days"

  };

      });      <HeroSection />

  return (

    <div className="min-h-screen bg-white">    }, 3000);

      <Navbar />

      <HeroSection />  };      <UploadSection />

      <UploadSection onUpload={handleUpload} />

      <ResultsSection results={analysisResults} />

    </div>

  );  const handleReset = () => {    </div>

};

    setUploadedFile(null);

export default HomePage;
    setAnalysisResults(null);  );const HomePage = () => {import UploadSection from './components/UploadSection';import HeroSection from './components/HeroSection'

  };

};

  return (

    <div className="min-h-screen bg-gray-50">  const [uploadedFile, setUploadedFile] = useState(null);

      <Navbar />

      <HeroSection />export default HomePage;

      <UploadSection onUpload={handleUpload} />  const [analysisResults, setAnalysisResults] = useState(null);import ResultsSection from './components/ResultsSection';import UploadSection from './components/UploadSection'

      {analysisResults && (

        <div className="bg-white py-16">

          <div className="max-w-4xl mx-auto px-4 text-center">

            <h3 className="text-2xl font-bold text-green-600 mb-4">{analysisResults.verdict}</h3>  const handleUpload = () => {import ResultsSection from './components/ResultsSection'

            <p className="text-lg text-gray-700">Confidence: {analysisResults.confidence}%</p>

            <p className="text-gray-600">Type: {analysisResults.planetType}</p>    setUploadedFile({ name: "sample_lightcurve.csv", size: "2.5 MB" });

            <p className="text-gray-600">Period: {analysisResults.orbitalPeriod}</p>

            <button     const HomePage = () => {import CouncilSection from './components/CouncilSection'

              onClick={handleReset}

              className="mt-4 bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"    setTimeout(() => {

            >

              Reset      setAnalysisResults({  const [uploadedFile, setUploadedFile] = useState(null);

            </button>

          </div>        verdict: "EXOPLANET DETECTED",

        </div>

      )}        confidence: 94.7,  const [analysisResults, setAnalysisResults] = useState(null);function HomePage() {

    </div>

  );        planetType: "Hot Jupiter",

};

        orbitalPeriod: "3.2 days",  const [uploadedFile, setUploadedFile] = useState(null)

export default HomePage;
        councilVotes: {

          detection: 11,  const handleUpload = () => {  const [analysisResult, setAnalysisResult] = useState(null)

          falsePositive: 1

        }    // Simulate file upload and analysis  const [isSidebarOpen, setIsSidebarOpen] = useState(false)

      });

    }, 3000);    setUploadedFile({ name: "sample_lightcurve.csv", size: "2.5 MB" });

  };

      const handleFileUpload = (file) => {

  const handleReset = () => {

    setUploadedFile(null);    // Simulate analysis results after a delay    setUploadedFile(file)

    setAnalysisResults(null);

  };    setTimeout(() => {    // Simulate analysis



  return (      setAnalysisResults({    setTimeout(() => {

    <div className="min-h-screen bg-gray-50">

      <Navbar />        verdict: "EXOPLANET DETECTED",      setAnalysisResult({

      <HeroSection />

      <UploadSection onUpload={handleUpload} />        confidence: 94.7,        verdict: "EXOPLANET DETECTED",

      {(uploadedFile || analysisResults) && (

        <ResultsSection         planetType: "Hot Jupiter",        confidence: 97.3,

          uploadedFile={uploadedFile}

          analysisResults={analysisResults}        orbitalPeriod: "3.2 days",        details: {

          onReset={handleReset}

        />        councilVotes: {          transitDepth: "0.123%",

      )}

    </div>          detection: 11,          period: "12.4 days",

  );

};          falsePositive: 1          radius: "1.2 Earth radii",



export default HomePage;        }          temperature: "284K"

      });        }

    }, 3000);      })

  };    }, 3000)

  }

  const handleReset = () => {

    setUploadedFile(null);  const handleReset = () => {

    setAnalysisResults(null);    setUploadedFile(null)

  };    setAnalysisResult(null)

  }

  return (

    <div className="min-h-screen bg-gray-50">  const sidebarItems = [

      <Navbar />    { icon: "🏠", label: "Dashboard", active: true },

      <HeroSection />    { icon: "📊", label: "Analytics", active: false },

      <UploadSection onUpload={handleUpload} />    { icon: "🔬", label: "Detection", active: false },

      {(uploadedFile || analysisResults) && (    { icon: "📁", label: "Projects", active: false },

        <ResultsSection     { icon: "⚙️", label: "Settings", active: false },

          uploadedFile={uploadedFile}  ]

          analysisResults={analysisResults}

          onReset={handleReset}  return (

        />    <div className="min-h-screen bg-gray-50 relative">

      )}      {/* Header */}

    </div>      <Header onSidebarToggle={setIsSidebarOpen} />

  );      

};      {/* Sidebar */}

      <AnimatePresence>

export default HomePage;        {isSidebarOpen && (
          <>
            {/* Overlay */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsSidebarOpen(false)}
              className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40"
            />
            
            {/* Sidebar */}
            <motion.div
              initial={{ x: -320 }}
              animate={{ x: 0 }}
              exit={{ x: -320 }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              className="fixed left-0 top-0 bottom-0 w-80 bg-white shadow-xl z-50 pt-20"
            >
              <div className="p-6">
                <h3 className="text-lg font-bold text-black mb-6">Navigation</h3>
                <nav className="space-y-2">
                  {sidebarItems.map((item, index) => (
                    <motion.a
                      key={index}
                      href="#"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.1 * index }}
                      className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                        item.active 
                          ? 'bg-black text-white' 
                          : 'text-gray-700 hover:bg-gray-100'
                      }`}
                    >
                      <span className="text-xl">{item.icon}</span>
                      <span className="font-medium">{item.label}</span>
                    </motion.a>
                  ))}
                </nav>
                
                <div className="mt-8 p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-semibold text-black mb-2">Recent Projects</h4>
                  <div className="space-y-2">
                    {["Kepler Analysis", "TESS Survey", "Transit Study"].map((project, index) => (
                      <div key={index} className="text-sm text-gray-600">
                        {project}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <div className="relative z-10">
        <HeroSection />
        
        {!analysisResult && (
          <UploadSection onFileUpload={handleFileUpload} uploadedFile={uploadedFile} />
        )}
        
        {analysisResult && (
          <ResultsSection result={analysisResult} onReset={handleReset} />
        )}
        
        <CouncilSection />
      </div>
    </div>
  )
}

export default HomePage