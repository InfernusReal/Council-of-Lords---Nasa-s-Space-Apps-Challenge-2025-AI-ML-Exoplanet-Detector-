import React from 'react';
import { motion } from 'framer-motion';
import CameraExoplanetDetector from './components/CameraExoplanetDetector';

const Telescope = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black text-white p-6">
      <motion.div
        className="max-w-7xl mx-auto"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent">
            🔭 Live Telescope Detection
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Point your camera at the night sky. AI extracts time/flux data and sends to backend for exoplanet detection.
          </p>
        </div>

        {/* PURE LIVE CAMERA ONLY */}
        <CameraExoplanetDetector />
      </motion.div>
    </div>
  );
};

export default Telescope;