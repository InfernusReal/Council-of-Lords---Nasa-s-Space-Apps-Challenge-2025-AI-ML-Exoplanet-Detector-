import React from 'react';
import { motion } from 'framer-motion';

const HeroSection = ({ councilStatus }) => {
  return (
    <div className="pt-16 bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          
          <motion.div
            className="space-y-8"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
          >
            <div className="space-y-4">
              <motion.h1 
                className="text-5xl lg:text-6xl font-bold text-gray-900 leading-tight"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
              >
                Exoplanet Detection
                <motion.span 
                  className="block text-blue-600"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.4 }}
                >
                  Reimagined
                </motion.span>
              </motion.h1>
              
              <motion.p 
                className="text-xl text-gray-600 max-w-lg leading-relaxed"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.6 }}
              >
                Level up your team's productivity with Infernus Space. Join projects, earn XP, climb leaderboards, and collaborate seamlessly with NASA-grade ensemble detection.
              </motion.p>
            </div>

            <motion.div 
              className="flex flex-col sm:flex-row gap-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.8 }}
            >
              <motion.button
                className="bg-black text-white px-8 py-3 rounded-lg font-medium hover:bg-gray-800 transition-colors"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Start Building
              </motion.button>
              <motion.button
                className="border-2 border-black text-black px-8 py-3 rounded-lg font-medium hover:bg-black hover:text-white transition-colors"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Watch Demo
              </motion.button>
            </motion.div>

            <motion.div 
              className="flex flex-col sm:flex-row gap-8 pt-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 1.0 }}
            >
              <div>
                <div className="text-3xl font-bold text-gray-900">12</div>
                <div className="text-sm text-gray-500">AI Specialists</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-gray-900">100%</div>
                <div className="text-sm text-gray-500">NASA Accuracy</div>
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${
                    councilStatus?.status === 'READY' ? 'bg-green-500 animate-pulse' : 'bg-red-500'
                  }`}></div>
                  <div>
                    <div className="text-sm font-bold text-gray-900">
                      {councilStatus?.status === 'READY' ? 'COUNCIL ONLINE' : 'COUNCIL OFFLINE'}
                    </div>
                    <div className="text-xs text-gray-500">
                      {councilStatus?.brutal_reality_mode === 'ACTIVE' ? 'Brutal Reality Mode' : 'Status Unknown'}
                    </div>
                  </div>
                </div>
              </div>
              <div>
                <div className="text-3xl font-bold text-gray-900">99.9%</div>
                <div className="text-sm text-gray-500">Uptime</div>
              </div>
            </motion.div>
          </motion.div>

          <motion.div
            className="relative"
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <motion.div 
              className="bg-white rounded-2xl shadow-2xl p-6 border border-gray-200"
              whileHover={{ y: -5 }}
              transition={{ duration: 0.3 }}
            >
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span className="ml-4 text-sm text-gray-600 font-medium">Council Dashboard</span>
                </div>
              </div>

              <div className="flex items-center space-x-4 mb-6 p-4 bg-gray-50 rounded-xl">
                <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center">
                  <span className="text-white font-bold">15</span>
                </div>
                <div className="flex-1">
                  <div className="text-lg font-semibold text-gray-900">Level 15 Developer</div>
                  <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                    <motion.div 
                      className="bg-blue-600 h-2 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: "75%" }}
                      transition={{ duration: 1.5, delay: 1.2 }}
                    ></motion.div>
                  </div>
                  <div className="text-sm text-gray-500 mt-1">2,450 XP</div>
                </div>
              </div>

              <div className="space-y-3">
                <motion.div 
                  className="flex items-center justify-between p-4 bg-green-50 border border-green-200 rounded-xl"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.6, delay: 1.4 }}
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-green-500 rounded-lg flex items-center justify-center">
                      <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div>
                      <div className="font-medium text-gray-900">Exoplanet Detection</div>
                      <div className="text-sm text-gray-500">+250 XP earned</div>
                    </div>
                  </div>
                  <span className="text-green-600 font-medium text-sm">Completed</span>
                </motion.div>

                <motion.div 
                  className="flex items-center justify-between p-4 bg-blue-50 border border-blue-200 rounded-xl"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.6, delay: 1.6 }}
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center">
                      <div className="w-2 h-2 bg-white rounded-full"></div>
                    </div>
                    <div>
                      <div className="font-medium text-gray-900">Council Analysis</div>
                      <div className="text-sm text-gray-500">In progress • 3 days left</div>
                    </div>
                  </div>
                  <span className="text-blue-600 font-medium text-sm">Active</span>
                </motion.div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default HeroSection;