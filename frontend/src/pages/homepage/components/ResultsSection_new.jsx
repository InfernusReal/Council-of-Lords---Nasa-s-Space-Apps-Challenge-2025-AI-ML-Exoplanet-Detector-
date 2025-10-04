import React from 'react';
import { motion } from 'framer-motion';

const ResultsSection = ({ results }) => {
  if (!results) {
    return (
      <div className="bg-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="text-3xl font-bold text-gray-900 mb-8">Council Analysis</h2>
              <p className="text-xl text-gray-600">Upload a dataset to begin analysis by the Council of Lords</p>
            </motion.div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Council of Lords Analysis</h2>
            <p className="text-xl text-gray-600">
              The Supreme Council has deliberated on your dataset
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
            {/* Verdict Card */}
            <motion.div 
              className="bg-white rounded-lg shadow-xl p-6 border-l-4"
              style={{
                borderLeftColor: results.councilVerdict === 'EXOPLANET' ? '#10B981' : '#EF4444'
              }}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-gray-900">Final Verdict</h3>
                <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                  results.councilVerdict === 'EXOPLANET' 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-red-100 text-red-800'
                }`}>
                  {results.councilVerdict}
                </div>
              </div>
              <div className="text-3xl font-bold mb-2" style={{
                color: results.councilVerdict === 'EXOPLANET' ? '#10B981' : '#EF4444'
              }}>
                {(results.confidence * 100).toFixed(1)}%
              </div>
              <p className="text-gray-600">{results.reason}</p>
            </motion.div>

            {/* Council Votes */}
            <motion.div 
              className="bg-white rounded-lg shadow-xl p-6"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Council Votes</h3>
              <div className="space-y-3">
                {results.individualVotes && Object.entries(results.individualVotes).map(([member, vote]) => (
                  <div key={member} className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-700">
                      {member.replace('_', ' ')}
                    </span>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        vote.prediction === 'EXOPLANET' 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {vote.prediction}
                      </span>
                      <span className="text-sm text-gray-600">
                        {(vote.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Red Flags Section */}
          {results.redFlags && results.redFlags.length > 0 && (
            <motion.div 
              className="bg-red-50 rounded-lg p-6 mb-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <h3 className="text-lg font-semibold text-red-800 mb-4">🚩 Red Flags Detected</h3>
              <div className="space-y-2">
                {results.redFlags.map((flag, index) => (
                  <div key={index} className="flex items-start space-x-2">
                    <span className="text-red-600 mt-1">⚠️</span>
                    <span className="text-red-700">{flag}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Pipeline Logs */}
          {results.pipelineLogs && results.pipelineLogs.length > 0 && (
            <motion.div 
              className="bg-gray-50 rounded-lg p-6 mb-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.5 }}
            >
              <h3 className="text-lg font-semibold text-gray-800 mb-4">📊 Analysis Pipeline</h3>
              <div className="bg-black rounded p-4 max-h-64 overflow-y-auto">
                <div className="font-mono text-sm space-y-1">
                  {results.pipelineLogs.map((log, index) => (
                    <div key={index} className="text-green-400">
                      {log}
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {/* Discoveries Section */}
          {results.discoveries && results.discoveries.length > 0 && (
            <motion.div 
              className="bg-white rounded-lg shadow-xl p-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              <h3 className="text-xl font-semibold text-gray-900 mb-6">🌍 Planetary Discoveries</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Planet
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Type
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Detection Method
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Radius (Earth = 1)
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Period (days)
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Confidence
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {results.discoveries.map((planet, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="text-sm font-medium text-gray-900">
                              {planet.name || `Planet ${index + 1}`}
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-blue-100 text-blue-800">
                            {planet.type}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {planet.method}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {planet.radius}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {planet.period}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="flex-1 mr-2">
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div 
                                  className={`h-2 rounded-full ${
                                    planet.confidence > 90 ? 'bg-green-500' : 
                                    planet.confidence > 70 ? 'bg-yellow-500' : 'bg-red-500'
                                  }`}
                                  style={{width: `${planet.confidence}%`}}
                                ></div>
                              </div>
                            </div>
                            <span className="text-sm text-gray-900">
                              {planet.confidence}%
                            </span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>
          )}

        </motion.div>
      </div>
    </div>
  );
};

export default ResultsSection;