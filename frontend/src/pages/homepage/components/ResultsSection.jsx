import React, { useState } from 'react';
import { motion } from 'framer-motion';
import HabitabilityAnalysis from './HabitabilityAnalysis';

const ResultsSection = ({ results = null }) => {
  const [activeTab, setActiveTab] = useState('overview');

  if (!results) {
    return (
      <div className="bg-gray-50 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Ready to Unleash the Council?
            </h2>
            <p className="text-gray-600 text-lg">
              Upload your dataset to see the Council of Lords' brutal reality analysis
            </p>
          </motion.div>
        </div>
      </div>
    );
  }

  const isExoplanet = results.councilVerdict === 'EXOPLANET';
  const confidenceColor = results.confidence > 80 ? 'text-green-600' : 
                          results.confidence > 60 ? 'text-yellow-600' : 'text-red-600';

  return (
    <div className="bg-gray-50 py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div 
          className="text-center mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            🏛️ Council Verdict Delivered
          </h2>
          <div className={`text-6xl font-bold mb-4 ${isExoplanet ? 'text-green-600' : 'text-red-600'}`}>
            {results.councilVerdict}
          </div>
          <p className="text-xl text-gray-600">
            {results.message}
          </p>
        </motion.div>

        {/* Stats Overview */}
        <motion.div 
          className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <div className={`bg-white rounded-xl p-6 text-center shadow-lg border-l-4 ${
            isExoplanet ? 'border-green-500' : 'border-red-500'
          }`}>
            <div className={`text-3xl font-bold mb-2 ${confidenceColor}`}>
              {results.confidence}%
            </div>
            <div className="text-gray-600">Council Confidence</div>
          </div>
          
          <div className="bg-white rounded-xl p-6 text-center shadow-lg border-l-4 border-blue-500">
            <div className="text-3xl font-bold text-blue-600 mb-2">
              {results.analysisTime}
            </div>
            <div className="text-gray-600">Processing Time</div>
          </div>
          
          <div className="bg-white rounded-xl p-6 text-center shadow-lg border-l-4 border-purple-500">
            <div className="text-3xl font-bold text-purple-600 mb-2">
              {Object.keys(results.individualVotes || {}).length}
            </div>
            <div className="text-gray-600">Specialists Voted</div>
          </div>
          
          <div className="bg-white rounded-xl p-6 text-center shadow-lg border-l-4 border-orange-500">
            <div className="text-3xl font-bold text-orange-600 mb-2">
              {results.redFlags?.length || 0}
            </div>
            <div className="text-gray-600">Red Flags</div>
          </div>
        </motion.div>

        {/* Detailed Results */}
        <motion.div 
          className="bg-white rounded-2xl shadow-xl overflow-hidden"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          {/* Tabs */}
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'overview', label: 'Council Verdict' },
                { id: 'habitability', label: '� Habitability Analysis' },
                { id: 'specialists', label: 'Specialist Votes' },
                { id: 'logs', label: 'Pipeline Logs' },
                { id: 'redflags', label: 'Red Flags' }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>

          <div className="p-6">
            {activeTab === 'overview' && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
              >
                <h3 className="text-2xl font-semibold text-gray-900 mb-6">
                  🏛️ Council Analysis Summary
                </h3>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div>
                    <h4 className="text-lg font-medium text-gray-900 mb-4">
                      Detection Parameters
                    </h4>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Planet Radius:</span>
                        <span className="font-medium">{results.discoveries?.[0]?.radius || 'Unknown'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Orbital Period:</span>
                        <span className="font-medium">{results.discoveries?.[0]?.period || 'Unknown'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Star Type:</span>
                        <span className="font-medium">{results.discoveries?.[0]?.type || 'Unknown'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Data Quality:</span>
                        <span className="font-medium">
                          {results.advanced_detection?.data_quality_score ? 
                            `${(results.advanced_detection.data_quality_score * 100).toFixed(1)}%` : 'Unknown'}
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-lg font-medium text-gray-900 mb-4">
                      Advanced Detection Flags
                    </h4>
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <div className={`w-3 h-3 rounded-full ${
                          results.advanced_detection?.v_shape_detected ? 'bg-red-500' : 'bg-green-500'
                        }`}></div>
                        <span className="text-sm">V-shape Eclipse: {results.advanced_detection?.v_shape_detected ? 'Detected' : 'Clear'}</span>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <div className={`w-3 h-3 rounded-full ${
                          results.advanced_detection?.instrumental_correlation ? 'bg-red-500' : 'bg-green-500'
                        }`}></div>
                        <span className="text-sm">Instrumental Issues: {results.advanced_detection?.instrumental_correlation ? 'Detected' : 'Clear'}</span>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <div className={`w-3 h-3 rounded-full ${
                          results.advanced_detection?.secondary_eclipse ? 'bg-yellow-500' : 'bg-green-500'
                        }`}></div>
                        <span className="text-sm">Secondary Eclipse: {results.advanced_detection?.secondary_eclipse ? 'Detected' : 'Clear'}</span>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <div className={`w-3 h-3 rounded-full ${
                          results.advanced_detection?.stellar_activity ? 'bg-orange-500' : 'bg-green-500'
                        }`}></div>
                        <span className="text-sm">Stellar Activity: {results.advanced_detection?.stellar_activity ? 'Detected' : 'Clear'}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'habitability' && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
              >
                <h3 className="text-2xl font-semibold text-gray-900 mb-6">
                   Exoplanet Habitability & Stellar System Analysis
                </h3>
                
                {isExoplanet ? (
                  <HabitabilityAnalysis results={results} />
                ) : (
                  <div className="text-center py-12">
                    <div className="text-6xl mb-4">🚫</div>
                    <h4 className="text-xl font-medium text-gray-900 mb-2">
                      No Exoplanet Detected
                    </h4>
                    <p className="text-gray-600">
                      The Council determined this signal is not a genuine exoplanet. Habitability analysis is only available for confirmed detections.
                    </p>
                  </div>
                )}
              </motion.div>
            )}

            {activeTab === 'specialists' && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
              >
                <h3 className="text-2xl font-semibold text-gray-900 mb-6">
                  🤖 Individual Specialist Votes
                </h3>
                
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          AI Specialist
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Vote
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Prediction Score
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Weight
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Influence
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {Object.entries(results.individualVotes || {}).map(([specialist, vote]) => {
                        const predictionPercent = Math.round(vote.prediction * 100);
                        const voteColor = vote.vote === 'EXOPLANET' ? 'bg-green-100 text-green-800' : 
                                         vote.vote === 'STAR' ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800';
                        
                        return (
                          <tr key={specialist} className="hover:bg-gray-50">
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="flex items-center">
                                <div className="text-sm font-medium text-gray-900">
                                  {specialist.replace(/_/g, ' ')}
                                </div>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${voteColor}`}>
                                {vote.vote}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="flex items-center">
                                <div className="flex-1 mr-2">
                                  <div className="w-full bg-gray-200 rounded-full h-2">
                                    <div 
                                      className={`h-2 rounded-full ${
                                        predictionPercent > 50 ? 'bg-green-500' : 'bg-red-500'
                                      }`}
                                      style={{width: `${predictionPercent}%`}}
                                    ></div>
                                  </div>
                                </div>
                                <span className="text-sm text-gray-900">
                                  {predictionPercent}%
                                </span>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {vote.weight?.toFixed(2) || '1.00'}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="text-sm text-gray-900">
                                {((vote.prediction * vote.weight) * 100).toFixed(1)}%
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            )}

            {activeTab === 'logs' && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
              >
                <h3 className="text-2xl font-semibold text-gray-900 mb-6">
                  📜 Real-time Pipeline Logs
                </h3>
                
                <div className="bg-black rounded-lg p-4 max-h-96 overflow-y-auto">
                  <div className="font-mono text-sm space-y-1">
                    {results.pipelineLogs?.map((log, index) => (
                      <div key={index} className={`${
                        log.includes('🔥') ? 'text-red-400' :
                        log.includes('✅') ? 'text-green-400' :
                        log.includes('🚨') ? 'text-yellow-400' :
                        log.includes('📁') ? 'text-blue-400' :
                        log.includes('🔧') ? 'text-purple-400' :
                        'text-gray-300'
                      }`}>
                        <span className="text-gray-500">[{index.toString().padStart(3, '0')}]</span> {log}
                      </div>
                    )) || <div className="text-gray-500">No logs available</div>}
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'redflags' && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
              >
                <h3 className="text-2xl font-semibold text-gray-900 mb-6">
                  🚨 Red Flags & Warnings
                </h3>
                
                {results.redFlags?.length > 0 ? (
                  <div className="space-y-4">
                    {results.redFlags.map((flag, index) => (
                      <div key={index} className="bg-red-50 border border-red-200 rounded-lg p-4">
                        <div className="flex items-start space-x-3">
                          <div className="text-red-500 text-xl">⚠️</div>
                          <div>
                            <div className="text-red-800 font-medium">{flag}</div>
                            <div className="text-red-600 text-sm mt-1">
                              This flag indicates potential issues that may affect detection accuracy.
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                    
                    <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 mt-6">
                      <h4 className="text-orange-800 font-medium mb-2">🔍 What Red Flags Mean:</h4>
                      <ul className="text-orange-700 text-sm space-y-1">
                        <li>• <strong>V-shape Eclipse:</strong> May indicate a binary star system rather than an exoplanet</li>
                        <li>• <strong>Instrumental Correlation:</strong> Signal might be caused by instrument artifacts</li>
                        <li>• <strong>Secondary Eclipse:</strong> Additional signal that needs investigation</li>
                        <li>• <strong>Stellar Activity:</strong> Host star variability may affect measurements</li>
                        <li>• <strong>Poor Data Quality:</strong> Low signal-to-noise ratio or incomplete data</li>
                      </ul>
                    </div>
                  </div>
                ) : (
                  <div className="bg-green-50 border border-green-200 rounded-lg p-6 text-center">
                    <div className="text-green-600 text-4xl mb-2">✅</div>
                    <div className="text-green-800 font-medium mb-1">All Clear!</div>
                    <div className="text-green-600">No red flags detected in this analysis.</div>
                  </div>
                )}
              </motion.div>
            )}

            {/* Exoplanets Table */}
            {results?.discoveries && results.discoveries.length > 0 && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
              >
                <h3 className="text-2xl font-semibold text-gray-900 mb-6">
                  Discovered Exoplanets
                </h3>
                
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Planet Name
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Type
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Method
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Radius
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Period
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Confidence
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {(results?.discoveries || []).map((planet) => (
                        <tr key={planet.id} className="hover:bg-gray-50">
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm font-medium text-gray-900">
                              {planet.name}
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

          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default ResultsSection;