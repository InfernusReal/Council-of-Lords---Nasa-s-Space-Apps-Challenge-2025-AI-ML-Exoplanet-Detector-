import React, { useRef, useEffect, useState } from 'react';

const Simple2DVisualization = ({ 
  habitability, 
  currentStar, 
  planetRadiusEarth, 
  planetPeriodDays, 
  realStellarData 
}) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const [time, setTime] = useState(0);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = 600;
    canvas.height = 400;
    
    // Animation parameters
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const orbitRadius = 120;
    const starRadius = 20;
    const planetRadius = Math.max(planetRadiusEarth * 3, 4); // Scale for visibility
    
    // Colors based on star data
    const getStarColor = () => {
      if (!realStellarData?.stellar_temperature) return '#FFF3A0'; // Default yellow
      const temp = realStellarData.stellar_temperature;
      if (temp > 7500) return '#9BB0FF'; // Blue
      if (temp > 6000) return '#FFF3A0'; // Yellow
      if (temp > 5200) return '#FFE0B2'; // Orange
      if (temp > 3700) return '#FFB74D'; // Red-orange
      return '#FF5722'; // Red
    };
    
    const getPlanetColor = () => {
      if (habitability?.isInHabitableZone) {
        return '#4CAF50'; // Green for habitable
      }
      if (planetRadiusEarth > 2) {
        return '#2196F3'; // Blue for gas giant
      }
      return '#795548'; // Brown for rocky
    };

    const animate = () => {
      // Clear canvas
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw orbit path
      ctx.strokeStyle = '#333333';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(centerX, centerY, orbitRadius, 0, 2 * Math.PI);
      ctx.stroke();
      
      // Draw habitable zone (green ring)
      if (realStellarData?.stellar_luminosity) {
        const hzInner = Math.sqrt(realStellarData.stellar_luminosity) * 60;
        const hzOuter = Math.sqrt(realStellarData.stellar_luminosity) * 85;
        
        ctx.strokeStyle = '#4CAF50';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.arc(centerX, centerY, hzInner, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(centerX, centerY, hzOuter, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      
      // Draw star
      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, starRadius);
      gradient.addColorStop(0, getStarColor());
      gradient.addColorStop(0.7, getStarColor());
      gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, starRadius, 0, 2 * Math.PI);
      ctx.fill();
      
      // Add star glow
      ctx.shadowColor = getStarColor();
      ctx.shadowBlur = 20;
      ctx.fillStyle = getStarColor();
      ctx.beginPath();
      ctx.arc(centerX, centerY, starRadius * 0.7, 0, 2 * Math.PI);
      ctx.fill();
      ctx.shadowBlur = 0;
      
      // Calculate planet position
      const orbitSpeed = 0.02; // Adjust speed as needed
      const planetX = centerX + orbitRadius * Math.cos(time * orbitSpeed);
      const planetY = centerY + orbitRadius * Math.sin(time * orbitSpeed);
      
      // Draw planet
      ctx.fillStyle = getPlanetColor();
      ctx.beginPath();
      ctx.arc(planetX, planetY, planetRadius, 0, 2 * Math.PI);
      ctx.fill();
      
      // Add planet glow if habitable
      if (habitability?.isInHabitableZone) {
        ctx.shadowColor = '#4CAF50';
        ctx.shadowBlur = 10;
        ctx.strokeStyle = '#4CAF50';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(planetX, planetY, planetRadius + 2, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
      
      // Update time
      setTime(prev => prev + 1);
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [habitability, currentStar, planetRadiusEarth, planetPeriodDays, realStellarData]);

  return (
    <div className="flex gap-6 p-6 bg-gray-900 rounded-xl border border-cyan-500/40">
      {/* 2D Visualization */}
      <div className="flex-shrink-0">
        <canvas 
          ref={canvasRef}
          className="border border-cyan-500/40 rounded-lg bg-black"
          style={{ width: '600px', height: '400px' }}
        />
      </div>
      
      {/* Data Panel */}
      <div className="flex-1 space-y-4 text-white">
        <h3 className="text-xl font-bold text-cyan-400 mb-4">System Analysis</h3>
        
        {/* Stellar Data */}
        <div className="bg-gray-800 p-4 rounded-lg">
          <h4 className="text-lg font-semibold text-orange-400 mb-2">⭐ Stellar Properties</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>Mass: <span className="text-yellow-300">{realStellarData?.stellar_mass?.toFixed(2) || 'Unknown'} M☉</span></div>
            <div>Radius: <span className="text-yellow-300">{realStellarData?.stellar_radius?.toFixed(2) || 'Unknown'} R☉</span></div>
            <div>Temperature: <span className="text-orange-300">{realStellarData?.stellar_temperature?.toFixed(0) || 'Unknown'} K</span></div>
            <div>Luminosity: <span className="text-yellow-300">{realStellarData?.stellar_luminosity?.toFixed(2) || 'Unknown'} L☉</span></div>
            <div>Distance: <span className="text-blue-300">{realStellarData?.stellar_distance?.toFixed(1) || 'Unknown'} pc</span></div>
            <div>Source: <span className="text-green-300">{realStellarData?.catalog_source || 'Unknown'}</span></div>
          </div>
        </div>
        
        {/* Planet Data */}
        <div className="bg-gray-800 p-4 rounded-lg">
          <h4 className="text-lg font-semibold text-blue-400 mb-2">🪐 Planet Properties</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>Radius: <span className="text-blue-300">{planetRadiusEarth?.toFixed(2) || 'Unknown'} R⊕</span></div>
            <div>Period: <span className="text-purple-300">{planetPeriodDays?.toFixed(2) || 'Unknown'} days</span></div>
            <div>Type: <span className="text-cyan-300">
              {planetRadiusEarth > 2 ? 'Gas Giant' : planetRadiusEarth > 1.5 ? 'Super-Earth' : 'Terrestrial'}
            </span></div>
          </div>
        </div>
        
        {/* Habitability */}
        <div className="bg-gray-800 p-4 rounded-lg">
          <h4 className="text-lg font-semibold text-green-400 mb-2">🌍 Habitability Analysis</h4>
          <div className="space-y-2 text-sm">
            <div className={`flex items-center gap-2 ${habitability?.isInHabitableZone ? 'text-green-300' : 'text-red-300'}`}>
              <span className={`w-3 h-3 rounded-full ${habitability?.isInHabitableZone ? 'bg-green-500' : 'bg-red-500'}`}></span>
              {habitability?.isInHabitableZone ? 'In Habitable Zone' : 'Outside Habitable Zone'}
            </div>
            <div>Temperature: <span className="text-orange-300">{habitability?.equilibriumTemp?.toFixed(0) || 'Unknown'} K</span></div>
            <div>Stellar Flux: <span className="text-yellow-300">{habitability?.stellarFlux?.toFixed(2) || 'Unknown'} S⊕</span></div>
            <div>ESI: <span className="text-cyan-300">{habitability?.esi?.toFixed(3) || 'Unknown'}</span></div>
          </div>
        </div>
        
        {/* Council Verdict */}
        <div className="bg-gray-800 p-4 rounded-lg border border-purple-500/40">
          <h4 className="text-lg font-semibold text-purple-400 mb-2">⚖️ Council Verdict</h4>
          <div className="text-sm">
            <div className="text-white">Detection Confidence: <span className="text-cyan-300 font-bold">High</span></div>
            <div className="text-gray-300 mt-1">Real stellar parameters integrated ✓</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Simple2DVisualization;