"""
🌟 HABITABILITY ANALYSIS ENGINE 🌟
Professional-grade habitability assessment for exoplanets
Ported from frontend HabitabilityAnalysis.jsx with enhanced calculations

This module provides comprehensive habitability analysis including:
- Physics-based orbital mechanics (Kepler's laws)
- Stellar luminosity and temperature calculations 
- Habitable zone determination
- Earth Similarity Index (ESI)
- Gas giant detection and classification
- Professional habitability scoring
"""

import math
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class HabitabilityClass(Enum):
    """Professional habitability classification system"""
    POTENTIALLY_HABITABLE = "Potentially Habitable"
    HABITABLE_ZONE = "Habitable Zone" 
    ROCKY_PLANET = "Rocky Planet"
    LARGE_PLANET = "Large Planet"
    GAS_GIANT = "Gas Giant"
    UNKNOWN = "Unknown"


@dataclass
class StellarParameters:
    """Stellar catalog parameters from TIC/Gaia/KIC sources"""
    mass: float              # Solar masses
    radius: float            # Solar radii
    temperature: float       # Kelvin
    luminosity: float        # Solar luminosities
    distance: float          # parsecs
    source: str             # TIC/Gaia/KIC/solar_default


@dataclass
class PlanetParameters:
    """Detected planet parameters"""
    period: float           # Days
    radius: float           # Earth radii
    gas_giant_detected: bool = False
    gas_giant_confidence: float = 0.0
    gas_giant_type: str = ""
    gas_giant_jupiter_radii: float = 0.0


@dataclass
class HabitabilityResult:
    """Complete habitability analysis results"""
    # Orbital mechanics
    semi_major_axis_au: float
    equilibrium_temperature: float
    
    # Habitable zone
    habitable_zone_inner: float
    habitable_zone_outer: float
    in_habitable_zone: bool
    
    # Planet characteristics
    is_rocky_size: bool
    temperature_range_suitable: bool
    
    # Similarity indices
    earth_similarity_index: float
    radius_esi: float
    temperature_esi: float
    flux_esi: float
    
    # Classification
    habitability_class: HabitabilityClass
    habitability_score: float
    habitability_description: str
    
    # Raw parameters (for reporting)
    stellar_params: StellarParameters
    planet_params: PlanetParameters


class HabitabilityAnalyzer:
    """
    Professional-grade habitability analysis engine
    
    Based on established exoplanet science and optimized for real-time analysis
    """
    
    # Physical constants
    G = 6.67430e-11          # Gravitational constant (m³ kg⁻¹ s⁻¹)
    M_SUN = 1.989e30         # Solar mass (kg)
    R_SUN = 6.96e8           # Solar radius (m)
    L_SUN = 3.828e26         # Solar luminosity (watts)
    AU = 1.496e11            # Astronomical unit (m)
    SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann constant (W m⁻² K⁻⁴)
    
    def __init__(self):
        """Initialize habitability analyzer"""
        pass
    
    def safe_number(self, value: Any, fallback: float) -> float:
        """Safely convert value to float with fallback"""
        try:
            num = float(value)
            return fallback if (math.isnan(num) or not math.isfinite(num)) else num
        except (ValueError, TypeError):
            return fallback
    
    def extract_stellar_parameters(self, results: Dict[str, Any]) -> StellarParameters:
        """Extract stellar parameters from backend results with safety checks"""
        return StellarParameters(
            mass=self.safe_number(results.get('stellar_mass'), 1.0),
            radius=self.safe_number(results.get('stellar_radius'), 1.0),
            temperature=self.safe_number(results.get('stellar_temperature'), 5778),
            luminosity=self.safe_number(results.get('stellar_luminosity'), 1.0),
            distance=self.safe_number(results.get('stellar_distance'), 100.0),
            source=results.get('catalog_source', 'unknown')
        )
    
    def extract_planet_parameters(self, results: Dict[str, Any]) -> PlanetParameters:
        """Extract planet parameters from backend results"""
        signal_analysis = results.get('signal_analysis', {})
        
        return PlanetParameters(
            period=self.safe_number(results.get('koi_period'), 365.0),
            radius=self.safe_number(results.get('koi_prad'), 1.0),
            gas_giant_detected=signal_analysis.get('gas_giant_detected', False),
            gas_giant_confidence=self.safe_number(signal_analysis.get('gas_giant_confidence'), 0.0),
            gas_giant_type=signal_analysis.get('gas_giant_type', ''),
            gas_giant_jupiter_radii=self.safe_number(signal_analysis.get('gas_giant_jupiter_radii'), 0.0)
        )
    
    def calculate_orbital_mechanics(self, stellar_mass: float, period_days: float) -> float:
        """
        Calculate semi-major axis using Kepler's Third Law
        a³ = (GM/4π²) * P²
        
        Args:
            stellar_mass: Star mass in solar masses
            period_days: Orbital period in days
            
        Returns:
            Semi-major axis in AU
        """
        stellar_mass_kg = stellar_mass * self.M_SUN
        period_seconds = period_days * 24 * 3600
        
        # Kepler's Third Law
        semi_major_axis_m = ((self.G * stellar_mass_kg * period_seconds**2) / (4 * math.pi**2))**(1/3)
        semi_major_axis_au = semi_major_axis_m / self.AU
        
        return semi_major_axis_au
    
    def calculate_equilibrium_temperature(self, stellar_luminosity: float, 
                                        semi_major_axis_au: float, 
                                        albedo: float = 0.3) -> float:
        """
        Calculate planet equilibrium temperature using Stefan-Boltzmann Law
        T_eq = [(L_star / 16πσa²)]^(1/4) * (1-A)^(1/4)
        
        Args:
            stellar_luminosity: Stellar luminosity in solar units
            semi_major_axis_au: Orbital distance in AU
            albedo: Bond albedo (default 0.3)
            
        Returns:
            Equilibrium temperature in Kelvin
        """
        stellar_luminosity_watts = stellar_luminosity * self.L_SUN
        semi_major_axis_m = semi_major_axis_au * self.AU
        
        # Calculate flux at planet
        flux_at_planet = stellar_luminosity_watts / (4 * math.pi * semi_major_axis_m**2)
        
        # Stefan-Boltzmann equation for equilibrium temperature
        equilibrium_temp = ((flux_at_planet * (1 - albedo)) / (4 * self.SIGMA_SB))**(1/4)
        
        return equilibrium_temp
    
    def calculate_habitable_zone(self, stellar_luminosity: float) -> Tuple[float, float]:
        """
        Calculate conservative habitable zone boundaries
        
        Args:
            stellar_luminosity: Stellar luminosity in solar units
            
        Returns:
            Tuple of (inner_edge_au, outer_edge_au)
        """
        # Conservative habitable zone (Kopparapu et al. 2013)
        inner_edge = math.sqrt(stellar_luminosity / 1.1)   # Runaway greenhouse
        outer_edge = math.sqrt(stellar_luminosity / 0.53)  # Maximum greenhouse
        
        return inner_edge, outer_edge
    
    def calculate_earth_similarity_index(self, planet_radius: float, 
                                       equilibrium_temp: float,
                                       flux_at_planet: float) -> Dict[str, float]:
        """
        Calculate Earth Similarity Index (ESI) and component indices
        
        Args:
            planet_radius: Planet radius in Earth radii
            equilibrium_temp: Equilibrium temperature in Kelvin
            flux_at_planet: Stellar flux at planet in W/m²
            
        Returns:
            Dictionary with ESI components and total ESI
        """
        # Earth reference values
        earth_temp = 288  # K
        earth_flux = 1361  # W/m² (solar constant)
        
        # Component ESI calculations
        radius_esi = 1 - abs((planet_radius - 1.0) / (planet_radius + 1.0))
        temp_esi = 1 - abs((equilibrium_temp - earth_temp) / (equilibrium_temp + earth_temp))
        flux_esi = 1 - abs((flux_at_planet/earth_flux - 1.0) / (flux_at_planet/earth_flux + 1.0))
        
        # Combined ESI (geometric mean)
        total_esi = (radius_esi * temp_esi * flux_esi)**(1/3)
        
        return {
            'radius_esi': radius_esi,
            'temperature_esi': temp_esi,
            'flux_esi': flux_esi,
            'total_esi': total_esi
        }
    
    def classify_habitability(self, stellar_params: StellarParameters,
                            planet_params: PlanetParameters,
                            semi_major_axis_au: float,
                            equilibrium_temp: float,
                            habitable_zone_inner: float,
                            habitable_zone_outer: float,
                            esi: float) -> Tuple[HabitabilityClass, float, str]:
        """
        Classify planet habitability based on multiple criteria
        
        Returns:
            Tuple of (classification, score, description)
        """
        # Basic checks
        in_habitable_zone = habitable_zone_inner <= semi_major_axis_au <= habitable_zone_outer
        is_rocky_size = planet_params.radius <= 2.0  # Earth-like to super-Earth
        temperature_suitable = 200 <= equilibrium_temp <= 320  # Liquid water range
        
        # Gas giant check
        if planet_params.gas_giant_detected:
            return (
                HabitabilityClass.GAS_GIANT,
                0.1,  # Low habitability for surface life
                f"{planet_params.gas_giant_type} - Not habitable for surface life, but may have habitable moons."
            )
        
        # Calculate habitability score (0-1 scale)
        score = 0.0
        
        # Zone score (40% weight)
        if in_habitable_zone:
            zone_score = 1.0 - abs(semi_major_axis_au - (habitable_zone_inner + habitable_zone_outer)/2) / ((habitable_zone_outer - habitable_zone_inner)/2)
            score += 0.4 * max(0, zone_score)
        
        # Size score (20% weight)
        if is_rocky_size:
            size_score = 1.0 - abs(planet_params.radius - 1.0) / 3.0  # Penalty for deviation from Earth size
            score += 0.2 * max(0, size_score)
        
        # Temperature score (25% weight)
        if temperature_suitable:
            temp_score = 1.0 - abs(equilibrium_temp - 288) / 200  # 288K = Earth average
            score += 0.25 * max(0, temp_score)
        
        # ESI contribution (15% weight)
        score += 0.15 * esi
        
        # Classification logic
        if in_habitable_zone and is_rocky_size and temperature_suitable:
            return (
                HabitabilityClass.POTENTIALLY_HABITABLE,
                min(1.0, score),
                "Located in the habitable zone with Earth-like characteristics."
            )
        elif in_habitable_zone and is_rocky_size:
            return (
                HabitabilityClass.HABITABLE_ZONE,
                min(0.8, score),
                "In the habitable zone but may have extreme temperatures."
            )
        elif is_rocky_size:
            return (
                HabitabilityClass.ROCKY_PLANET,
                min(0.6, score),
                "Rocky composition but outside the habitable zone."
            )
        else:
            return (
                HabitabilityClass.LARGE_PLANET,
                min(0.4, score),
                "Too large for surface habitability."
            )
    
    def analyze_habitability(self, results: Dict[str, Any]) -> HabitabilityResult:
        """
        Complete habitability analysis for detected exoplanet
        
        Args:
            results: Backend analysis results containing stellar and planet parameters
            
        Returns:
            Complete HabitabilityResult with all calculations and classifications
        """
        # Extract parameters
        stellar_params = self.extract_stellar_parameters(results)
        planet_params = self.extract_planet_parameters(results)
        
        # Orbital mechanics
        semi_major_axis_au = self.calculate_orbital_mechanics(stellar_params.mass, planet_params.period)
        equilibrium_temp = self.calculate_equilibrium_temperature(stellar_params.luminosity, semi_major_axis_au)
        
        # Habitable zone
        hz_inner, hz_outer = self.calculate_habitable_zone(stellar_params.luminosity)
        in_habitable_zone = hz_inner <= semi_major_axis_au <= hz_outer
        
        # Planet characteristics
        is_rocky_size = planet_params.radius <= 2.0
        temperature_suitable = 200 <= equilibrium_temp <= 320
        
        # Earth Similarity Index
        stellar_luminosity_watts = stellar_params.luminosity * self.L_SUN
        semi_major_axis_m = semi_major_axis_au * self.AU
        flux_at_planet = stellar_luminosity_watts / (4 * math.pi * semi_major_axis_m**2)
        
        esi_components = self.calculate_earth_similarity_index(
            planet_params.radius, equilibrium_temp, flux_at_planet
        )
        
        # Classification
        hab_class, hab_score, hab_desc = self.classify_habitability(
            stellar_params, planet_params, semi_major_axis_au, equilibrium_temp,
            hz_inner, hz_outer, esi_components['total_esi']
        )
        
        return HabitabilityResult(
            # Orbital mechanics
            semi_major_axis_au=semi_major_axis_au,
            equilibrium_temperature=equilibrium_temp,
            
            # Habitable zone
            habitable_zone_inner=hz_inner,
            habitable_zone_outer=hz_outer,
            in_habitable_zone=in_habitable_zone,
            
            # Planet characteristics
            is_rocky_size=is_rocky_size,
            temperature_range_suitable=temperature_suitable,
            
            # Similarity indices
            earth_similarity_index=esi_components['total_esi'],
            radius_esi=esi_components['radius_esi'],
            temperature_esi=esi_components['temperature_esi'],
            flux_esi=esi_components['flux_esi'],
            
            # Classification
            habitability_class=hab_class,
            habitability_score=hab_score,
            habitability_description=hab_desc,
            
            # Raw parameters
            stellar_params=stellar_params,
            planet_params=planet_params
        )
    
    def get_catalog_info(self, source: str) -> Dict[str, str]:
        """Get information about stellar catalog source"""
        catalog_info = {
            'TIC': {
                'name': 'TESS Input Catalog',
                'badge': '🛰️ TIC',
                'description': 'High-precision stellar parameters from the TESS space telescope mission. Professional-grade data for exoplanet host stars.'
            },
            'Gaia': {
                'name': 'Gaia Data Release 3',
                'badge': '🌌 Gaia DR3',
                'description': 'Ultra-precise stellar parameters from ESA\'s Gaia space observatory. Provides accurate distances, temperatures, and stellar properties.'
            },
            'KIC': {
                'name': 'Kepler Input Catalog',
                'badge': '🔭 KIC',
                'description': 'Stellar parameters from the original Kepler space telescope mission. Proven data source for thousands of exoplanet discoveries.'
            },
            'solar_default': {
                'name': 'Solar Default Values',
                'badge': '☀️ Solar',
                'description': 'Standard solar parameters used when catalog data is unavailable. Still provides accurate physics calculations.'
            }
        }
        
        return catalog_info.get(source, {
            'name': 'Unknown Source',
            'badge': '❓ Unknown',
            'description': 'Stellar parameters from an unknown or mixed source. Physics calculations remain valid.'
        })


# Convenience function for easy import
def analyze_habitability(results: Dict[str, Any]) -> HabitabilityResult:
    """
    Convenience function to perform complete habitability analysis
    
    Args:
        results: Backend analysis results
        
    Returns:
        Complete habitability analysis results
    """
    analyzer = HabitabilityAnalyzer()
    return analyzer.analyze_habitability(results)


if __name__ == "__main__":
    # Test the habitability analyzer
    test_results = {
        'stellar_mass': 1.0,
        'stellar_radius': 1.0,
        'stellar_temperature': 5778,
        'stellar_luminosity': 1.0,
        'stellar_distance': 100.0,
        'catalog_source': 'TIC',
        'koi_period': 365.25,
        'koi_prad': 1.0,
        'signal_analysis': {
            'gas_giant_detected': False,
            'gas_giant_confidence': 0.0,
            'gas_giant_type': '',
            'gas_giant_jupiter_radii': 0.0
        }
    }
    
    result = analyze_habitability(test_results)
    print(f"🌟 Habitability Analysis Results:")
    print(f"   Classification: {result.habitability_class.value}")
    print(f"   Score: {result.habitability_score:.3f}")
    print(f"   ESI: {result.earth_similarity_index:.3f}")
    print(f"   In Habitable Zone: {result.in_habitable_zone}")
    print(f"   Equilibrium Temperature: {result.equilibrium_temperature:.1f} K")