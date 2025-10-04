import React, { useState, useEffect } from 'react';
import HomePage from './pages/homepage';
import TelescopePage from './pages/telescope';

function App() {
  const [currentPage, setCurrentPage] = useState('homepage');

  useEffect(() => {
    // Handle hash-based routing
    const handleHashChange = () => {
      const hash = window.location.hash.slice(1); // Remove the #
      if (hash === 'telescope') {
        setCurrentPage('telescope');
      } else {
        setCurrentPage('homepage');
      }
    };

    // Set initial page based on current hash
    handleHashChange();

    // Listen for hash changes
    window.addEventListener('hashchange', handleHashChange);
    
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  return (
    <div>
      {currentPage === 'homepage' && <HomePage />}
      {currentPage === 'telescope' && <TelescopePage />}
    </div>
  );
}

export default App;
