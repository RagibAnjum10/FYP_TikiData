import React from 'react';
import { useNavigate } from 'react-router-dom';

function HomePage() {
  const navigate = useNavigate();

  return (
    <div style={{
      height: '100vh',
      width: '100%',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Barcelona-themed background */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        background: 'linear-gradient(135deg, #004D98 0%, #A50044 100%)',
        zIndex: -1
      }}></div>
      
      {/* Content container */}
      <div style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        width: '100%',
        maxWidth: '900px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        textAlign: 'center',
        padding: '0 20px',
        color: 'white'
      }}>
        {/* Logo */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          marginBottom: '20px'
        }}>
          <div style={{
            width: '50px',
            height: '50px',
            backgroundColor: 'white',
            borderRadius: '50%',
            marginRight: '15px',
            position: 'relative'
          }}>
            <div style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              width: '80%',
              height: '80%',
              background: '#232323',
              borderRadius: '50%',
              clipPath: 'polygon(30% 0%, 70% 0%, 100% 30%, 100% 70%, 70% 100%, 30% 100%, 0% 70%, 0% 30%)'
            }}></div>
          </div>
          <h1 style={{
            color: 'white',
            fontSize: '3.5rem',
            margin: 0,
            fontWeight: 700
          }}>
            TikiData
          </h1>
        </div>
        
        {/* Tagline */}
        <div style={{ marginBottom: '40px' }}>
          <p style={{
            fontSize: '1.5rem',
            opacity: 0.9
          }}>
            Intelligent football predictions powered by data science
          </p>
        </div>
        
        {/* Features */}
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '30px',
          marginBottom: '50px',
          flexWrap: 'wrap'
        }}>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            background: 'rgba(0,0,0,0.2)',
            padding: '20px',
            borderRadius: '10px',
            width: '180px'
          }}>
            <span style={{ fontSize: '2.5rem', marginBottom: '15px' }}>🏆</span>
            <span>Premier League Focus</span>
          </div>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            background: 'rgba(0,0,0,0.2)',
            padding: '20px',
            borderRadius: '10px',
            width: '180px'
          }}>
            <span style={{ fontSize: '2.5rem', marginBottom: '15px' }}>📊</span>
            <span>Advanced Analytics</span>
          </div>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            background: 'rgba(0,0,0,0.2)',
            padding: '20px',
            borderRadius: '10px',
            width: '180px'
          }}>
            <span style={{ fontSize: '2.5rem', marginBottom: '15px' }}>🎯</span>
            <span>Accurate Predictions</span>
          </div>
        </div>
        
        {/* Action Button */}
        <div>
          <button 
            style={{
              padding: '12px 30px',
              fontSize: '1.2rem',
              fontWeight: 600,
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              backgroundColor: '#ff9500',
              color: 'white'
            }}
            onClick={() => navigate('/predict')}
          >
            Make Prediction
          </button>
        </div>
      </div>
      
      {/* Navigation buttons in top right */}
      <div className="nav-buttons">
        <button
          className="nav-btn"
          onClick={() => navigate('/')}
        >
          Home
        </button>
        <button
          className="nav-btn nav-btn--primary"
          onClick={() => navigate('/predict')}
        >
          Prediction
        </button>
      </div>
    </div>
  );
}

export default HomePage;