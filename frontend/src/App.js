import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Container, Row, Col, Card } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import FifaTeamSelection from './FifaTeamSelection';

// This is a standalone solution that combines both pages
// and doesn't depend on React Router
function App() {
  // State to determine which view to show
  const [currentView, setCurrentView] = useState('home');
  
  // States for the prediction feature
  const [teams, setTeams] = useState([]);
  const [home, setHome] = useState('');
  const [away, setAway] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Fixed API URL with /api prefix
  const API = 'http://localhost:8000/api';

  // Default teams in case API fails
  const DEFAULT_TEAMS = [
    "Manchester City", "Liverpool", "Arsenal", "Manchester United", 
    "Chelsea", "Tottenham", "Newcastle United", "Aston Villa", 
    "Brighton & Hove Albion", "West Ham United", "Crystal Palace", 
    "Wolverhampton", "Everton", "Leicester City", "Brentford", 
    "Fulham", "Bournemouth", "Nottingham Forest", "Leeds United", 
    "Southampton"
  ];

  // Load teams when we switch to prediction view
  useEffect(() => {
    if (currentView === 'predict') {
      loadTeams();
    }
  }, [currentView]); // eslint-disable-line react-hooks/exhaustive-deps

  // Function to load teams
  const loadTeams = () => {
    axios.get(`${API}/teams`)
      .then(res => {
        setTeams(res.data);
        if (res.data.length >= 2) {
          setHome(res.data[0]);
          setAway(res.data[1]);
        }
      })
      .catch(err => {
        console.log("Teams loading error:", err);
        setError('Cannot load teams from API. Using default list.');
        setTeams(DEFAULT_TEAMS);
        setHome(DEFAULT_TEAMS[0]);
        setAway(DEFAULT_TEAMS[1]);
      });
  };

  // Handle prediction submission
  const handleSubmit = (e) => {
    if (e) e.preventDefault(); // This is needed because the FIFA component passes the event object
    if (home === away) {
      setError('Teams must differ');
      return;
    }
    
    setLoading(true);
    setError('');
    
    axios.post(`${API}/predict`, { home_team: home, away_team: away })
      .then(res => {
        setPrediction(res.data);
        setError('');
      })
      .catch(err => {
        console.log("Prediction error:", err);
        setError('Prediction failed');
        
        // Optional fallback prediction
        if (home === "Liverpool" && away === "Southampton") {
          setPrediction({
            home_team: home,
            away_team: away,
            home_win_probability: 0.75,
            draw_probability: 0.15,
            away_win_probability: 0.10,
            prediction: "H",
            model_used: "Fallback Model",
            key_factors: {
              "note": "Using fallback Liverpool vs Southampton prediction (API unavailable)"
            }
          });
        }
      })
      .finally(() => setLoading(false));
  };

  // Render the Home Page
  if (currentView === 'home') {
    return (
      <div style={{
        height: '100vh',
        width: '100%',
        position: 'relative',
        overflow: 'hidden',
        fontFamily: '"Roboto", sans-serif'
      }}>
        {/* Animated Barcelona-themed background with gradient movement */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          background: 'linear-gradient(135deg, #004D98 0%, #A50044 100%)',
          zIndex: -1,
          animation: 'gradientMove 15s ease infinite'
        }}></div>
        
        {/* Animated floating particles */}
        <div className="particles"></div>
        
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
          color: 'white',
          animation: 'fadeIn 1.5s ease-out'
        }}>
          {/* Logo with animation */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            marginBottom: '20px',
            animation: 'slideDown 1s ease'
          }}>
            <div style={{
              width: '50px',
              height: '50px',
              backgroundColor: 'white',
              borderRadius: '50%',
              marginRight: '15px',
              position: 'relative',
              animation: 'spin 10s linear infinite'
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
              fontWeight: 700,
              background: 'linear-gradient(to right, #ffffff, #f8e287)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              animation: 'shimmer 3s infinite'
            }}>
              TikiData
            </h1>
          </div>
          
          {/* Tagline with typing animation */}
          <div style={{ 
            marginBottom: '40px',
            animation: 'fadeIn 2s ease'
          }}>
            <p className="typewriter" style={{
              fontSize: '1.5rem',
              opacity: 0.9
            }}>
              Intelligent football predictions powered by data science
            </p>
          </div>
          
          {/* Features with animations */}
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '30px',
            marginBottom: '50px',
            flexWrap: 'wrap'
          }}>
            {/* Premier League Focus - Pulsing animation */}
            <div 
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                background: 'rgba(0,0,0,0.2)',
                padding: '20px',
                borderRadius: '10px',
                width: '180px',
                animation: 'pulse 2s infinite',
                boxShadow: '0 0 10px rgba(255, 255, 255, 0.2)',
                transition: 'transform 0.3s, box-shadow 0.3s',
                animationDelay: '0s'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-10px)';
                e.currentTarget.style.boxShadow = '0 15px 20px rgba(0, 0, 0, 0.3)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 0 10px rgba(255, 255, 255, 0.2)';
              }}
            >
              <span style={{ 
                fontSize: '2.5rem', 
                marginBottom: '15px',
                display: 'inline-block',
                animation: 'bounce 2s infinite'
              }}>🏆</span>
              <span>Premier League Focus</span>
            </div>
            
            {/* Advanced Analytics - Floating animation */}
            <div 
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                background: 'rgba(0,0,0,0.2)',
                padding: '20px',
                borderRadius: '10px',
                width: '180px',
                animation: 'float 3s infinite',
                boxShadow: '0 0 10px rgba(255, 255, 255, 0.2)',
                transition: 'transform 0.3s, box-shadow 0.3s',
                animationDelay: '0.5s'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-10px)';
                e.currentTarget.style.boxShadow = '0 15px 20px rgba(0, 0, 0, 0.3)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 0 10px rgba(255, 255, 255, 0.2)';
              }}
            >
              <span style={{ 
                fontSize: '2.5rem', 
                marginBottom: '15px',
                display: 'inline-block',
                animation: 'rotate 3s infinite'
              }}>📊</span>
              <span>Advanced Analytics</span>
            </div>
            
            {/* Accurate Predictions - Glow animation */}
            <div 
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                background: 'rgba(0,0,0,0.2)',
                padding: '20px',
                borderRadius: '10px',
                width: '180px',
                animation: 'glow 2s infinite alternate',
                boxShadow: '0 0 10px rgba(255, 255, 255, 0.2)',
                transition: 'transform 0.3s, box-shadow 0.3s',
                animationDelay: '1s'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-10px)';
                e.currentTarget.style.boxShadow = '0 15px 20px rgba(0, 0, 0, 0.3)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 0 10px rgba(255, 255, 255, 0.2)';
              }}
            >
              <span style={{ 
                fontSize: '2.5rem', 
                marginBottom: '15px',
                display: 'inline-block',
                animation: 'shake 3s infinite'
              }}>🎯</span>
              <span>Accurate Predictions</span>
            </div>
          </div>
          
          {/* UPDATED: Modern Action Button */}
          <div style={{
            animation: 'fadeIn 3s ease'
          }}>
            <button 
              className="modern-action-button"
              onClick={() => setCurrentView('predict')}
            >
              <span className="button-content">Make Prediction</span>
              <span className="button-icon">→</span>
            </button>
          </div>
        </div>
        
        {/* UPDATED: Modern Navigation buttons in top right */}
        <div style={{
          position: 'absolute',
          top: '20px',
          right: '30px',
          display: 'flex',
          gap: '15px',
          zIndex: 10,
          animation: 'slideInRight 1s ease'
        }}>
          <button 
            className="modern-nav-button"
            onClick={() => setCurrentView('home')}
          >
            Home
          </button>
          <button 
            className="modern-nav-button modern-nav-button-primary"
            onClick={() => setCurrentView('predict')}
          >
            Prediction
          </button>
        </div>

        {/* Animations CSS */}
        <style jsx="true">{`
          /* Animation Keyframes */
          @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
          }
          
          @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
          }
          
          @keyframes glow {
            0% { box-shadow: 0 0 5px rgba(255, 149, 0, 0.3); }
            100% { box-shadow: 0 0 20px rgba(255, 149, 0, 0.7); }
          }
          
          @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-10px); }
            60% { transform: translateY(-5px); }
          }
          
          @keyframes rotate {
            0% { transform: rotate(0deg); }
            25% { transform: rotate(10deg); }
            75% { transform: rotate(-10deg); }
            100% { transform: rotate(0deg); }
          }
          
          @keyframes shake {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-2px); }
            20%, 40%, 60%, 80% { transform: translateX(2px); }
          }
          
          @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
          }
          
          @keyframes slideDown {
            0% { transform: translateY(-50px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
          }
          
          @keyframes slideInRight {
            0% { transform: translateX(50px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
          }
          
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          
          @keyframes shimmer {
            0% { background-position: -100% 0; }
            100% { background-position: 100% 0; }
          }
          
          @keyframes gradientMove {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
          }
          
          /* Typewriter animation */
          .typewriter {
            overflow: hidden;
            border-right: .15em solid orange;
            white-space: nowrap;
            margin: 0 auto;
            letter-spacing: .15em;
            animation: 
              typing 3.5s steps(40, end),
              blink-caret .75s step-end infinite;
          }
          
          @keyframes typing {
            from { width: 0 }
            to { width: 100% }
          }
          
          @keyframes blink-caret {
            from, to { border-color: transparent }
            50% { border-color: orange; }
          }
          
          /* NEW: Modern button styles */
          .modern-action-button {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            padding: 12px 30px;
            font-size: 1.1rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            border: none;
            border-radius: 12px;
            background: #ff9500;
            color: white;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            box-shadow: 0 6px 16px rgba(255, 149, 0, 0.4);
          }
          
          .modern-action-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(255, 149, 0, 0.5);
            background: #ff9500;
          }
          
          .modern-action-button:active {
            transform: translateY(1px);
          }
          
          .button-content {
            position: relative;
            z-index: 1;
            transition: transform 0.3s ease;
          }
          
          .button-icon {
            position: relative;
            z-index: 1;
            transition: transform 0.3s ease;
            display: inline-block;
          }
          
          .modern-action-button:hover .button-content {
            transform: translateX(-5px);
          }
          
          .modern-action-button:hover .button-icon {
            transform: translateX(3px);
          }
          
          .modern-action-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(120deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            transform: translateX(-100%);
            transition: all 0.6s;
          }
          
          .modern-action-button:hover::before {
            transform: translateX(100%);
          }
          
          /* Modern nav buttons */
          .modern-nav-button {
            padding: 8px 16px;
            font-size: 0.95rem;
            font-weight: 500;
            border: none;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(5px);
            color: white;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.1);
          }
          
          .modern-nav-button:hover {
            transform: translateY(-2px);
            background: rgba(255, 255, 255, 0.25);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          }
          
          .modern-nav-button-primary {
            background: rgba(255, 149, 0, 0.8);
            box-shadow: 0 2px 8px rgba(255, 149, 0, 0.3);
            border: 1px solid rgba(255, 149, 0, 0.3);
          }
          
          .modern-nav-button-primary:hover {
            background: rgba(255, 149, 0, 0.9);
            box-shadow: 0 4px 12px rgba(255, 149, 0, 0.4);
          }
          
          /* Particle animation */
          .particles {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
          }
          
          .particles:before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
              radial-gradient(circle, white 1px, transparent 1px),
              radial-gradient(circle, white 1px, transparent 1px);
            background-size: 40px 40px;
            background-position: 0 0, 20px 20px;
            animation: particlesDrift 20s linear infinite;
            opacity: 0.15;
          }
          
          @keyframes particlesDrift {
            0% { transform: translate(0, 0); }
            100% { transform: translate(-40px, 20px); }
          }
        `}</style>
      </div>
    );
  }

  // Otherwise, render the Prediction Page with Barcelona theme and animations
  return (
    <div className="prediction-page-container">
      {/* Barcelona-themed background with gradient movement */}
      <div className="prediction-bg-gradient"></div>
      
      {/* Animated floating particles */}
      <div className="particles prediction-particles"></div>
      
      <Container className="mt-5 prediction-container">
        <Row>
          <Col>
            <h1 className="text-center mb-4 animated-title">Football Match Prediction</h1>
            <Card className="animated-card">
              <Card.Body>
                {/* Replace the Form with FifaTeamSelection component */}
                <FifaTeamSelection 
                  teams={teams} 
                  home={home} 
                  away={away} 
                  setHome={setHome} 
                  setAway={setAway} 
                  loading={loading} 
                  handleSubmit={handleSubmit} 
                  error={error} 
                />
              </Card.Body>
            </Card>

            {prediction && (
              <Card className="mt-4 prediction-card result-card">
                <Card.Header className="text-center result-header">
                  <h3>Prediction Result</h3>
                  <h4>{prediction.home_team} vs {prediction.away_team}</h4>
                </Card.Header>
                <Card.Body>
                  <Row>
                    <Col className="text-center">
                      <h5>Home Win</h5>
                      <div className="probability-circle bg-success animated-circle">
                        {(prediction.home_win_probability * 100).toFixed(1)}%
                      </div>
                    </Col>
                    <Col className="text-center">
                      <h5>Draw</h5>
                      <div className="probability-circle bg-warning animated-circle">
                        {(prediction.draw_probability * 100).toFixed(1)}%
                      </div>
                    </Col>
                    <Col className="text-center">
                      <h5>Away Win</h5>
                      <div className="probability-circle bg-danger animated-circle">
                        {(prediction.away_win_probability * 100).toFixed(1)}%
                      </div>
                    </Col>
                  </Row>
                  <div className="mt-4 text-center">
                    <h5>Outcome: <span className="prediction-result animated-result">{
                      prediction.prediction === 'H' ? prediction.home_team :
                      prediction.prediction === 'A' ? prediction.away_team : 'Draw'
                    }</span></h5>
                  </div>
                  
                  {/* Key factors section */}
                  {prediction.key_factors && (
                    <div className="mt-4 animated-factors">
                      <h5>Key Factors:</h5>
                      <ul className="key-factors">
                        {Object.entries(prediction.key_factors).map(([key, value], index) => (
                          <li key={key} style={{animationDelay: `${index * 0.2}s`}} className="factor-item">
                            <strong>{key.replace(/_/g, ' ')}:</strong> {value}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </Card.Body>
              </Card>
            )}
            
            {/* UPDATED: Modern Back button */}
            <div className="text-center mt-4 mb-5">
              <button 
                className="modern-back-button"
                onClick={() => setCurrentView('home')}
              >
                <span className="button-icon-left">←</span>
                <span className="button-content">Back to Home</span>
              </button>
            </div>
          </Col>
        </Row>

        {/* Add CSS for prediction page animations */}
        <style jsx="true">{`
          /* Prediction page styles with Barcelona theme */
          .prediction-page-container {
            min-height: 100vh;
            width: 100%;
            position: relative;
            padding-bottom: 2rem;
            overflow-x: hidden;
          }
          
          .prediction-bg-gradient {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(0, 77, 152, 0.9) 0%, rgba(165, 0, 68, 0.9) 100%);
            z-index: -2;
            animation: gradientMove 15s ease infinite;
            background-size: 400% 400%;
          }
          
          .prediction-particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
          }
          
          .prediction-particles:before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
              radial-gradient(circle, white 1px, transparent 1px),
              radial-gradient(circle, white 1px, transparent 1px);
            background-size: 40px 40px;
            background-position: 0 0, 20px 20px;
            animation: particlesDrift 20s linear infinite;
            opacity: 0.15;
          }
          
          .prediction-container {
            position: relative;
            z-index: 1;
            animation: fadeIn 0.8s ease-out;
          }
          
          .animated-title {
            color: white;
            text-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
            animation: slideDown 0.8s ease;
          }
          
          .animated-card {
            animation: fadeIn 1s ease;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
            transition: box-shadow 0.3s ease, transform 0.3s ease;
            background-color: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(5px);
            border: none;
            border-radius: 12px;
          }
          
          .animated-card:hover {
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3);
            transform: translateY(-5px);
          }
          
          .result-card {
            background-color: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(5px);
            border-radius: 12px;
            overflow: hidden;
          }
          
          .result-header {
            background: linear-gradient(to right, #004D98, #A50044);
            color: white;
            border: none;
          }
          
          .probability-circle {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 1rem auto;
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
            animation: fadeIn 1s ease, pulse 2s infinite;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
          }
          
          .animated-result {
            animation: fadeIn 1.2s ease, glow 2s infinite alternate;
            font-weight: bold;
            color: #3498db;
            background-color: #eef7ff;
            padding: 5px 15px;
            border-radius: 20px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
          }
          
          .animated-factors {
            animation: fadeIn 1.5s ease;
          }
          
          .key-factors {
            list-style-type: none;
            padding-left: 0;
          }
          
          .key-factors li {
            padding: 10px;
            margin-bottom: 5px;
            background-color: rgba(248, 249, 250, 0.8);
            border-radius: 5px;
            animation: slideInFromRight 0.5s ease forwards;
            opacity: 0;
            transform: translateX(50px);
            transition: all 0.3s ease;
          }
          
          .key-factors li:hover {
            background-color: rgba(248, 249, 250, 1);
            transform: translateX(5px);
          }
          
          .key-factors li strong {
            color: #ff9500;
            text-transform: capitalize;
          }
          
          /* UPDATED: Modern back button */
          .modern-back-button {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            padding: 10px 24px;
            background-color: rgba(255, 255, 255, 0.15);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            font-weight: 500;
            backdrop-filter: blur(5px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            animation: fadeIn 2s ease;
          }
          
          .modern-back-button:hover {
            background-color: rgba(255, 255, 255, 0.25);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
          }
          
          .modern-back-button:active {
            transform: translateY(1px);
          }
          
          .button-icon-left {
            transition: transform 0.3s ease;
          }
          
          .modern-back-button:hover .button-icon-left {
            transform: translateX(-3px);
          }
          
          /* Adding glassmorphism effect for any submit buttons in FifaTeamSelection component */
          :global(.prediction-submit-btn) {
            background: rgba(255, 149, 0, 0.85) !important;
            backdrop-filter: blur(5px) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
            padding: 10px 22px !important;
            font-weight: 500 !important;
            letter-spacing: 0.5px !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            box-shadow: 0 6px 15px rgba(255, 149, 0, 0.3) !important;
            position: relative !important;
            overflow: hidden !important;
          }
          
          :global(.prediction-submit-btn:hover) {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(255, 149, 0, 0.4) !important;
            background: rgba(255, 149, 0, 0.95) !important;
          }
          
          :global(.prediction-submit-btn:active) {
            transform: translateY(1px) !important;
          }
          
          :global(.prediction-submit-btn::before) {
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: -100% !important;
            width: 100% !important;
            height: 100% !important;
            background: linear-gradient(120deg, transparent, rgba(255, 255, 255, 0.3), transparent) !important;
            transition: all 0.6s !important;
          }
          
          :global(.prediction-submit-btn:hover::before) {
            left: 100% !important;
          }
          
          :global(.prediction-submit-btn:disabled) {
            background: rgba(180, 180, 180, 0.7) !important;
            cursor: not-allowed !important;
            box-shadow: none !important;
          }
          
          /* Matching styles for any select dropdowns in FifaTeamSelection */
          :global(.team-select) {
            border-radius: 8px !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
            padding: 8px 12px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
            transition: all 0.3s ease !important;
          }
          
          :global(.team-select:focus) {
            border-color: #ff9500 !important;
            box-shadow: 0 0 0 3px rgba(255, 149, 0, 0.2) !important;
            outline: none !important;
          }
          
          @keyframes slideInFromRight {
            0% { transform: translateX(50px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
          }
        `}</style>
      </Container>
    </div>
  );
}

export default App;