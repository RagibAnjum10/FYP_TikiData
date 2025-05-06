import React, { useState } from 'react';
import { Form, Row, Col, Button } from 'react-bootstrap';

const FifaTeamSelection = ({ teams, home, away, setHome, setAway, loading, handleSubmit, error }) => {
  const [activeTab, setActiveTab] = useState('home'); // 'home' or 'away'
  
  // Team logos mapping (sample Premier League teams)
  const teamLogos = {
    "Manchester City": "https://resources.premierleague.com/premierleague/badges/t43.png",
    // Using the colored Liverpool logo that will stand out better
    "Liverpool": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
    "Arsenal": "https://resources.premierleague.com/premierleague/badges/t3.png",
    "Manchester United": "https://resources.premierleague.com/premierleague/badges/t1.png",
    "Chelsea": "https://resources.premierleague.com/premierleague/badges/t8.png",
    "Tottenham": "https://resources.premierleague.com/premierleague/badges/t6.png",
    "Newcastle United": "https://resources.premierleague.com/premierleague/badges/t4.png",
    "Aston Villa": "https://resources.premierleague.com/premierleague/badges/t7.png",
    "Brighton & Hove Albion": "https://resources.premierleague.com/premierleague/badges/t36.png",
    "West Ham United": "https://resources.premierleague.com/premierleague/badges/t21.png",
    "Crystal Palace": "https://resources.premierleague.com/premierleague/badges/t31.png",
    "Wolverhampton": "https://resources.premierleague.com/premierleague/badges/t39.png",
    "Everton": "https://resources.premierleague.com/premierleague/badges/t11.png",
    "Leicester City": "https://resources.premierleague.com/premierleague/badges/t13.png",
    "Brentford": "https://resources.premierleague.com/premierleague/badges/t94.png",
    "Fulham": "https://resources.premierleague.com/premierleague/badges/t54.png",
    "Bournemouth": "https://resources.premierleague.com/premierleague/badges/t91.png",
    "Nottingham Forest": "https://resources.premierleague.com/premierleague/badges/t17.png",
    "Leeds United": "https://resources.premierleague.com/premierleague/badges/t2.png",
    "Southampton": "https://resources.premierleague.com/premierleague/badges/t20.png"
  };

  // Function to get team logo or fallback
  const getTeamLogo = (teamName) => {
    // For demo purposes, use a placeholder if logo not found
    return teamLogos[teamName] || "/api/placeholder/80/80";
  };

  return (
    <div className="fifa-team-selection">
      {/* Team Selection Tabs */}
      <div className="team-tabs">
        <button 
          className={`team-tab ${activeTab === 'home' ? 'active' : ''}`}
          onClick={() => setActiveTab('home')}
        >
          HOME TEAM
        </button>
        <button 
          className={`team-tab ${activeTab === 'away' ? 'active' : ''}`}
          onClick={() => setActiveTab('away')}
        >
          AWAY TEAM
        </button>
      </div>
      
      {/* Active Team & Current Selection */}
      <div className="selected-teams">
        <div className="home-team">
          <div className="team-badge">
            <img src={getTeamLogo(home)} alt={home} className="team-logo" />
          </div>
          <div className="team-name">{home}</div>
        </div>
        
        <div className="versus">VS</div>
        
        <div className="away-team">
          <div className="team-badge">
            <img src={getTeamLogo(away)} alt={away} className="team-logo" />
          </div>
          <div className="team-name">{away}</div>
        </div>
      </div>
      
      {/* Team Selection Grid */}
      <div className="team-grid">
        <h4>{activeTab === 'home' ? 'Select Home Team' : 'Select Away Team'}</h4>
        <div className="teams-container">
          {teams.map(team => (
            <div 
              key={team}
              className={`team-item ${
                (activeTab === 'home' && team === home) || 
                (activeTab === 'away' && team === away) ? 'selected' : ''
              }`}
              onClick={() => {
                if (activeTab === 'home') {
                  setHome(team);
                  // Auto switch to away tab after selecting home
                  setActiveTab('away');
                } else {
                  setAway(team);
                }
              }}
            >
              <div className="team-item-inner">
                <img 
                  src={getTeamLogo(team)} 
                  alt={team} 
                  className="team-item-logo" 
                />
                <div className="team-item-name">{team}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Prediction Button & Error Message */}
      <div className="predict-actions">
        <Button 
          onClick={handleSubmit}
          className="fifa-predict-button"
          disabled={loading || home === away}
        >
          {loading ? 'SIMULATING MATCH...' : 'PREDICT MATCH'}
        </Button>
        
        {error && <div className="fifa-error">{error}</div>}
      </div>
      
      {/* FIFA-inspired styles */}
      <style jsx="true">{`
        .fifa-team-selection {
          padding: 15px;
          background-color: rgba(255, 255, 255, 0.95);
          border-radius: 10px;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .team-tabs {
          display: flex;
          margin-bottom: 20px;
          border-bottom: 3px solid #f2f2f2;
        }
        
        .team-tab {
          flex: 1;
          padding: 12px 0;
          text-align: center;
          background: none;
          border: none;
          font-weight: 600;
          font-size: 16px;
          color: #555;
          position: relative;
          cursor: pointer;
          transition: all 0.3s ease;
        }
        
        .team-tab:focus {
          outline: none;
        }
        
        .team-tab.active {
          color: #2255d1;
        }
        
        .team-tab.active:after {
          content: '';
          position: absolute;
          bottom: -3px;
          left: 0;
          width: 100%;
          height: 3px;
          background-color: #2255d1;
          animation: slideIn 0.3s ease forwards;
        }
        
        @keyframes slideIn {
          from { transform: scaleX(0); }
          to { transform: scaleX(1); }
        }
        
        .selected-teams {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 30px;
          padding: 15px;
          background: linear-gradient(to right, #004099, #1a237e);
          border-radius: 8px;
          color: white;
        }
        
        .home-team, .away-team {
          display: flex;
          flex-direction: column;
          align-items: center;
          transition: all 0.3s ease;
        }
        
        .team-badge {
          width: 70px;
          height: 70px;
          display: flex;
          justify-content: center;
          align-items: center;
          background-color: rgba(255, 255, 255, 0.9);
          border-radius: 50%;
          margin-bottom: 10px;
          box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
          transition: all 0.3s ease;
        }
        
        .team-logo {
          max-width: 55px;
          max-height: 55px;
          animation: pulseTeam 2s infinite alternate;
        }
        
        @keyframes pulseTeam {
          0% { transform: scale(1); }
          100% { transform: scale(1.05); }
        }
        
        .team-name {
          font-weight: 600;
          text-align: center;
          font-size: 14px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          max-width: 100px;
        }
        
        .versus {
          font-size: 22px;
          font-weight: 700;
          background: rgba(255, 255, 255, 0.15);
          border-radius: 50%;
          width: 50px;
          height: 50px;
          display: flex;
          justify-content: center;
          align-items: center;
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
          animation: glow 1.5s infinite alternate;
        }
        
        @keyframes glow {
          0% { box-shadow: 0 0 5px rgba(255, 255, 255, 0.3); }
          100% { box-shadow: 0 0 20px rgba(255, 255, 255, 0.8); }
        }
        
        .team-grid {
          margin-bottom: 25px;
        }
        
        .team-grid h4 {
          margin-bottom: 15px;
          color: #333;
          font-weight: 600;
          font-size: 18px;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .teams-container {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
          gap: 12px;
          max-height: 280px;
          overflow-y: auto;
          padding: 5px;
          position: relative;
        }
        
        .team-item {
          cursor: pointer;
          transition: all 0.2s ease;
          border-radius: 6px;
          overflow: hidden;
          position: relative;
        }
        
        .team-item:hover {
          transform: translateY(-3px);
          box-shadow: 0 5px 15px rgba(34, 85, 209, 0.2);
        }
        
        .team-item.selected {
          border: 2px solid #2255d1;
          background-color: rgba(34, 85, 209, 0.1);
        }
        
        .team-item.selected:after {
          content: '✓';
          position: absolute;
          top: 5px;
          right: 5px;
          background-color: #2255d1;
          color: white;
          border-radius: 50%;
          width: 20px;
          height: 20px;
          display: flex;
          justify-content: center;
          align-items: center;
          font-size: 12px;
        }
        
        .team-item-inner {
          padding: 10px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          background-color: white;
          transition: all 0.2s ease;
        }
        
        .team-item-logo {
          width: 40px;
          height: 40px;
          object-fit: contain;
          margin-bottom: 8px;
        }
        
        .team-item-name {
          font-size: 12px;
          font-weight: 500;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          max-width: 90px;
        }
        
        .predict-actions {
          display: flex;
          flex-direction: column;
          align-items: center;
        }
        
        .fifa-predict-button {
          background: linear-gradient(to right, #2255d1, #1e40af);
          border: none;
          border-radius: 30px;
          padding: 12px 30px;
          font-weight: 600;
          font-size: 16px;
          letter-spacing: 1px;
          box-shadow: 0 4px 15px rgba(34, 85, 209, 0.3);
          cursor: pointer;
          transition: all 0.3s ease;
          animation: pulseButton 2s infinite alternate;
        }
        
        @keyframes pulseButton {
          0% { transform: scale(1); box-shadow: 0 4px 15px rgba(34, 85, 209, 0.3); }
          100% { transform: scale(1.03); box-shadow: 0 4px 20px rgba(34, 85, 209, 0.5); }
        }
        
        .fifa-predict-button:hover:not(:disabled) {
          transform: translateY(-3px);
          box-shadow: 0 6px 20px rgba(34, 85, 209, 0.4);
        }
        
        .fifa-predict-button:active:not(:disabled) {
          transform: translateY(0);
          box-shadow: 0 2px 10px rgba(34, 85, 209, 0.2);
        }
        
        .fifa-predict-button:disabled {
          background: linear-gradient(to right, #a0aec0, #cbd5e0);
          box-shadow: none;
          cursor: not-allowed;
          animation: none;
        }
        
        .fifa-error {
          margin-top: 15px;
          color: #e53e3e;
          background-color: #fff5f5;
          padding: 8px 15px;
          border-radius: 4px;
          font-size: 14px;
          text-align: center;
          border-left: 3px solid #e53e3e;
          animation: shake 0.5s linear;
        }
        
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
          20%, 40%, 60%, 80% { transform: translateX(5px); }
        }

        /* Additional scrollbar styling */
        .teams-container::-webkit-scrollbar {
          width: 6px;
        }
        
        .teams-container::-webkit-scrollbar-track {
          background: #f1f1f1;
          border-radius: 10px;
        }
        
        .teams-container::-webkit-scrollbar-thumb {
          background: #c1c1c1;
          border-radius: 10px;
        }
        
        .teams-container::-webkit-scrollbar-thumb:hover {
          background: #a8a8a8;
        }
      `}</style>
    </div>
  );
};

export default FifaTeamSelection;