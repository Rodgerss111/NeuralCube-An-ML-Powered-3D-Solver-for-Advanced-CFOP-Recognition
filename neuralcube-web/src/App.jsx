import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import labelMap from './f2l_label_map.json'; // Importing the dictionary you generated
import CubeModel from './components/CubeModel'; // <-- NEW IMPORT

import './App.css'; // Standard Vite CSS

function App() {
  // State Management for the AI
  const [model, setModel] = useState(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [currentPrediction, setCurrentPrediction] = useState("Awaiting Cube State...");

  // 1. The Initialization Hook: Loads the AI on startup
  useEffect(() => {
    async function loadAIBrain() {
      try {
        console.log("Initializing NeuralCube Brain...");
        // tf.loadLayersModel looks in the public folder by default
        const loadedModel = await tf.loadLayersModel('/web_model_f2l/model.json');
        
        setModel(loadedModel);
        setIsModelLoading(false);
        console.log("F2L Model Loaded Successfully, Sir!");
      } catch (error) {
        console.error("Error loading the AI model:", error);
        setCurrentPrediction("Error: Could not load AI.");
      }
    }

    loadAIBrain();
  }, []);

  // 2. The Inference Function (With TTA Placeholder)
  const handleSolveClick = async () => {
    if (!model) return;

    // TODO: In the next step, we will get the actual 324-node array from the 3D cube.
    // For now, this is a mock tensor simulating a perfect canonical state.
    const mockCubeState = Array(324).fill(0); 
    mockCubeState[0] = 1; // Just setting one value to 1 for the mock data

    // Convert JS array to a 2D Tensor: [1 row, 324 columns]
    const inputTensor = tf.tensor2d([mockCubeState], [1, 324]);

    // Ask the AI to predict
    const prediction = model.predict(inputTensor);
    
    // Extract the highest probability and its index
    const probabilities = await prediction.data();
    const highestConfidence = Math.max(...probabilities);
    const predictedIndex = probabilities.indexOf(highestConfidence);

    // Map the index back to your F2L Case String
    const predictedLabel = labelMap[predictedIndex.toString()];

    // Test-Time Augmentation (TTA) Logic
    if (highestConfidence < 0.95) {
      setCurrentPrediction("Confidence too low. Rotating U-face and retrying...");
      // TODO: Add logic to virtually rotate the 3D cube's U-face and recall handleSolveClick()
    } else {
      setCurrentPrediction(`Predicted Case: ${predictedLabel} (Confidence: ${(highestConfidence * 100).toFixed(2)}%)`);
    }

    // Clean up memory (crucial for TFJS performance)
    inputTensor.dispose();
    prediction.dispose();
  };

  return (
    <div className="app-container" style={{ fontFamily: 'sans-serif', textAlign: 'center', padding: '2rem' }}>
      <h1>NeuralCube: F2L Solver</h1>
      <p>Advanced CFOP Recognition Engine</p>

      {/* The AI Status Indicator */}
      <div style={{ margin: '2rem 0', padding: '1rem', border: '1px solid #ccc', borderRadius: '8px', backgroundColor: '#f9f9f9' }}>
        {isModelLoading ? (
          <h3 style={{ color: '#d9534f' }}>Loading AI Brain into Browser Memory...</h3>
        ) : (
          <h3 style={{ color: '#5cb85c' }}>AI Brain Online and Ready</h3>
        )}
      </div>

      {/* The 3D Cube Canvas will go here */}
      <div style={{ width: '400px', height: '400px', margin: '0 auto', backgroundColor: '#222', borderRadius: '8px', boxShadow: '0 4px 10px rgba(0,0,0,0.3)'}}>
        <CubeModel />
        <p style={{ color: '#888' }}>[Three.js Interactive Cube Component Placeholder]</p>
      </div>

      <div style={{ marginTop: '2rem' }}>
        <button 
          onClick={handleSolveClick} 
          disabled={isModelLoading}
          style={{ padding: '10px 20px', fontSize: '1.2rem', cursor: isModelLoading ? 'not-allowed' : 'pointer' }}
        >
          Diagnose Cube State
        </button>
      </div>

      <div style={{ marginTop: '1.5rem', fontSize: '1.2rem', fontWeight: 'bold' }}>
        {currentPrediction}
      </div>
    </div>
  );
}

export default App;