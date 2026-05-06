import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import labelMap from './components/f2l_label_map.json';
import CubeModel from './components/CubeModel';
import './App.css';

const PALETTE = {
  White: '#FFFFFF', Yellow: '#FFFF00', Red: '#FF0000',
  Orange: '#FFA500', Green: '#008000', Blue: '#0000FF'
};


// The 8 F2L piece identities (color sets, not positions)
const F2L_PIECES = {
  FR: {
    corner: new Set([2, 4, 0]), // Red + Blue + White
    edge:   new Set([2, 4])     // Red + Blue
  },
  FL: { corner: new Set([3, 4, 0]), edge: new Set([3, 4]) }, // Orange + Blue + White
  BR: { corner: new Set([2, 5, 0]), edge: new Set([2, 5]) }, // Red + Green + White
  BL: { corner: new Set([3, 5, 0]), edge: new Set([3, 5]) }, // Orange + Green + White
};

// Given the 54-int array, find where a piece is by its color identity
const _findPiece = (intStickers, colorSet) => {
  // Corner positions: groups of 3 sticker indices
  const CORNER_GROUPS = [
    [0,20,38],  // UFL
    [2,27,47],  // UFR  ← target for FR corner when in U
    [6,18,36],  // UBL  (misnomer: actually ULF in standard, depends on your face order)
    [8,29,45],  // UBR
    [9,19,41],  // DFL
    [11,30,44], // DFR  ← target slot
    [15,21,39], // DBL
    [17,32,42], // DBR
  ];
  const EDGE_GROUPS = [
    [1,37],[3,19],[5,28],[7,46], // U edges
    [10,40],[12,21],[14,30],[16,48], // D edges (approx)
    [23,34],[25,43],[32,52],[41,50], // Middle layer
  ];

  for (const group of [...CORNER_GROUPS, ...EDGE_GROUPS]) {
    const colors = new Set(group.map(i => intStickers[i]));
    // Check if the piece's colors match
    if ([...colorSet].every(c => colors.has(c)) && colors.size === colorSet.size) {
      return group; // Return the indices where it currently lives
    }
  }
  return null;
};

// App.jsx — corrected COLOR_TO_INT
const COLOR_TO_INT = {
  '#FFFFFF': 0, // White  (D face center) → 0
  '#FFFF00': 1, // Yellow (U face center) → 1
  '#FF0000': 2, // Red    (R face center) → 2  ✅ fixed
  '#FFA500': 3, // Orange (L face center) → 3  ✅ fixed
  '#0000FF': 4, // Blue   (F face center) → 4
  '#008000': 5  // Green  (B face center) → 5
};

const MOVE_LABELS = new Set([
  'U', "U'", 'U2',
  'D', "D'", 'D2',
  'L', "L'", 'L2',
  'R', "R'", 'R2',
  'F', "F'", 'F2',
  'B', "B'", 'B2',
  'y', "y'", 'y2'
]);

const isMoveLabel = (label) => MOVE_LABELS.has(label || '');

const extractOrderedStickers = (ref) => {
  const state = ref.current;
  const stickers = [];
  
  // 1. U Face (y = 1)
  for(let z of [-1, 0, 1]) for(let x of [-1, 0, 1]) stickers.push(state[`U_${x}_1_${z}`]);
  // 2. D Face (y = -1)
  for(let z of [1, 0, -1]) for(let x of [-1, 0, 1]) stickers.push(state[`D_${x}_-1_${z}`]);
  // 3. L Face (x = -1)
  for(let y of [1, 0, -1]) for(let z of [-1, 0, 1]) stickers.push(state[`L_-1_${y}_${z}`]);
  // 4. R Face (x = 1)
  for(let y of [1, 0, -1]) for(let z of [1, 0, -1]) stickers.push(state[`R_1_${y}_${z}`]);
  // 5. F Face (z = 1)
  for(let y of [1, 0, -1]) for(let x of [-1, 0, 1]) stickers.push(state[`F_${x}_${y}_1`]);
  // 6. B Face (z = -1)
  for(let y of [1, 0, -1]) for(let x of [1, 0, -1]) stickers.push(state[`B_${x}_${y}_-1`]);

  return stickers;
};

// NEW: Data Masking Function (The "Blinders")
// This zeroes out the noise from irrelevant slots, assuming the Front-Right slot is the active target.
// THE PERFECTLY SOLVED BASELINE STATE
const SOLVED_STATE = [
  1, 1, 1, 1, 1, 1, 1, 1, 1, // U (Yellow)
  0, 0, 0, 0, 0, 0, 0, 0, 0, // D (White)
  2, 2, 2, 2, 2, 2, 2, 2, 2, // L (Orange)
  3, 3, 3, 3, 3, 3, 3, 3, 3, // R (Red)
  4, 4, 4, 4, 4, 4, 4, 4, 4, // F (Blue)
  5, 5, 5, 5, 5, 5, 5, 5, 5  // B (Green)
];

// NEW: The Mathematical Blinders
const maskNoise = (intStickers) => {
  // Start with a mathematically perfect, clean cube
  const masked = [...SOLVED_STATE];
  
  // These are the exact tensor indices for the U-Layer and the Front-Right Slot
  // The EXACT 26 stickers required for the AI to see the F2L setup
  const ACTIVE_INDICES = [
    // 1. The Entire Top Layer (Roof + 4 Walls)
    0, 1, 2, 3, 4, 5, 6, 7, 8,     // U Face (Roof)
    18, 19, 20,                    // L Face (Top Row)
    27, 28, 29,                    // R Face (Top Row)
    36, 37, 38,                    // F Face (Top Row)
    45, 46, 47,                    // B Face (Top Row)
    
    // 2. The Front-Right Slot (The 5 stickers of the target Edge and Corner)
    11, // Down face of the FR corner
    30, // Right face of the FR edge
    33, // Right face of the FR corner
    41, // Front face of the FR edge
    44  // Front face of the FR corner
  ];

  // Look at the user's messy cube, extract ONLY the active pieces, 
  // and paste them onto our perfectly clean cube.
  ACTIVE_INDICES.forEach(index => {
    masked[index] = intStickers[index];
  });

  return masked; // Send the noise-free cube to the AI
};

function App() {
  const [model, setModel] = useState(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [currentPrediction, setCurrentPrediction] = useState("Awaiting Cube State...");
  const [activeColor, setActiveColor] = useState(PALETTE.White);
  
  // NEW: Debug State
  const [debugTensor, setDebugTensor] = useState([]);

  const cubeStateRef = useRef({});

  useEffect(() => {
    async function loadAIBrain() {
      try {
        // Appending a timestamp forces the browser to bypass the cache!
        const loadedModel = await tf.loadLayersModel(`/web_model_f2l/model.json?t=${new Date().getTime()}`);
        setModel(loadedModel);
        setIsModelLoading(false);
      } catch (error) {
        console.error("Error loading the AI model:", error);
      }
    }
    loadAIBrain();
  }, []);

  const handleSolveClick = async () => {
    if (!model) return;

    const rawStickers = extractOrderedStickers(cubeStateRef);
    let intStickers = rawStickers.map(hex => COLOR_TO_INT[hex] !== undefined ? COLOR_TO_INT[hex] : 0);

    // 1. APPLY THE MASK: Blind the AI to the chaotic noise in other slots
    intStickers = maskNoise(intStickers);

    setDebugTensor(intStickers);

    const oneHotArray = [];
    intStickers.forEach(val => {
      for (let i = 0; i < 6; i++) {
        oneHotArray.push(i === val ? 1 : 0);
      }
    });

    const inputTensor = tf.tensor2d([oneHotArray], [1, 324]);
    const prediction = model.predict(inputTensor);
    const probabilities = await prediction.data();
    
    const highestConfidence = Math.max(...probabilities);
    const predictedIndex = probabilities.indexOf(highestConfidence);
    const predictedLabel = labelMap[predictedIndex.toString()] || 'UNKNOWN';

    // 2. THE STATE MACHINE LOGIC
    if (highestConfidence < 0.65) {
      setCurrentPrediction(`Confidence too low: Expected ${predictedLabel} (${(highestConfidence * 100).toFixed(2)}%)`);
    } else if (predictedLabel === 'F2L_SOLVED' || predictedLabel === 'F2L_0_SOLVED') {
      setCurrentPrediction(`✓ Front-Right slot is solved! Rotate cube to next slot.`);
    } else if (isMoveLabel(predictedLabel)) {
      setCurrentPrediction(`Next move: ${predictedLabel} (${(highestConfidence * 100).toFixed(1)}%)`);
    } else if ((predictedLabel || '').startsWith("ACTION_EXTRACT")) {
      // If the AI recognizes a trapped piece, it issues an extraction command!
      setCurrentPrediction(`Trapped Piece Detected! Command: ${predictedLabel} (${(highestConfidence * 100).toFixed(2)}%)`);
      // TODO: In the future, trigger a virtual 3D animation here to extract the piece automatically.
    } else {
      setCurrentPrediction(`Algorithm: ${predictedLabel} (${(highestConfidence * 100).toFixed(1)}%)`);
    }

    inputTensor.dispose();
    prediction.dispose();
  };

  return (
    <div className="app-container" style={{ fontFamily: 'sans-serif', textAlign: 'center', padding: '2rem' }}>
      <h1>NeuralCube: F2L Solver</h1>
      
      {/* Palette */}
      <div style={{ margin: '1rem auto', padding: '10px', backgroundColor: '#333', borderRadius: '8px', display: 'inline-block' }}>
        <div style={{ display: 'flex', gap: '10px' }}>
          {Object.entries(PALETTE).map(([name, hex]) => (
            <button key={name} onClick={() => setActiveColor(hex)}
              style={{
                backgroundColor: hex, width: '30px', height: '30px', borderRadius: '50%',
                border: activeColor === hex ? '3px solid #0f0' : '2px solid white', cursor: 'pointer'
              }}
            />
          ))}
        </div>
      </div>

      <div style={{ width: '400px', height: '400px', margin: '0 auto', backgroundColor: '#222', borderRadius: '8px' }}>
        <CubeModel activeColor={activeColor} cubeStateRef={cubeStateRef} />
      </div>

      <div style={{ marginTop: '1.5rem' }}>
        <button onClick={handleSolveClick} disabled={isModelLoading} style={{ padding: '10px 20px', fontSize: '1.2rem', cursor: 'pointer' }}>
          Diagnose Cube State
        </button>
      </div>

      <div style={{ marginTop: '1rem', fontSize: '1.2rem', fontWeight: 'bold', color: '#333' }}>
        {currentPrediction}
      </div>

      {/* NEW: THE DIAGNOSTIC TERMINAL */}
      <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#111', color: '#0f0', fontFamily: 'monospace', fontSize: '0.9rem', borderRadius: '8px', wordBreak: 'break-all', maxWidth: '600px', margin: '2rem auto' }}>
        <p style={{ color: 'white', borderBottom: '1px solid #444', paddingBottom: '5px', margin: '0 0 10px 0' }}>// Diagnostic Tensor Output</p>
        {debugTensor.length > 0 ? `[${debugTensor.join(', ')}]` : "Awaiting array generation..."}
      </div>
    </div>
  );
}

export default App;