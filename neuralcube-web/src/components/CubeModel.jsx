import React, { useState, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

const getCubieColors = (x, y, z) => {
  return [
    x === 1 ? '#FF0000' : '#000000',   // Right (Red)
    x === -1 ? '#FFA500' : '#000000',  // Left (Orange)
    y === 1 ? '#FFFF00' : '#000000',   // Top (Yellow)
    y === -1 ? '#FFFFFF' : '#000000',  // Bottom (White)
    z === 1 ? '#0000FF' : '#000000',   // Front (Blue) - SWAPPED AS REQUESTED
    z === -1 ? '#008000' : '#000000',  // Back (Green) - SWAPPED AS REQUESTED
  ];
};

// Face mapping to give each sticker a unique name in our notebook
const FACE_NAMES = ['R', 'L', 'U', 'D', 'F', 'B'];

const Cubie = ({ position, activeColor, cubeStateRef }) => {
  const [colors, setColors] = useState(getCubieColors(position[0], position[1], position[2]));
  const pointerDownPos = useRef({ x: 0, y: 0 });

  // On initial load, write the starting colors into the global reference notebook
  useEffect(() => {
    colors.forEach((color, index) => {
      if (color !== '#000000') {
        const faceName = FACE_NAMES[index];
        const stickerId = `${faceName}_${position[0]}_${position[1]}_${position[2]}`;
        cubeStateRef.current[stickerId] = color;
      }
    });
  }, []);

  const handlePointerDown = (e) => {
    e.stopPropagation();
    pointerDownPos.current = { x: e.clientX, y: e.clientY };
  };

  const handlePointerUp = (e) => {
    e.stopPropagation(); 
    const moveX = Math.abs(e.clientX - pointerDownPos.current.x);
    const moveY = Math.abs(e.clientY - pointerDownPos.current.y);
    if (moveX > 5 || moveY > 5) return; // Prevent ghost clicks from dragging

    const faceIndex = Math.floor(e.faceIndex / 2);

    if (colors[faceIndex] !== '#000000' && activeColor) {
      const newColors = [...colors];
      newColors[faceIndex] = activeColor;
      setColors(newColors);

      // THE FIX: Save the new color into the parent's notebook
      const faceName = FACE_NAMES[faceIndex];
      const stickerId = `${faceName}_${position[0]}_${position[1]}_${position[2]}`;
      cubeStateRef.current[stickerId] = activeColor;
    }
  };
  
  return (
    <mesh position={position} onPointerDown={handlePointerDown} onPointerUp={handlePointerUp}>
      <boxGeometry args={[0.95, 0.95, 0.95]} />
      {colors.map((color, index) => (
        <meshStandardMaterial key={index} attach={`material-${index}`} color={color} roughness={0.1} />
      ))}
    </mesh>
  );
};

export default function CubeModel({ activeColor, cubeStateRef }) {
  const cubies = [];
  for (let x = -1; x <= 1; x++) {
    for (let y = -1; y <= 1; y++) {
      for (let z = -1; z <= 1; z++) {
        cubies.push(
          <Cubie 
            key={`${x}-${y}-${z}`} 
            position={[x, y, z]} 
            activeColor={activeColor} 
            cubeStateRef={cubeStateRef} // Pass the notebook down to every cubie
          />
        );
      }
    }
  }

  return (
    <Canvas camera={{ position: [4, 4, 4], fov: 45 }} style={{ width: '100%', height: '100%', borderRadius: '8px' }}>
      <ambientLight intensity={1.2} />
      <directionalLight position={[10, 10, 5]} intensity={1.0} />
      <group>{cubies}</group>
      <OrbitControls enablePan={false} minDistance={3} maxDistance={10} makeDefault />
    </Canvas>
  );
}