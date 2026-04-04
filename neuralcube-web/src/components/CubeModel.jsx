import React from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

// Helper function to color the outside faces of the cube like a real Rubik's Cube,
// while keeping the inside faces black.
// Three.js Box material order: [Right, Left, Top, Bottom, Front, Back]
const getCubieColors = (x, y, z) => {
  return [
    x === 1 ? '#B71234' : '#000000',   // Right (Red)
    x === -1 ? '#FF5800' : '#000000',  // Left (Orange)
    y === 1 ? '#FFD500' : '#000000',   // Top (Yellow)
    y === -1 ? '#FFFFFF' : '#000000',  // Bottom (White)
    z === 1 ? '#009B48' : '#000000',   // Front (Green)
    z === -1 ? '#0046AD' : '#000000',  // Back (Blue)
  ];
};

// A single 1x1x1 piece of the Rubik's Cube
const Cubie = ({ position }) => {
  const colors = getCubieColors(position[0], position[1], position[2]);
  
  return (
    <mesh position={position}>
      {/* 0.95 size leaves a 0.05 gap between pieces, making it look realistic */}
      <boxGeometry args={[0.95, 0.95, 0.95]} />
      
      {/* Apply the 6 colors to the 6 faces of this specific cubie */}
      {colors.map((color, index) => (
        <meshStandardMaterial 
          key={index} 
          attach={`material-${index}`} 
          color={color} 
          roughness={0.2} // Makes the plastic look slightly shiny
        />
      ))}
    </mesh>
  );
};

export default function CubeModel() {
  // Generate the 27 cubies in a 3x3x3 grid
  const cubies = [];
  for (let x = -1; x <= 1; x++) {
    for (let y = -1; y <= 1; y++) {
      for (let z = -1; z <= 1; z++) {
        cubies.push(<Cubie key={`${x}-${y}-${z}`} position={[x, y, z]} />);
      }
    }
  }

  return (
    // The Canvas is the portal to the 3D world
    <Canvas camera={{ position: [4, 4, 4], fov: 45 }} style={{ width: '100%', height: '100%', borderRadius: '8px' }}>
      {/* Lighting so we can see the colors */}
      <ambientLight intensity={0.8} />
      <directionalLight position={[10, 10, 5]} intensity={1.5} />
      
      {/* Render the 27 cubies */}
      <group>{cubies}</group>
      
      {/* Allows the user to rotate the camera around the cube using their mouse */}
      <OrbitControls enablePan={false} minDistance={3} maxDistance={10} />
    </Canvas>
  );
}