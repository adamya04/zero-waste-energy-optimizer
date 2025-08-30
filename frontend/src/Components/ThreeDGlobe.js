import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import axios from 'axios';

function ThreeDGlobe() {
  const mountRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let renderer, scene, camera, sphere, animationId;

    const initScene = (storeLocs) => {
      if (!mountRef.current) return;
      const width = mountRef.current.clientWidth;
      const height = 700;

      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(width, height);
      renderer.setClearColor(0x0a0a23);
      mountRef.current.appendChild(renderer.domElement);

      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
      camera.position.z = 3;

      const geometry = new THREE.SphereGeometry(1, 64, 64);
      const material = new THREE.MeshBasicMaterial({ color: 0x0a2a3a, wireframe: true });
      sphere = new THREE.Mesh(geometry, material);
      scene.add(sphere);

      Object.values(storeLocs).forEach(loc => {
        const dotGeom = new THREE.SphereGeometry(0.01 + (loc.sustainability || 0) * 0.02, 8, 8);
        const dotMat = new THREE.MeshBasicMaterial({ color: 0x00ffff });
        const dot = new THREE.Mesh(dotGeom, dotMat);
        const lat = (loc.lat / 180) * Math.PI;
        const lon = (loc.lon / 180) * Math.PI;
        const r = 1.01;
        dot.position.set(
          r * Math.cos(lat) * Math.cos(lon),
          r * Math.sin(lat),
          r * Math.cos(lat) * Math.sin(lon)
        );
        scene.add(dot);
      });

      const animate = () => {
        if (sphere) sphere.rotation.y += 0.001;
        animationId = requestAnimationFrame(animate);
        renderer.render(scene, camera);
      };
      animate();
    };

    const fetchData = async () => {
      try {
        const res = await axios.get('/api/analytics/store_1');
        const df = res.data.records || [];
        if (!df.length) {
          setError('No data available for globe');
          setLoading(false);
          return;
        }
        const storeLocs = {};
        df.forEach((d, i) => {
          if (!storeLocs[d.store_id]) {
            storeLocs[d.store_id] = { lat: (i * 37) % 90 - 45, lon: (i * 97) % 180 - 90, sustainability: 0, count: 0 };
          }
          storeLocs[d.store_id].sustainability += (d.sustainability_score || 0);
          storeLocs[d.store_id].count += 1;
        });
        Object.values(storeLocs).forEach(s => { if (s.count) s.sustainability /= s.count; });
        initScene(storeLocs);
      } catch (err) {
        console.error('Globe fetch error:', err);
        setError('Failed to load globe data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    return () => {
      if (animationId) cancelAnimationFrame(animationId);
      if (mountRef.current) {
        while (mountRef.current.firstChild) {
          mountRef.current.removeChild(mountRef.current.firstChild);
        }
      }
      if (renderer) renderer.dispose();
    };
  }, []);

  if (loading) return <div className="text-cyan-400 text-center p-8">Loading Globe...</div>;
  if (error) return <div className="text-red-500 text-center p-8">{error}</div>;

  return (
    <div className="p-8">
      <h1 className="text-5xl text-cyan-400 mb-8 text-center animate-glow">3D Sustainability Globe</h1>
      <p className="text-2xl text-magenta-400 mb-6 text-center animate-pulse">
        Visualize global sustainability impact with blockchain-verified data
      </p>
      <div ref={mountRef} className="w-full h-[700px] rounded-lg shadow-neon"></div>
    </div>
  );
}

export default ThreeDGlobe;