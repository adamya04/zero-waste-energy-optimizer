import React, { createContext, useContext, useEffect } from 'react';
import { io } from 'socket.io-client';

const SocketIOContext = createContext();

export function SocketIOProvider({ children }) {
  const socket = io('/', { path: '/socket.io', transports: ['websocket'] });

  useEffect(() => {
    socket.on('connect', () => console.log('Connected to WebSocket'));
    socket.on('disconnect', () => console.log('Disconnected from WebSocket'));
    socket.on('inventory_update', (data) => console.log('Inventory update:', data));
    socket.on('analytics_update', (data) => console.log('Analytics update:', data));
    return () => {
      socket.off('connect');
      socket.off('disconnect');
      socket.off('inventory_update');
      socket.off('analytics_update');
      socket.disconnect();
    };
  }, [socket]);

  return (
    <SocketIOContext.Provider value={socket}>
      {children}
    </SocketIOContext.Provider>
  );
}

export const useSocket = () => useContext(SocketIOContext);