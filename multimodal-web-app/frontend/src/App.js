import React from 'react';
import FaceRegisterForm from './components/FaceRegisterForm';
import VoiceRegisterForm from './components/VoiceRegisterForm';
import VerifyForm from "./components/VerifyForm";
import './App.css';

function App() {
  return (
    <div>
      <h1>Multimodal Verification Portal</h1>
      <FaceRegisterForm />
      <VoiceRegisterForm />
        <VerifyForm />
    </div>
  );
}

export default App;

