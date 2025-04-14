import React, { useState, useRef } from 'react';
import { motion } from "framer-motion";
import { toast, ToastContainer } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';
import Webcam from "react-webcam";
import '../styles/FormStyles.css';

function VerifyForm() {
  // State variables
  const [faceFile, setFaceFile] = useState(null);
  const [voiceFile, setVoiceFile] = useState(null);
  const [policy, setPolicy] = useState('any');
  const [verificationResult, setVerificationResult] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);

  // State to control disable/enable of inputs
  const [faceInputDisabled, setFaceInputDisabled] = useState(false);
  const [voiceInputDisabled, setVoiceInputDisabled] = useState(false);

  // Refs
  const webcamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);

  // Validate file type and size
  const validateFile = (file, type) => {
    if (!file) return false;

    // File type validation
    const allowedImageTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    const allowedAudioTypes = [
      // WAV variations
      'audio/wav',
      'audio/wave',
      'audio/x-wav',

      // MP3 variations
      'audio/mpeg',
      'audio/mp3',

      // M4A variations
      'audio/mp4',
      'audio/m4a',
      'audio/x-m4a'
    ];

    const maxFileSize = 5 * 1024 * 1024; // 5MB

    if (type === 'image' && !allowedImageTypes.includes(file.type)) {
      toast.error('Invalid image type. Please upload JPG or PNG.');
      return false;
    }

    if (type === 'audio' && !allowedAudioTypes.includes(file.type)) {
      console.log('Attempted file type:', file.type); // Debug line
      toast.error('Invalid audio type. Please upload WAV, MP3, or M4A.');
      return false;
    }

    if (file.size > maxFileSize) {
      toast.error('File size exceeds 5MB limit.');
      return false;
    }

    return true;
  };

  // Capture face from webcam
  const captureFace = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      setCapturedImage(imageSrc);
      setFaceInputDisabled(true); // Disable file upload
      toast.success('📷 Face captured!');
    } else {
      toast.error('Failed to capture image. Please try again.');
    }
  };

  // Start recording voice
  const startRecording = async () => {
    try {
      audioChunksRef.current = [];
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const audioFile = new File([audioBlob], 'recorded_audio.wav', { type: 'audio/wav' });

        setVoiceFile(audioFile);
        setVoiceInputDisabled(true); // Disable file upload
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
        }
        toast.success('🎙️ Voice recorded!');
      };

      setIsRecording(true);
      mediaRecorder.start();

      setTimeout(() => {
        mediaRecorder.stop();
        setIsRecording(false);
      }, 5000);
    } catch (err) {
      console.error('Microphone access error:', err);
      toast.error('Could not access microphone. Please check permissions.');
      setIsRecording(false);
    }
  };

  // Handle face file upload
  const handleFaceFileUpload = (e) => {
    const file = e.target.files[0];
    if (validateFile(file, 'image')) {
      setFaceFile(file);
      setCapturedImage(null); // Clear webcam capture if file is uploaded
      setFaceInputDisabled(true); // Disable capture
    }
  };

  // Handle voice file upload
  const handleVoiceFileUpload = (e) => {
    const file = e.target.files[0];
    if (validateFile(file, 'audio')) {
      setVoiceFile(file);
      setVoiceInputDisabled(true); // Disable record button
    }
  };

  // Clear captured image
  const clearCapturedImage = () => {
    setCapturedImage(null);
    setFaceInputDisabled(false);
  };

  // Clear uploaded face file
  const clearFaceFile = () => {
    setFaceFile(null);
    setFaceInputDisabled(false);
  };

  // Clear recorded voice
  const clearVoiceRecording = () => {
    setVoiceFile(null);
    setVoiceInputDisabled(false);
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    // Validation
    if (!faceFile && !capturedImage && !voiceFile) {
      toast.error('Please provide at least one verification input.');
      return;
    }

    const formData = new FormData();

    // Append face data
    if (faceFile) {
      formData.append('face', faceFile);
    } else if (capturedImage) {
      try {
        const blob = await fetch(capturedImage).then((res) => res.blob());
        formData.append('face', blob, 'captured_face.jpg');
      } catch (err) {
        toast.error('Error processing captured image.');
        return;
      }
    }

    // Append voice data
    if (voiceFile) {
      formData.append('voice', voiceFile);
    }

    // Append policy
    formData.append('policy', policy);

    try {
      const response = await fetch('http://127.0.0.1:5000/verify', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (response.ok) {
        setVerificationResult(data);
        toast.success('✅ Verification completed.');
      } else {
        toast.error(data.error || 'Verification failed.');
      }
    } catch (err) {
      console.error(err);
      toast.error('Server error. Please try again.');
    }
  };

  return (
    <motion.div
      className="form-container"
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <ToastContainer position="top-right" autoClose={3000} />
      <h2>Verify Identity</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Face Verification:</label>
          {capturedImage ? (
            <div>
              <img src={capturedImage} alt="Captured Face" width="100%" style={{ borderRadius: '10px', marginBottom: '10px' }} />
              <button type="button" onClick={clearCapturedImage}>Clear Image</button>
            </div>
          ) : (
            <Webcam ref={webcamRef} screenshotFormat="image/jpeg" width="100%" />
          )}
          <button type="button" onClick={captureFace} disabled={faceInputDisabled}>📸 Capture Face</button>
          <p>OR</p>
          <input type="file" accept=".jpg,.jpeg,.png" onChange={handleFaceFileUpload} disabled={faceInputDisabled} />
          {faceFile && <button type="button" onClick={clearFaceFile}>Clear Upload</button>}
        </div>

        <div className="form-group">
          <label>Voice Verification:</label>
          {isRecording ? <p>🎤 Recording... (5 sec)</p> : <button type="button" onClick={startRecording} disabled={voiceInputDisabled}>🎙️ Record Voice</button>}
          <p>OR</p>
          <input type="file" accept=".wav,.mp3,.m4a" onChange={handleVoiceFileUpload} disabled={voiceInputDisabled} />
          {voiceFile && <button type="button" onClick={clearVoiceRecording}>Clear Upload</button>}
        </div>

        <div className="form-group">
          <label>Verification Policy:</label>
          <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
            <option value="any">Any</option>
            <option value="all">All</option>
            <option value="face_only">Face Only</option>
            <option value="voice_only">Voice Only</option>
          </select>
        </div>

        <motion.button
          type="submit"
          whileTap={{ scale: 0.95 }}
          whileHover={{ scale: 1.05 }}
        >
          Verify
        </motion.button>
      </form>

      {verificationResult && (
        <motion.div
          className="verification-result"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <h3>Verification Result</h3>
          <p>
            <strong>Verified:</strong>
            {verificationResult.verified ? ' ✅ Success' : ' ❌ Failed'}
          </p>
          <p><strong>Face Verified:</strong> {verificationResult.face_verified ? verificationResult.face_person || 'Verified User' : 'Unknown ❌'}</p>
          <p><strong>Voice Verified:</strong> {verificationResult.voice_verified ? verificationResult.voice_person || 'Verified User' : 'Unknown ❌'}</p>
        </motion.div>
      )}
    </motion.div>
  );
}

export default VerifyForm;