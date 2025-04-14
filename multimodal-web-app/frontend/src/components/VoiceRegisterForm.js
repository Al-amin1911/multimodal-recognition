import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'react-toastify';
import '../styles/FormStyles.css';

function VoiceRegisterForm() {
  const [name, setName] = useState('');
  const [audioFile, setAudioFile] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!name || !audioFile) {
      toast.error('Please provide a name and an audio file!');
      return;
    }

    const formData = new FormData();
    formData.append('name', name);
    formData.append('audio', audioFile);

    try {
      const response = await fetch('http://127.0.0.1:5000/register/voice', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (response.ok) {
        toast.success(`🔊 ${data.message}`);
        setName('');
        setAudioFile(null);
      } else {
        toast.error(data.error || 'Failed to register voice.');
      }
    } catch (err) {
      console.error(err);
      toast.error('Server error. Try again.');
    }
  };

  return (
    <motion.div
      className="form-container"
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.2, ease: 'easeOut' }}
    >
      <h2>Register Voice</h2>
      <form onSubmit={handleSubmit} encType="multipart/form-data">
        <div className="form-group">
          <label>Name:</label>
          <input
            type="text"
            value={name}
            placeholder="e.g. jeff_bezos"
            onChange={(e) => setName(e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <label>Audio File:</label>
          <input
            type="file"
            accept=".wav,.mp3,.m4a"
            onChange={(e) => setAudioFile(e.target.files[0])}
            required
          />
        </div>
        <motion.button
          type="submit"
          whileTap={{ scale: 0.95 }}
          whileHover={{ scale: 1.05 }}
        >
          Submit
        </motion.button>
      </form>
    </motion.div>
  );
}

export default VoiceRegisterForm;
