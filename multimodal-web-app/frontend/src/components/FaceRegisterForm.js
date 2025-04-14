import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'react-toastify';
import '../styles/FormStyles.css';

function FaceRegisterForm() {
  const [name, setName] = useState('');
  const [imageFiles, setImageFiles] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!name || imageFiles.length === 0) {
      toast.error('Please provide a name and at least one image!');
      return;
    }

    const formData = new FormData();
    formData.append('name', name);
    for (let i = 0; i < imageFiles.length; i++) {
      formData.append('images', imageFiles[i]);
    }

    try {
      const response = await fetch('http://127.0.0.1:5000/register/face', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (response.ok) {
        toast.success(`🎉 ${data.message}`);
        setName('');
        setImageFiles([]);
      } else {
        toast.error(data.error || 'Failed to register face.');
      }
    } catch (err) {
      console.error(err);
      toast.error('Server error.');
    }
  };

  return (
    <motion.div className="form-container">
      <h2>Register Face</h2>
      <form onSubmit={handleSubmit} encType="multipart/form-data">
        <input type="text" value={name} onChange={(e) => setName(e.target.value)} required />
        <input type="file" accept=".jpg,.jpeg,.png" multiple onChange={(e) => setImageFiles(Array.from(e.target.files))} required />
        <button type="submit">Submit</button>
      </form>
    </motion.div>
  );
}

export default FaceRegisterForm;
