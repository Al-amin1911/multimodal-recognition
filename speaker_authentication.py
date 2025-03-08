import sounddevice as sd
import librosa
import numpy as np
import tempfile
from scipy.io.wavfile import write
from resemblyzer import preprocess_wav, VoiceEncoder
from qdrant_client import QdrantClient
from pydub import AudioSegment
import customtkinter as ctk
from tkinter import filedialog
import os
import logging
from typing import Optional
import threading

# Set environment variables for FFmpeg
os.environ['PATH'] = os.environ['PATH'] + ';C:\\ffmpeg\\bin'
os.environ['IMAGEIO_FFMPEG_EXE'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'

# Set PyDub paths
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffmpeg = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

# Configure PyDub to find FFmpeg
from pydub.utils import which
AudioSegment.converter = which("ffmpeg")

class SpeakerAuthApp:
    """
    Speaker Authentication Application with GUI interface.
    Handles voice recording, file upload, and speaker verification.
    """

    def check_ffmpeg(self):
        """Check if FFmpeg is installed and accessible"""
        try:
            logging.info("Testing FFmpeg with paths:")
            logging.info(f"Converter: {AudioSegment.converter}")
            logging.info(f"FFmpeg: {AudioSegment.ffmpeg}")
            logging.info(f"FFprobe: {AudioSegment.ffprobe}")
            
            # Create temp directory in current working directory instead of system temp
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create a temporary file in our temp directory
            temp_path = os.path.join(temp_dir, 'test.wav')
            
            # Test with a small dummy file
            dummy = AudioSegment.silent(duration=100)  # Create 100ms of silence
            dummy.export(temp_path, format='wav')
            
            # Clean up
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
                
            return True
            
        except Exception as e:
            logging.error(f"FFmpeg check failed with error: {str(e)}")
            # Check if files exist
            paths = [AudioSegment.converter, AudioSegment.ffmpeg, AudioSegment.ffprobe]
            for path in paths:
                if not os.path.exists(path):
                    logging.error(f"File does not exist: {path}")
            return False

    def __init__(self):
        """Initialize the application and its components."""
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='speaker_auth.log'
        )

        # Initialze core components
        self.setup_core_components()

        # Initialize GUI
        self.setup_gui()
        pass

    def setup_core_components(self):
        """Initialize the voice encoder and Qdrant client."""
        try:
            self.encoder = VoiceEncoder("cuda" if self.check_cuda() else "cpu")
            logging.info("Voice encoder initialized successfully")
        except Exception as e:
            logging.error(f"failed to initialize voice encoder: {e}")
            raise

        try:
            self.client = QdrantClient(
                url = "https://1c94b6cb-537b-4b28-9fbb-934eb3f2c6d8.europe-west3-0.gcp.cloud.qdrant.io:6333",
                api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ1ODgzNjI3fQ.7ueosobxV8059GgQG3Qal6S1idDFauCkaCvDHLDv-n0"
            )
            logging.info("Qdrant client initialized sucessfully")
        except Exception as e:
            logging.error(f"Failed to initialize Qdrant client: {e}")
            raise

        # Authentication settings
        self.AUTH_THRESHOLD = 0.7
        self.RECORD_DURATION = 5
        self.SAMPLE_RATE = 16000 

    def check_cuda(self) -> bool:
        """Check if CUDA  is available"""
        try:
            import torch
            return torch.cuda.is_available
        except:
            return False
        
    def setup_gui(self):
        """Set up the GUI components."""
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.app = ctk.CTk()
        self.app.geometry("400x400")
        self.app.title("Speaker Authentication")
        
        # Main title
        self.title_label = ctk.CTkLabel(
            self.app, 
            text="Speaker Authentication System",
            font=("Arial", 24)
        )
        self.title_label.pack(pady=20)
        
        # Status display
        self.status_label = ctk.CTkLabel(
            self.app,
            text="Ready",
            font=("Arial", 16)
        )
        self.status_label.pack(pady=10)
        
        # Buttons
        self.upload_btn = ctk.CTkButton(
            self.app,
            text="Upload Voice File",
            command=self.upload_audio,
            font=("Arial", 14)
        )
        self.upload_btn.pack(pady=10)
        
        self.record_btn = ctk.CTkButton(
            self.app,
            text="Record Voice (3s)",
            command=self.start_recording,
            font=("Arial", 14)
        )
        self.record_btn.pack(pady=10)
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(self.app)
        self.progress.pack(pady=10)
        self.progress.set(0)

    def update_status(self, message: str, is_error: bool = False):
        """Update the status display with a message"""
        self.status_label.configure(
            text=message,
            text_color="red" if is_error else "white"    
            )
        self.app.update()
    
    def record_audio(self) -> Optional[str]: #Returns a temporary file path (string) or None if recording fails, Optional[str] indicates the method can return either a string or None
        """Record audio and save to temp file"""
        try:
            self.update_status("Recording...")
            audio = sd.rec(
                int(self.RECORD_DURATION * self.SAMPLE_RATE),
                samplerate=self.SAMPLE_RATE,
                channels=1, # Mono audio recording
                dtype="float32"
            )

            # Show Recording process
            for i in range(self.RECORD_DURATION *10):
                self.progress.set((i + 1) / (self.RECORD_DURATION * 10)) # Creates a smooth progress bar during recording, Divides recording duration into 10 segments per second
                self.app.update()
                sd.sleep(100) # sleep for 100ms

            sd.wait(100)

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                write(temp_file.name, self.SAMPLE_RATE, (audio * 32767).astype(np.int16))
                return temp_file.name

        except Exception as e:
            self.update_status(f"Recording failed: {str(e)}", True)
            logging.error(f"Recording failed: {e}")
            return None
        
    def preprocess_audio(self, file_path: str) -> Optional[np.ndarray]:
        """Preprocess audio file for authentication"""  
        try:
            logging.info(f"Preprocessing file: {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            if file_path.endswith(".mp3"):
                if not self.check_ffmpeg():
                    self.update_status("FFmpeg not found. Please install FFmpeg to process MP3 files.", True)
                    logging.error("FFmpeg not installed")
                    return None
                
                try:
                    logging.info(f"Attempting to convert MP3: {file_path}")
                    logging.info(f"FFmpeg path: {AudioSegment.converter}")
                    
                    # Create temp directory
                    temp_dir = os.path.join(os.getcwd(), 'temp')
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Load MP3
                    logging.info("Loading MP3 file...")
                    audio = AudioSegment.from_file(file_path, format="mp3")
                    
                    # Create WAV path
                    wav_path = os.path.join(temp_dir, os.path.basename(file_path).replace(".mp3", ".wav"))
                    logging.info(f"Exporting to WAV: {wav_path}")
                    
                    # Export to WAV
                    audio.export(wav_path, format="wav")
                    logging.info("WAV export successful")
                    
                    file_path = wav_path
                    logging.info(f"Converted MP3 to WAV: {wav_path}")
                except Exception as e:
                    self.update_status("Error converting MP3. Please ensure FFmpeg is properly installed.", True)
                    logging.error(f"MP3 conversion error: {e}")
                    logging.error(f"MP3 conversion error details:", exc_info=True)
                    return None

            wav, _ = librosa.load(file_path, sr=self.SAMPLE_RATE)
            preprocessed_wav = preprocess_wav(wav)
            
            return preprocessed_wav
        
        except Exception as e:
            self.update_status(f"Preprocessing failed: {str(e)}", True)
            logging.error(f"Preprocessing failed: {e}", exc_info=True)
            return None
    
    def authenticate_user(self, input_audio: str):
        """Authenticate user against enrolled voices"""
        try:
            self.update_status(f"Preprocessing...")
            test_wav = self.preprocess_audio(input_audio)
            if test_wav is None:
                return
            
            test_embedding = self.encoder.embed_utterance(test_wav)
            results = self.client.search(
                collection_name="my-collection",
                query_vector=test_embedding.tolist(),
                limit=1
            )

            if results and results[0].score >= self.AUTH_THRESHOLD:
                self.update_status(
                    f"✓ Access granted: {results[0].payload['speaker']}\nScore: {results[0].score:.2f}"
                )
            else:
                self.update_status(f"✗ Access denied!: Score: {results[0].score:.2f}", True)

        except Exception as e:
            self.update_status(f"Authentication failed: {str(e)}", True)
            logging.error(f"Authentication failed: {e}")

    def start_recording(self):
        """Start recording in a seperate thread."""
        self.record_btn.configure(state="disabled")
        self.upload_btn.configure(state="disabled")

        def record_thread():
            temp_audio = self.record_audio()
            if temp_audio and os.path.exists(temp_audio):
                self.authenticate_user(temp_audio)

            self.progress.set(0)
            self.record_btn.configure(state="normal")
            self.upload_btn.configure(state="normal")

        threading.Thread(target=record_thread).start()

    def upload_audio(self):
        """Handle audio file upload."""
        try:
            # Get file path as string
            file_path = filedialog.askopenfilename(
                title="Select Audio File",
                filetypes=[("Audio Files", "*.wav;*.mp3")]
            )
            
            if file_path:
                # Debug logging
                logging.info(f"Selected file path: {file_path}")
                logging.info(f"File exists check: {os.path.exists(file_path)}")
                
                # Disable buttons during processing
                self.record_btn.configure(state="disabled")
                self.upload_btn.configure(state="disabled")
                
                # Update status and show file path
                self.update_status(f"Processing file: {os.path.basename(file_path)}\nPath: {file_path}")
                
                # Authenticate user with the file path string
                self.authenticate_user(file_path)
                
        except Exception as e:
            # Show detailed error in status
            error_msg = f"Error processing file: {str(e)}"
            self.update_status(error_msg, is_error=True)
            logging.error(f"File upload error: {e}", exc_info=True)
            
        finally:
            # Always re-enable buttons
            self.record_btn.configure(state="normal")
            self.upload_btn.configure(state="normal")

    def run(self):
        """Start the application."""
        self.app.mainloop()


def main():
    try:
        app = SpeakerAuthApp()
        app.run()
    except Exception as e:
        logging.error(f"Application failed to start: {e}")
        raise

if __name__ == "__main__":
    main()