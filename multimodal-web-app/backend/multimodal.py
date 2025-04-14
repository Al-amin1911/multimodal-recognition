import customtkinter as ctk
import cv2
import sounddevice as sd
import numpy as np
import os
import logging
import threading
import tempfile
from scipy.io.wavfile import write
from PIL import Image
from tkinter import filedialog
from typing import Optional, Dict, Tuple

# Face recognition imports
from opencv.fr import FR
from opencv.fr.search.schemas import SearchRequest

# Speaker authentication imports
from resemblyzer import preprocess_wav, VoiceEncoder
from qdrant_client import QdrantClient
from pydub import AudioSegment
from pydub.utils import which

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='multimodal_verification.log'
)

# Set environment variables for FFmpeg
os.environ['PATH'] = os.environ['PATH'] + ';C:\\ffmpeg\\bin'
os.environ['IMAGEIO_FFMPEG_EXE'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'

# Set PyDub paths
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffmpeg = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

# Configure PyDub to find FFmpeg
AudioSegment.converter = which("ffmpeg")


class VerificationResult:
    """Class to store verification results from different modalities"""

    def __init__(self):
        self.face_verified = False
        self.face_score = 0.0
        self.face_person = None

        self.voice_verified = False
        self.voice_score = 0.0
        self.voice_person = None

    def is_verified(self, policy="any") -> bool:
        """Check if verification passes based on the selected policy"""
        if policy == "any":
            return self.face_verified or self.voice_verified
        elif policy == "all":
            return self.face_verified and self.voice_verified
        elif policy == "face_only":
            return self.face_verified
        elif policy == "voice_only":
            return self.voice_verified
        return False

    def get_summary(self) -> str:
        """Get a text summary of verification results"""
        summary = []

        if self.face_verified:
            summary.append(f"Face verified: {self.face_person} (score: {self.face_score:.2f})")
        else:
            summary.append(f"Face not verified (score: {self.face_score:.2f})")

        if self.voice_verified:
            summary.append(f"Voice verified: {self.voice_person} (score: {self.voice_score:.2f})")
        else:
            summary.append(f"Voice not verified (score: {self.voice_score:.2f})")

        return "\n".join(summary)


class MultimodalVerificationSystem:
    """Integrated system for face and voice verification"""

    def __init__(self):
        # Initialize face recognition
        self.setup_face_recognition()

        # Initialize speaker authentication
        self.setup_speaker_authentication()

        # Common settings
        self.verification_policy = "any"  # Options: "any", "all", "face_only", "voice_only"

        # Initialize GUI
        self.setup_gui()

    def setup_face_recognition(self):
        """Initialize face recognition components"""
        try:
            # Establish connection
            BACKEND_URL = "https://eu.opencv.fr"
            DEVELOPER_KEY = "7BRQ8i1YWQ2MWM4NjYtMDUyNy00ZTEzLThmZGItZmZmOWRhNjE0ZWFj"

            # Initialize the SDK
            self.face_sdk = FR(BACKEND_URL, DEVELOPER_KEY)
            self.face_threshold = 0.8  # Confidence threshold
            logging.info("Face recognition initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize face recognition: {e}")
            raise

    def setup_speaker_authentication(self):
        """Initialize speaker authentication components"""
        try:
            self.voice_encoder = VoiceEncoder("cuda" if self.check_cuda() else "cpu")
            logging.info("Voice encoder initialized successfully")

            self.voice_client = QdrantClient(
                url="https://1c94b6cb-537b-4b28-9fbb-934eb3f2c6d8.europe-west3-0.gcp.cloud.qdrant.io:6333",
                api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ1ODgzNjI3fQ.7ueosobxV8059GgQG3Qal6S1idDFauCkaCvDHLDv-n0"
            )
            logging.info("Qdrant client initialized successfully")

            self.voice_threshold = 0.7  # Authentication threshold
            self.record_duration = 5  # seconds
            self.sample_rate = 16000  # Hz

            # Supported audio formats
            self.supported_audio_formats = [".wav", ".mp3", ".m4a"]
        except Exception as e:
            logging.error(f"Failed to initialize speaker authentication: {e}")
            raise

    def check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def check_ffmpeg(self) -> bool:
        """Check if FFmpeg is installed and accessible"""
        try:
            logging.info("Testing FFmpeg with paths:")
            logging.info(f"Converter: {AudioSegment.converter}")

            # Create temp directory in current working directory
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
                if path and not os.path.exists(path):
                    logging.error(f"File does not exist: {path}")
            return False

    def setup_gui(self):
        """Set up the GUI components"""
        ctk.set_appearance_mode("grey")
        ctk.set_default_color_theme("blue")

        self.app = ctk.CTk()
        self.app.geometry("600x800")
        self.app.title("Multimodal Verification System")

        # Main title
        self.title_label = ctk.CTkLabel(
            self.app,
            text="Multimodal Verification System",
            font=("Arial", 24)
        )
        self.title_label.pack(pady=20)

        # Face verification frame
        self.face_frame = ctk.CTkFrame(self.app)
        self.face_frame.pack(pady=10, padx=20, fill="x")

        self.face_title = ctk.CTkLabel(
            self.face_frame,
            text="Face Verification",
            font=("Arial", 18)
        )
        self.face_title.pack(pady=10)

        self.face_upload_btn = ctk.CTkButton(
            self.face_frame,
            text="Upload Image",
            command=self.upload_face_image
        )
        self.face_upload_btn.pack(pady=5)

        self.face_camera_btn = ctk.CTkButton(
            self.face_frame,
            text="Capture from Camera",
            command=self.capture_face_image
        )
        self.face_camera_btn.pack(pady=5)

        self.face_image_label = ctk.CTkLabel(self.face_frame)
        self.face_image_label.pack(pady=10)

        self.face_status_label = ctk.CTkLabel(
            self.face_frame,
            text="Ready for face verification",
            font=("Arial", 14)
        )
        self.face_status_label.pack(pady=5)

        # Voice verification frame
        self.voice_frame = ctk.CTkFrame(self.app)
        self.voice_frame.pack(pady=10, padx=20, fill="x")

        self.voice_title = ctk.CTkLabel(
            self.voice_frame,
            text="Voice Verification",
            font=("Arial", 18)
        )
        self.voice_title.pack(pady=10)

        self.voice_upload_btn = ctk.CTkButton(
            self.voice_frame,
            text="Upload Audio",
            command=self.upload_voice_audio
        )
        self.voice_upload_btn.pack(pady=5)

        self.voice_record_btn = ctk.CTkButton(
            self.voice_frame,
            text=f"Record Voice ({self.record_duration}s)",
            command=self.record_voice_audio
        )
        self.voice_record_btn.pack(pady=5)

        self.voice_progress = ctk.CTkProgressBar(self.voice_frame)
        self.voice_progress.pack(pady=10)
        self.voice_progress.set(0)

        self.voice_status_label = ctk.CTkLabel(
            self.voice_frame,
            text="Ready for voice verification",
            font=("Arial", 14)
        )
        self.voice_status_label.pack(pady=5)

        # Verification policy
        self.policy_frame = ctk.CTkFrame(self.app)
        self.policy_frame.pack(pady=10, padx=20, fill="x")

        self.policy_label = ctk.CTkLabel(
            self.policy_frame,
            text="Verification Policy:",
            font=("Arial", 16)
        )
        self.policy_label.pack(pady=5, side="left")

        self.policy_var = ctk.StringVar(value="any")

        self.policy_dropdown = ctk.CTkOptionMenu(
            self.policy_frame,
            values=["any", "all", "face_only", "voice_only"],
            variable=self.policy_var,
            command=self.update_policy
        )
        self.policy_dropdown.pack(pady=5, side="left", padx=10)

        # Verify button and result
        self.verify_btn = ctk.CTkButton(
            self.app,
            text="Verify",
            font=("Arial", 18),
            height=40,
            command=self.verify_user
        )
        self.verify_btn.pack(pady=20)

        self.result_frame = ctk.CTkFrame(self.app)
        self.result_frame.pack(pady=10, padx=20, fill="x")

        self.result_title = ctk.CTkLabel(
            self.result_frame,
            text="Verification Results",
            font=("Arial", 18)
        )
        self.result_title.pack(pady=10)

        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text="",
            font=("Arial", 16),
            wraplength=500
        )
        self.result_label.pack(pady=10)

        # Initialize results
        self.verification_result = VerificationResult()
        self.face_image_path = None
        self.voice_audio_path = None

    def update_policy(self, choice):
        """Update the verification policy"""
        self.verification_policy = choice

    def update_face_status(self, message: str, is_error: bool = False):
        """Update the face status display"""
        self.face_status_label.configure(
            text=message,
            text_color="red" if is_error else "white"
        )
        self.app.update()

    def update_voice_status(self, message: str, is_error: bool = False):
        """Update the voice status display"""
        self.voice_status_label.configure(
            text=message,
            text_color="red" if is_error else "white"
        )
        self.app.update()

    def update_result(self, message: str, is_error: bool = False):
        """Update the result display"""
        self.result_label.configure(
            text=message,
            text_color="red" if is_error else "white"
        )
        self.app.update()

    # Face verification methods
    def upload_face_image(self):
        """Open file dialog to select an image for face verification"""
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )

        if file_path and os.path.exists(file_path):
            self.face_image_path = file_path
            self.update_face_status(f"Image selected: {os.path.basename(file_path)}")

            # Display the selected image
            try:
                pil_image = Image.open(file_path)
                displayed_image = ctk.CTkImage(pil_image, size=(200, 200))
                self.face_image_label.configure(image=displayed_image)
                self.face_image_label.image = displayed_image  # Keep a reference
            except Exception as e:
                self.update_face_status(f"Error loading image: {str(e)}", True)
        else:
            self.update_face_status("No valid image selected", True)

    def capture_face_image(self):
        """Capture face image from camera"""
        self.update_face_status("Opening camera...")

        # Disable buttons during capture
        self.face_upload_btn.configure(state="disabled")
        self.face_camera_btn.configure(state="disabled")

        def capture_thread():
            try:
                # Open the webcam
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

                if not cap.isOpened():
                    self.update_face_status("Unable to access camera", True)
                    return

                self.update_face_status("Press 'C' to capture, or 'Q' to quit")

                # Create temp directory if it doesn't exist
                temp_dir = os.path.join(os.getcwd(), 'temp')
                os.makedirs(temp_dir, exist_ok=True)

                # Prepare captured file path
                captured_file_path = os.path.join(temp_dir, "captured_face.jpg")

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        self.update_face_status("Failed to capture frame", True)
                        break

                    # Display the frame
                    cv2.imshow("Camera - Press 'C' to capture, or 'Q' to quit", frame)

                    # Wait for input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("c"):  # 'c' key to capture
                        cv2.imwrite(captured_file_path, frame)
                        cap.release()
                        cv2.destroyAllWindows()

                        self.face_image_path = captured_file_path
                        self.update_face_status("Image captured successfully")

                        # Display the captured image
                        try:
                            pil_image = Image.open(captured_file_path)
                            displayed_image = ctk.CTkImage(pil_image, size=(200, 200))
                            self.face_image_label.configure(image=displayed_image)
                            self.face_image_label.image = displayed_image  # Keep a reference
                        except Exception as e:
                            self.update_face_status(f"Error loading captured image: {str(e)}", True)

                        break
                    elif key == ord("q"):  # 'q' key to quit
                        cap.release()
                        cv2.destroyAllWindows()
                        self.update_face_status("Camera capture cancelled")
                        break

            except Exception as e:
                self.update_face_status(f"Camera error: {str(e)}", True)
                logging.error(f"Camera error: {e}")

            finally:
                # Re-enable buttons
                self.face_upload_btn.configure(state="normal")
                self.face_camera_btn.configure(state="normal")

        # Start capture in a separate thread
        threading.Thread(target=capture_thread).start()

    def verify_face(self) -> bool:
        """Verify face using the selected image"""
        if not self.face_image_path or not os.path.exists(self.face_image_path):
            self.update_face_status("No valid face image selected", True)
            return False

        try:
            self.update_face_status("Verifying face...")

            # Create a search request with the image
            search_request = SearchRequest([self.face_image_path])
            results = self.face_sdk.search.search(search_request)

            if results:
                # Get the best match and score
                best_match = results[0].person.name
                score = results[0].score if hasattr(results[0], 'score') else 0.0

                # Update verification result
                self.verification_result.face_verified = True
                self.verification_result.face_person = best_match
                self.verification_result.face_score = score

                self.update_face_status(f"Face verified: {best_match}")
                return True
            else:
                self.verification_result.face_verified = False
                self.verification_result.face_person = None
                self.verification_result.face_score = 0.0

                self.update_face_status("No face match found", True)
                return False

        except Exception as e:
            self.update_face_status(f"Face verification error: {str(e)}", True)
            logging.error(f"Face verification error: {e}")

            self.verification_result.face_verified = False
            self.verification_result.face_person = None
            self.verification_result.face_score = 0.0

            return False

    # Voice verification methods
    def upload_voice_audio(self):
        """Handle audio file upload for voice verification"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Audio File",
                filetypes=[("Audio Files", "*.wav;*.mp3;*.m4a")]
            )

            if file_path and os.path.exists(file_path):
                self.voice_audio_path = file_path
                self.update_voice_status(f"Audio selected: {os.path.basename(file_path)}")
            else:
                self.update_voice_status("No valid audio file selected", True)

        except Exception as e:
            self.update_voice_status(f"Error selecting audio file: {str(e)}", True)
            logging.error(f"Audio file selection error: {e}")

    def record_voice_audio(self):
        """Record audio for voice verification"""
        self.voice_record_btn.configure(state="disabled")
        self.voice_upload_btn.configure(state="disabled")

        def record_thread():
            try:
                self.update_voice_status("Recording...")

                # Record audio
                audio = sd.rec(
                    int(self.record_duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32"
                )

                # Show recording progress
                for i in range(self.record_duration * 10):
                    self.voice_progress.set((i + 1) / (self.record_duration * 10))
                    self.app.update()
                    sd.sleep(100)  # Sleep for 100ms

                sd.wait(100)

                # Create temp directory if it doesn't exist
                temp_dir = os.path.join(os.getcwd(), 'temp')
                os.makedirs(temp_dir, exist_ok=True)

                # Save to temp file
                temp_file_path = os.path.join(temp_dir, "recorded_voice.wav")
                write(temp_file_path, self.sample_rate, (audio * 32767).astype(np.int16))

                self.voice_audio_path = temp_file_path
                self.update_voice_status("Voice recording completed")

            except Exception as e:
                self.update_voice_status(f"Recording failed: {str(e)}", True)
                logging.error(f"Voice recording failed: {e}")

            finally:
                self.voice_progress.set(0)
                self.voice_record_btn.configure(state="normal")
                self.voice_upload_btn.configure(state="normal")

        # Start recording in a separate thread
        threading.Thread(target=record_thread).start()

    def convert_audio_to_wav(self, file_path: str) -> Optional[str]:
        """Convert audio file to WAV format using FFmpeg"""
        try:
            # Check if FFmpeg is available
            if not self.check_ffmpeg():
                self.update_voice_status("FFmpeg not found. Please install FFmpeg to process audio files", True)
                logging.error("FFmpeg not installed or configured properly")
                return None

            # Get file extension
            _, file_ext = os.path.splitext(file_path)
            file_ext = file_ext.lower()

            # If already WAV, return the original path
            if file_ext == '.wav':
                return file_path

            # Create temp directory
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Create WAV path
            wav_path = os.path.join(temp_dir, os.path.basename(file_path).replace(file_ext, ".wav"))

            # Log conversion details
            logging.info(f"Converting {file_ext[1:].upper()} to WAV: {file_path} -> {wav_path}")
            self.update_voice_status(f"Converting {file_ext[1:].upper()} to WAV format...")

            try:
                # Load audio file with PyDub
                audio = AudioSegment.from_file(file_path, format=file_ext[1:])

                # Export to WAV with specific settings for better compatibility
                audio = audio.set_channels(1)  # Convert to mono
                audio = audio.set_frame_rate(self.sample_rate)  # Set sample rate
                audio.export(wav_path, format="wav")

                logging.info(f"Successfully converted {file_ext[1:].upper()} to WAV")
                return wav_path

            except Exception as e:
                error_msg = f"Error converting {file_ext[1:].upper()} file: {str(e)}"
                self.update_voice_status(error_msg, True)
                logging.error(error_msg, exc_info=True)
                return None

        except Exception as e:
            self.update_voice_status(f"Audio conversion error: {str(e)}", True)
            logging.error(f"Audio conversion error: {e}", exc_info=True)
            return None

    def preprocess_voice_audio(self, file_path: str) -> Optional[np.ndarray]:
        """Preprocess audio file for voice verification"""
        try:
            logging.info(f"Preprocessing file: {file_path}")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")

            # Convert the audio file to WAV if needed
            _, file_ext = os.path.splitext(file_path)
            file_ext = file_ext.lower()

            if file_ext != '.wav':
                converted_path = self.convert_audio_to_wav(file_path)
                if not converted_path:
                    return None
                file_path = converted_path

            import librosa
            wav, _ = librosa.load(file_path, sr=self.sample_rate)
            preprocessed_wav = preprocess_wav(wav)

            return preprocessed_wav

        except Exception as e:
            self.update_voice_status(f"Preprocessing failed: {str(e)}", True)
            logging.error(f"Voice preprocessing failed: {e}", exc_info=True)
            return None

    def verify_voice(self) -> bool:
        """Verify voice using the selected or recorded audio"""
        if not self.voice_audio_path or not os.path.exists(self.voice_audio_path):
            self.update_voice_status("No valid voice audio selected", True)
            return False

        try:
            self.update_voice_status("Preprocessing voice audio...")

            # Preprocess the audio
            preprocessed_wav = self.preprocess_voice_audio(self.voice_audio_path)
            if preprocessed_wav is None:
                return False

            self.update_voice_status("Verifying voice...")

            # Create embedding and search
            voice_embedding = self.voice_encoder.embed_utterance(preprocessed_wav)
            results = self.voice_client.search(
                collection_name="my-collection",
                query_vector=voice_embedding.tolist(),
                limit=1
            )

            if results and results[0].score >= self.voice_threshold:
                # Update verification result
                self.verification_result.voice_verified = True
                self.verification_result.voice_person = results[0].payload.get('speaker', 'Unknown')
                self.verification_result.voice_score = results[0].score

                self.update_voice_status(f"Voice verified: {self.verification_result.voice_person}")
                return True
            else:
                score = results[0].score if results else 0.0

                self.verification_result.voice_verified = False
                self.verification_result.voice_person = None
                self.verification_result.voice_score = score

                self.update_voice_status(f"Voice not verified (score: {score:.2f})", True)
                return False

        except Exception as e:
            self.update_voice_status(f"Voice verification error: {str(e)}", True)
            logging.error(f"Voice verification error: {e}")

            self.verification_result.voice_verified = False
            self.verification_result.voice_person = None
            self.verification_result.voice_score = 0.0

            return False

    def verify_user(self):
        """Perform complete user verification using both modalities"""
        # Reset previous results
        self.verification_result = VerificationResult()

        # Update result status
        self.update_result("Verifying user...")

        # Verify face if image is selected
        if self.face_image_path:
            self.verify_face()

        # Verify voice if audio is selected
        if self.voice_audio_path:
            self.verify_voice()

        # Check if verification passes based on policy
        is_verified = self.verification_result.is_verified(self.verification_policy)

        # Update result display
        if is_verified:
            self.update_result(
                f"✓ VERIFICATION PASSED (Policy: {self.verification_policy})\n\n"
                f"{self.verification_result.get_summary()}"
            )
        else:
            self.update_result(
                f"✗ VERIFICATION FAILED (Policy: {self.verification_policy})\n\n"
                f"{self.verification_result.get_summary()}",
                is_error=True
            )

    def run(self):
        """Start the application"""
        self.app.mainloop()


def main():
    try:
        system = MultimodalVerificationSystem()
        system.run()
    except Exception as e:
        logging.error(f"Application failed to start: {e}")
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()