from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import logging
import numpy as np
from multimodal import VerificationResult
from opencv.fr import FR
from opencv.fr.search.schemas import SearchRequest
from opencv.fr.persons.schemas import PersonBase
from resemblyzer import preprocess_wav, VoiceEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from pydub import AudioSegment
from pydub.utils import which
import librosa
from flask_cors import CORS



app = Flask(__name__)

CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)

def check_cuda() -> bool:
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

# Minimal verification system without GUI
developer_key = "7BRQ8i1YWQ2MWM4NjYtMDUyNy00ZTEzLThmZGItZmZmOWRhNjE0ZWFj"
backend_url = "https://eu.opencv.fr"
face_threshold = 0.8
voice_threshold = 0.7
sample_rate = 16000
collection_name = "my-collection"

face_sdk = FR(backend_url, developer_key)
voice_encoder = VoiceEncoder("cuda" if check_cuda() else "cpu")
voice_client = QdrantClient(
    url="https://1c94b6cb-537b-4b28-9fbb-934eb3f2c6d8.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ1ODgzNjI3fQ.7ueosobxV8059GgQG3Qal6S1idDFauCkaCvDHLDv-n0"
)

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffmpeg = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

def convert_to_wav(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".wav":
        return file_path
    temp_path = file_path.replace(ext, ".wav")
    audio = AudioSegment.from_file(file_path, format=ext[1:])
    audio = audio.set_channels(1).set_frame_rate(sample_rate)
    audio.export(temp_path, format="wav")
    return temp_path

def verify_face(image_path, result: VerificationResult):
    try:
        search_request = SearchRequest([image_path])
        results = face_sdk.search.search(search_request)
        if results:
            best_match = results[0].person.name
            score = results[0].score
            result.face_verified = score >= face_threshold
            result.face_person = best_match
            result.face_score = score
        else:
            result.face_verified = False
    except Exception as e:
        logging.error(f"Face verification error: {e}")
        result.face_verified = False

def verify_voice(audio_path, result: VerificationResult):
    try:
        wav_path = convert_to_wav(audio_path)
        wav, _ = librosa.load(wav_path, sr=sample_rate)
        preprocessed = preprocess_wav(wav)
        embedding = voice_encoder.embed_utterance(preprocessed)
        results = voice_client.search(collection_name=collection_name, query_vector=embedding.tolist(), limit=1)
        if results and results[0].score >= voice_threshold:
            result.voice_verified = True
            result.voice_person = results[0].payload.get("speaker")
            result.voice_score = results[0].score
        else:
            result.voice_verified = False
            result.voice_score = results[0].score if results else 0.0
    except Exception as e:
        logging.error(f"Voice verification error: {e}")
        result.voice_verified = False

@app.route("/verify", methods=["POST"])
def verify():
    try:
        face_file = request.files.get("face")
        voice_file = request.files.get("voice")
        policy = request.form.get("policy", "any")

        if not face_file and not voice_file:
            return jsonify({"error": "At least one modality (face or voice) is required."}), 400

        result = VerificationResult()

        if face_file:
            face_path = os.path.join(UPLOAD_FOLDER, f"face_{uuid.uuid4().hex}_{secure_filename(face_file.filename)}")
            face_file.save(face_path)
            verify_face(face_path, result)

        if voice_file:
            voice_path = os.path.join(UPLOAD_FOLDER, f"voice_{uuid.uuid4().hex}_{secure_filename(voice_file.filename)}")
            voice_file.save(voice_path)
            verify_voice(voice_path, result)

        verified = result.is_verified(policy)

        return jsonify({
            "verified": verified,
            "policy": policy,
            "face_verified": result.face_verified,
            "face_person": result.face_person,
            "face_score": result.face_score,
            "voice_verified": result.voice_verified,
            "voice_person": result.voice_person,
            "voice_score": result.voice_score
        })

    except Exception as e:
        logging.exception("Verification failed")
        return jsonify({"error": str(e)}), 500

@app.route("/register/face", methods=["POST"])
def register_face():
    try:
        name = request.form.get("name")
        image = request.files.get("images")
        if not name or not image:
            return jsonify({"error": "Name and image are required"}), 400

        image_paths = []
        path = os.path.join(UPLOAD_FOLDER, f"face_reg_{uuid.uuid4().hex}_{secure_filename(image.filename)}")
        image.save(path)
        image_paths.append(path)

        person = PersonBase(images=image_paths, name=name)
        face_sdk.persons.create(person)

        logging.info(f"saved image path: {path}")
        return jsonify({"message": f"Face registered for {name}"})
    except Exception as e:
        logging.exception("Face registration failed")
        return jsonify({"error": str(e)}), 500

@app.route("/register/voice", methods=["POST"])
def register_voice():
    try:
        name = request.form.get("name")
        audio = request.files.get("audio")
        if not name or not audio:
            return jsonify({"error": "Name and audio are required"}), 400

        audio_path = os.path.join(UPLOAD_FOLDER, f"voice_reg_{uuid.uuid4().hex}_{secure_filename(audio.filename)}")
        audio.save(audio_path)
        wav_path = convert_to_wav(audio_path)
        wav, _ = librosa.load(wav_path, sr=sample_rate)
        preprocessed = preprocess_wav(wav)
        embedding = voice_encoder.embed_utterance(preprocessed)

        # Ensure collection exists
        try:
            voice_client.get_collection(collection_name)
        except:
            voice_client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=256, distance=Distance.COSINE))

        voice_client.upsert(collection_name=collection_name, points=[{
            "id": uuid.uuid4().int >> 64,
            "vector": embedding.tolist(),
            "payload": {"speaker": name}
        }])

        return jsonify({"message": f"Voice registered for {name}"})
    except Exception as e:
        logging.exception("Voice registration failed")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)