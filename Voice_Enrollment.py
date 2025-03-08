from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Setup Quadrant connection
client = QdrantClient(
    url="https://1c94b6cb-537b-4b28-9fbb-934eb3f2c6d8.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ1ODgzNjI3fQ.7ueosobxV8059GgQG3Qal6S1idDFauCkaCvDHLDv-n0",
)

# Function to check if the collection exists
def collection_exists(client, collection_name):
    try:
        # Attempt to get collection info
        client.get_collection(collection_name)
        return True
    except:
        return False

# Check if collection exists, if not, create it
collection_name = "my-collection"
if not collection_exists(client, collection_name):
    try:
        client.create_collection(
            collection_name= "my-collection",
            vectors_config=VectorParams(size=256, distance=Distance.COSINE)
        )
        print(f"Collection '{collection_name}' created successfully!")
    except Exception as e:
        print(f"Error creating collection: {e}")
        raise

# Load and process voice files
try:
    encoder = VoiceEncoder("cuda")
except:
    encoder = VoiceEncoder("cpu")

# Ensure the wav files are found
wav_fpaths = list(Path("C:/Users/www13/Downloads/train").rglob("*.mp3" "*.m4a"))
if not wav_fpaths:
    raise FileNotFoundError("No audio files found! Check the file path.")

speakers = [wav.stem for wav in wav_fpaths]

# Preprocess and extract embeddings, applying tqdm with proper settings for Windows terminals
wavs = [preprocess_wav(wav) for wav in tqdm(wav_fpaths, desc="Processing voices", ncols=80)]
embeddings = np.array([encoder.embed_utterance(wav) for wav in wavs])

# Store in Qdrant
points = [{"id": i+1, "vector": emb.tolist(), "payload": {"speaker": speakers[i]}} for i, emb in enumerate(embeddings)]
try:
    client.upsert(collection_name=collection_name, points=points)
    print("Speakers enrolled successfully!")
except Exception as e:
    print(f"Error during upsert: {e}")