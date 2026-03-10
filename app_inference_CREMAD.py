import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import librosa
import cv2
import random
import os
import subprocess
import json
from PIL import Image

from model.AudioVideo import AVGBShareClassifier
from data.template import config
from utils.utils import deep_update_dict


emotion_map = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Fear",
    4: "Disgust",
    5: "Angry"
}


# --------------------------------------------------
# Extract audio
# --------------------------------------------------

def extract_audio(video_path, output_wav):

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "22050",
        output_wav
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# --------------------------------------------------
# Extract frames
# --------------------------------------------------

def extract_frames(video_path, frame_dir):

    os.makedirs(frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(fps),1)

    count = 0
    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if count % frame_interval == 0:
            cv2.imwrite(f"{frame_dir}/frame_{frame_id}.jpg", frame)
            frame_id += 1

        count += 1

    cap.release()


# --------------------------------------------------
# Audio preprocessing
# --------------------------------------------------

def process_audio(wav_path):

    samples, rate = librosa.load(wav_path, sr=22050)

    resamples = np.tile(samples, 20)[:22050 * 20]
    resamples[resamples > 1.] = 1.
    resamples[resamples < -1.] = -1.

    spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)

    spectrogram = np.log(np.abs(spectrogram) + 1e-7)

    spectrogram = torch.tensor(spectrogram).unsqueeze(0).unsqueeze(0).float()

    return spectrogram


# --------------------------------------------------
# Frame preprocessing
# --------------------------------------------------

def process_frame(frame_dir):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])

    frames = os.listdir(frame_dir)

    frame_path = os.path.join(frame_dir, random.choice(frames))

    img = Image.open(frame_path).convert("RGB")

    img = transform(img)

    images = torch.zeros((1,3,224,224))
    images[0] = img

    images = torch.permute(images,(1,0,2,3)).unsqueeze(0)

    return images


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):

    cfg = config

    with open("data/crema.json","r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)

    model = AVGBShareClassifier(config=cfg).cuda()

    state_dict = torch.load(model_path, map_location="cuda")

    model.load_state_dict(state_dict, strict=False)

    model.eval()

    return model


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict(video_file, model):

    os.makedirs("temp", exist_ok=True)

    video_path = f"temp/{video_file.name}"
    wav_path = "temp/audio.wav"
    frame_dir = "temp/frames"

    with open(video_path, "wb") as f:
        f.write(video_file.read())

    extract_audio(video_path, wav_path)

    extract_frames(video_path, frame_dir)

    spectrogram = process_audio(wav_path).cuda()

    images = process_frame(frame_dir).cuda()

    with torch.no_grad():

        o_a, o_v = model(spectrogram, images)

        out_a,_,_ = model.classfier(o_a, is_a=True)
        out_v,_,_ = model.classfier(o_v, is_a=False)

        out = 0.4*out_a + 0.6*out_v

        probs = F.softmax(out, dim=1)

        pred = torch.argmax(probs, dim=1).item()

    return pred, probs


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("🎭 Multimodal Emotion Recognition")

st.write("Upload a video (.mp4 or .flv) and the model will predict the emotion.")

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4","flv"]
)

model_path = "crema_GB_best_model_0.8183.pth"

model = load_model(model_path)


if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("Predict Emotion"):

        with st.spinner("Processing video..."):

            pred, probs = predict(uploaded_file, model)

        st.success(f"Predicted Emotion: **{emotion_map[pred]}**")

        st.subheader("Prediction Probabilities")

        for i,p in enumerate(probs[0]):
            st.write(f"{emotion_map[i]} : {p.item():.3f}")