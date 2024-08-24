import streamlit as st
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
from difflib import SequenceMatcher
from io import BytesIO
from st_audiorec import st_audiorec

# Load the pretrained Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Function to load and process the audio file
def load_audio(audio_bytes):
    # Convert bytes to a file-like object
    audio_file = BytesIO(audio_bytes)
    # Load the audio file using torchaudio
    waveform, sample_rate = torchaudio.load(audio_file, format="wav")
    
    # If stereo, convert to mono by averaging the channels
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)
    
    return waveform, sample_rate

# Function to transcribe audio file to text
def transcribe_audio(audio_bytes):
    waveform, sample_rate = load_audio(audio_bytes)
    
    # Resample if the sample rate is not 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Ensure the waveform is of shape [sequence_length]
    waveform = waveform.squeeze()  # Remove all dimensions with size 1
    
    # Prepare the input for the model
    input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
    
    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
    # Decode the predicted IDs into text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    
    return transcription

# Function to calculate the similarity score
def calculate_similarity(transcript, correct_answer):
    # Calculate the similarity using SequenceMatcher
    similarity = SequenceMatcher(None, transcript.upper(), correct_answer.upper()).ratio()
    return similarity * 100  # Convert to percentage

# Streamlit App
st.title("AI口語老師")
correct_answer = "HOW IS THE WEATHER. IT IS SUNNY"
st.write(correct_answer)
# Record audio using st_audiorec
audio_file = st_audiorec()

if audio_file is not None:
    
    if st.button("繳交"):
        transcript = transcribe_audio(audio_file)
        st.write(f"AI老師聽到的是: **{transcript}**")

        
        # Calculate and display the similarity score
        similarity_score = calculate_similarity(transcript, correct_answer)
        #st.write(f"Similarity Score: **{similarity_score:.2f}%**")
        if similarity_score>=70:
            st.write(f"評語:你很棒! 拿到**{similarity_score:.2f}分**")
        elif similarity_score>=40 and similarity_score<70:
            st.write(f"評語:還不錯! 拿到**{similarity_score:.2f}分**")
        elif similarity_score<40:
            st.write(f"評語:再加油! 拿到**{similarity_score:.2f}分**")