import streamlit as st
import os
import io
import time
import sounddevice as sd
import soundfile as sf
from detect import DetectStuttering
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(wav_file_path):
    # Load the audio file
    y, sr = librosa.load(wav_file_path)

    # Compute the spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    return plt

def display_result(result, show_spectrogram):
    st.subheader("Analysis Result")
    print(result)
    if "no stutter" in result:
        st.success(result)
    else:
        # Extracting detailed information
        lines = result.splitlines()
        
        stuttering_seconds = lines[2].split(":")[1].strip()
        stuttering_percentage = lines[3].split(":")[1].strip()
        stuttered_chars = lines[4].split(":")[1].strip()
        transcription = (lines[5].split(":")[1].strip()).capitalize()
        transcript_without_stuttering = (lines[6].split(":")[1].strip()).capitalize()
        st.success("Stuttering Detected")
        st.code(
        f"Stuttered Chars: {stuttered_chars}\n"
        # f"Stuttering Seconds: {stuttering_seconds}\n"
        f"Transcription: {transcription}\n"
        f"Transcript Without Stuttering: {transcript_without_stuttering}")

def record_voice():
    fs = 16000
    duration = 5
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    sf.write('audio_file.wav', myrecording, fs)

def detect_stuttering(uploaded_file, text, show_spectrogram):
    if uploaded_file:
        file_path_audio = "audio_file.wav"
        
        # Use BytesIO to handle the uploaded file
        uploaded_file.seek(0)
        audio_data, fs = sf.read(io.BytesIO(uploaded_file.read()))

        sf.write(file_path_audio, audio_data, fs)
        output = DetectStuttering(file_path_audio, text).predict_megamgem()
        display_result(output, show_spectrogram)
    else:
        # Handle the case when audio is recorded instead of uploaded
        output = DetectStuttering('audio_file.wav', text).predict_megamgem()
        display_result(output, show_spectrogram)

def main():
    st.title("Stutter Detection 🗣️")
    # Reference Text Section
    st.subheader("Reference Text")
    ref_text = st.text_area("Enter Reference Text", height=1, max_chars=200)

    # Voice Recorder Section
    st.sidebar.subheader("Voice Recorder")
    option = st.sidebar.selectbox("Select Audio Source", ["Record Voice", "Upload Audio File (WAV format)"])
    show_spectrogram = st.sidebar.checkbox("Show Spectrogram", value=True)  # Checkbox to decide whether to show the spectrogram or not

    if option == "Record Voice":
        recording_message = st.empty()
        if st.sidebar.button("Record Voice"):
            recording_message.text("Recording...")
            record_voice()
            recording_message.text("Recording Complete.")

            # Display spectrogram after recording
            if show_spectrogram:
                st.subheader("Spectrogram")
                plt_spectrogram = plot_spectrogram("audio_file.wav")
                st.pyplot(plt_spectrogram)

    elif option == "Upload Audio File (WAV format)":
        uploaded_file = st.sidebar.file_uploader("Upload Audio File (WAV format)", type=["wav"])
        recording_message = None  # No recording message for file upload

        # Display spectrogram after file upload
        if uploaded_file and show_spectrogram:
            st.subheader("Spectrogram")
            plt_spectrogram = plot_spectrogram(uploaded_file)
            st.pyplot(plt_spectrogram)

    # Results Section
    if st.button("Detect"):
        if ref_text and (option == "Record Voice" or uploaded_file is not None):
            # Add loading spinner while processing
            with st.spinner("Detecting..."):
                detect_stuttering(uploaded_file, ref_text, show_spectrogram) if option == "Upload Audio File (WAV format)" else detect_stuttering(None, ref_text, show_spectrogram)
        else:
            st.warning("Please provide both target text and audio file.")

if __name__ == "__main__":
    main()
