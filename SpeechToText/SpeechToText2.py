from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sounddevice as sd
from scipy.io.wavfile import write
import noisereduce 
import numpy as np

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

SAMPLE_RATE = 16000
CHUNK_SECONDS = 3
FILENAME = "recording.wav"

def save_wav(audio, filename):
    """Save audio as a WAV file"""
    audio_int16 = (audio * 32767).astype('int16')
    write(filename, SAMPLE_RATE, audio_int16)

def reduce_noise(audio):
    """Reduce background noise"""
    return noisereduce.reduce_noise(y=audio, sr=SAMPLE_RATE)

def record():
    print("Speak ")
    transcription_list = []
    full_rec = []

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
            while True:
                audio_chunk, _ = stream.read(CHUNK_SECONDS * SAMPLE_RATE)
                audio = audio_chunk.flatten()
                denoised_audio = reduce_noise(audio)
                full_rec.append(audio) 
                input_features = processor(denoised_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
                predicted_ids = model.generate(input_features)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                transcription_list.append(transcription)
                combined_transcription = ' '.join(transcription_list).strip()
                print(f"\rTranscription: {combined_transcription}", end='')
    except KeyboardInterrupt:
        print("\nRecording stopped.")
        full_rec_list = np.concatenate(full_rec)
        save_wav(full_rec_list, FILENAME)

if __name__ == "__main__":
    record()
