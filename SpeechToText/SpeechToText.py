from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sounddevice as sd

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")  
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

SAMPLE_RATE = 16000  
CHUNK_SECONDS = 3    

def record():
    print("Speak")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
            while True:
              
                audio_chunk, _ = stream.read(CHUNK_SECONDS * SAMPLE_RATE)
                audio = audio_chunk.flatten()
                input_features = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
                predicted_ids = model.generate(input_features)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

                print("Transcription:", transcription)

    except KeyboardInterrupt:
        print("\nRecording stopped.")

if __name__ == "__main__":
    record()
