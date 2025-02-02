import sys
import torch
import ssl
import ffmpeg
import os
from tempfile import NamedTemporaryFile
import subprocess
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


# Load the Whisper model using transformers
print("Loading Whisper model...")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# Create unverified SSL context to handle certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

def extract_audio(video_path):
    try:
        # Create a temporary file for the extracted audio
        with NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        print(f"Input video path: {video_path}")
        print(f"Output audio path: {temp_audio_path}")
        
        # Extract audio using ffmpeg command with more detailed options
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-ar', '44100',
            '-ac', '2',
            '-b:a', '192k',
            '-f', 'mp3',
            '-y',
            temp_audio_path
        ]
        
        # Run ffmpeg with error output
        process = subprocess.run(
            command,
            capture_output=True,
            text=True
        )
        
        if process.returncode != 0:
            print(f"FFmpeg stderr: {process.stderr}")
            raise Exception(f"FFmpeg failed with return code {process.returncode}")

        return temp_audio_path
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        if 'process' in locals() and process.stderr:
            print(f"FFmpeg error output: {process.stderr}")
        return None

def transcribe_audio(file_path):
    try:
        video_extensions = ('.mp4', '.webm', '.mov')
        is_video = file_path.lower().endswith(video_extensions)
        audio_path = extract_audio(file_path) if is_video else file_path

        if is_video and not audio_path:
            return False

        # Load and process audio
        print(f"Transcribing audio...")
        audio_input, sr = librosa.load(audio_path, sr=16000)
        input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features

        # Generate transcription with timestamps
        predicted_ids = model.generate(
            input_features,
            language='en',
            task='transcribe'
        )
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        # Process audio in chunks for timestamps
        chunk_size = sr * 10  # 30 seconds per chunk
        timestamps = []
        
        for i in range(0, len(audio_input), chunk_size):
            chunk = audio_input[i:i+chunk_size]
            chunk_features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features
            chunk_ids = model.generate(
                chunk_features,
                language='en',
                task='transcribe'
            )
            chunk_text = processor.batch_decode(chunk_ids, skip_special_tokens=True)[0]
            
            if chunk_text.strip():  # Only add non-empty transcriptions
                start_time = i / sr
                end_time = min((i + chunk_size) / sr, len(audio_input) / sr)
                timestamps.append({
                    'start': round(start_time, 2),
                    'end': round(end_time, 2),
                    'text': chunk_text.strip()
                })

        # Clean up temporary audio file
        if is_video and os.path.exists(audio_path):
            os.unlink(audio_path)

        return {
            'full_text': transcription[0],
            'segments': timestamps
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python transcribe.py <path_to_audio_file>")
        return

    audio_path = sys.argv[1]
    transcribe_audio(audio_path)

if __name__ == "__main__":
    main()