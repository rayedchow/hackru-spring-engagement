from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import random
from engagement import calculate_engagement
from whisper import transcribe_audio
import cv2

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'webm'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/analyze-engagement")
async def analyze_engagement(video: UploadFile = File(...)):
    print("Received request")
    print("File:", video.filename)
    
    if not video.filename:
        raise HTTPException(status_code=400, detail="No selected file")
    
    if not allowed_file(video.filename):
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
    
    try:
        # Create temporary directory
        temp_dir = os.path.join(os.getcwd(), 'tmp', f'tmpfile_{random.randint(1000, 9999)}')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded file
        temp_path = os.path.abspath(os.path.join(temp_dir, video.filename))
        print(f"Saving file to: {temp_path}")
        
        content = await video.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Validate video
        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
            
        frame_count = 0
        while cap.isOpened():
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1
        cap.release()
        
        print(f"Processing video at: {temp_path} with {frame_count} frames")
        engagement_results = calculate_engagement(temp_path)
        transcription = transcribe_audio(temp_path)
        
        # Analyze engagement levels for each transcription segment
        low_engagement_segments = []
        engagement_threshold = 50  # Define low engagement as below 50
        
        for segment in transcription['segments']:
            # Find engagement scores during this segment
            segment_scores = [
                entry['final_score'] 
                for entry in engagement_results['combined_history']
                if segment['start'] <= entry['timestamp'] <= segment['end']
            ]
            
            # If average engagement during this segment is low, add to list
            if segment_scores:
                avg_segment_engagement = sum(segment_scores) / len(segment_scores)
                if avg_segment_engagement < engagement_threshold:
                    low_engagement_segments.append({
                        'timestamp': f"{int(segment['start']//60)}:{int(segment['start']%60):02d}",
                        'text': segment['text'],
                        'engagement_score': round(avg_segment_engagement, 2)
                    })
        
        # Add transcription analysis to results
        results = {
            **engagement_results,
            'transcription': transcription,
            'low_engagement_segments': low_engagement_segments
        }
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        os.rmdir(temp_dir)
        
        return results
        
    except Exception as e:
        # Cleanup on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)