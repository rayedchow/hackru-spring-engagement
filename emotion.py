import cv2
from deepface import DeepFace
import numpy as np

def analyze_emotions(video_path, frame_interval=30):
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Count frames first
    total_frames = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        total_frames += 1
    
    # Reset video capture
    cap.release()
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fallback to standard fps
    
    frame_count = 0
    engagement_history = []
    
    print(f"Starting analysis of {total_frames} frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue
            
        print(f"\rProcessing frame {frame_count}", end="")

        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            if isinstance(results, dict):
                results = [results]
            
            total_focus_score = 0
            total_engagement_score = 0
            num_people = len(results)
            
            for result in results:
                emotions_dict = result['emotion']
                
                engagement_score = (
                    emotions_dict['happy'] + 
                    emotions_dict['surprise'] - 
                    emotions_dict['sad'] * 0.5
                )
                
                focus_score = (
                    emotions_dict['neutral'] * 1.5 - 
                    (emotions_dict['angry'] + emotions_dict['fear'] + emotions_dict['surprise']) * 0.5
                )
                
                normalized_focus = min(100, max(0, focus_score))
                normalized_engagement = min(100, max(0, engagement_score * 2))
                
                total_focus_score += normalized_focus
                total_engagement_score += normalized_engagement
                
            if num_people > 0:
                avg_score = int((total_focus_score + total_engagement_score) / (2 * num_people))
                engagement_history.append({
                    'timestamp': frame_count / fps,
                    'score': avg_score,
                    'num_people': num_people
                })
                
        except Exception as e:
            pass

    cap.release()
    
    # Calculate average number of people
    avg_people = np.mean([entry['num_people'] for entry in engagement_history]) if engagement_history else 0
    
    return {
        'engagement_history': engagement_history,
        'total_duration': frame_count / fps,
        'fps': fps,
        'average_score': np.mean([entry['score'] for entry in engagement_history]) if engagement_history else 0,
        'min_score': min([entry['score'] for entry in engagement_history]) if engagement_history else 0,
        'max_score': max([entry['score'] for entry in engagement_history]) if engagement_history else 0,
        'average_people': round(avg_people, 2)
    }

# Update the main section to display the new metric
if __name__ == "__main__":
    video_path = "IMG_6908.MOV"
    results = analyze_emotions(video_path)
    
    print("\nEngagement Analysis Results:")
    for entry in results['engagement_history']:
        minutes = int(entry['timestamp'] // 60)
        seconds = int(entry['timestamp'] % 60)
        print(f"Time {minutes:02d}:{seconds:02d} - Score: {entry['score']} (People: {entry['num_people']})")
    
    print("\nSummary Statistics:")
    print(f"Average Score: {results['average_score']:.2f}")
    print(f"Minimum Score: {results['min_score']}")
    print(f"Maximum Score: {results['max_score']}")
    print(f"Average Number of People: {results['average_people']}")