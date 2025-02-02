import cv2
import mediapipe as mp
import numpy as np

def analyze_nods(video_path, frame_interval=30, nod_threshold=0.01):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
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
    
    nod_count = 0
    last_y = None
    last_state = False
    frame_count = 0
    nod_history = []
    
    print(f"Starting nod analysis of {total_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue
            
        print(f"\rProcessing frame {frame_count}/{total_frames}", end="")
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            nose_y = face_landmarks.landmark[4].y
            print(nose_y)
            print(last_y)
            
            if last_y is not None:
                movement = nose_y - last_y
                print(movement)
                is_nodding = abs(movement) > nod_threshold
                
                if is_nodding and movement > 0 and not last_state:
                    nod_count += 1
                    last_state = True
                    timestamp = frame_count / fps
                    nod_history.append(timestamp)
                elif not is_nodding:
                    last_state = False
            
            last_y = nose_y
    
    cap.release()
    
    return {
        'nod_count': nod_count,
        'nod_history': nod_history,
        'total_duration': frame_count / fps,
        'fps': fps
    }

# Example usage
if __name__ == "__main__":
    video_path = "IMG_6908.MOV"
    results = analyze_nods(video_path)
    
    if results['nod_history']:
        print("\nNodding Timeline:")
        for i, timestamp in enumerate(results['nod_history']):
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            print(f"Nod {i+1} at {minutes:02d}:{seconds:02d}")

        print("\nSummary Statistics:")
        print(f"Total Nods: {results['nod_count']}")
        if len(results['nod_history']) > 1:
            nod_intervals = np.diff(results['nod_history'])
            print(f"Average time between nods: {np.mean(nod_intervals):.2f} seconds")
            print(f"Shortest interval: {np.min(nod_intervals):.2f} seconds")
            print(f"Longest interval: {np.max(nod_intervals):.2f} seconds")