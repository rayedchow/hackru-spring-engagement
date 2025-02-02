import nods
import emotion
import numpy as np

def calculate_engagement(filePath):
    base_engagement = emotion.analyze_emotions(filePath)
    nod_calculations = nods.analyze_nods(filePath)
    
    # Create a combined engagement history
    combined_history = []
    
    # Process each emotion entry
    for entry in base_engagement['engagement_history']:
        timestamp = entry['timestamp']
        base_score = entry['score']
        num_people = entry['num_people']
        
        # Check for nods at this timestamp
        nod_boost = 0
        for nod_time in nod_calculations['nod_history']:
            # If nod occurred within 1 second of this timestamp
            if abs(nod_time - timestamp) <= 1:
                # Calculate boost based on one person being fully engaged
                if num_people > 0:
                    individual_max = 100  # Max score for the nodding person
                    others_base = base_score * (num_people - 1)  # Original scores for others
                    group_score = (individual_max + others_base) / num_people
                    nod_boost = group_score - base_score
        
        # Combine scores (cap at 100)
        combined_score = min(100, base_score + nod_boost)

        combined_history.append({
            'timestamp': timestamp,
            'base_score': base_score,
            'nod_boost': round(nod_boost, 2),
            'final_score': round(combined_score, 2),
            'num_people': num_people
        })
    
    # Calculate aggregate statistics
    final_scores = [entry['final_score'] for entry in combined_history]
    
    return {
        'combined_history': combined_history,
        'average_score': np.mean(final_scores) if final_scores else 0,
        'max_score': max(final_scores) if final_scores else 0,
        'min_score': min(final_scores) if final_scores else 0,
        'total_nods': nod_calculations['nod_count'],
        'average_people': base_engagement['average_people']
    }

if __name__ == "__main__":
    filePath = input("file path: ")
    results = calculate_engagement(filePath)
    
    print("\nEngagement Analysis Results:")
    for entry in results['combined_history']:
        minutes = int(entry['timestamp'] // 60)
        seconds = int(entry['timestamp'] % 60)
        print(f"Time {minutes:02d}:{seconds:02d} - Score: {entry['final_score']} "
              f"(Base: {entry['base_score']} + Nod: {entry['nod_boost']})")
    
    print("\nSummary Statistics:")
    print(f"Average Score: {results['average_score']:.2f}")
    print(f"Maximum Score: {results['max_score']}")
    print(f"Minimum Score: {results['min_score']}")
    print(f"Total Nods: {results['total_nods']}")
    print(f"Average People Detected: {results['average_people']}")