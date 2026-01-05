import json
import sys
from collections import Counter

def summarize_jsonl_results(file_path):
    total_count = 0
    scores = []
    verdicts = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue # Skip malformed lines
                
                judge_data = data.get("judge", {})
                
                # Extract metrics
                score = judge_data.get("score")
                verdict = judge_data.get("verdict")
                
                if score is not None:
                    scores.append(score)
                if verdict:
                    verdicts.append(verdict.lower())
                
                total_count += 1

        if total_count == 0:
            print(f"No valid data found in {file_path}.")
            return

        # Calculations
        avg_score = sum(scores) / len(scores) if scores else 0
        verdict_counts = Counter(verdicts)
        accuracy = (verdict_counts.get("correct", 0) / total_count) * 100

        # Display Results
        print("-" * 40)
        print(f"📊 EVALUATION SUMMARY: {file_path}")
        print("-" * 40)
        print(f"Total Samples    : {total_count}")
        print(f"Overall Accuracy : {accuracy:.2f}%")
        print(f"Mean Score (0-5) : {avg_score:.2f}")
        print("-" * 40)
        print("VERDICT BREAKDOWN:")
        for v_type, count in verdict_counts.items():
            percentage = (count / total_count) * 100
            print(f" • {v_type.capitalize():<10}: {count} ({percentage:.2f}%)")
        print("-" * 40)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Check if the filename argument is provided
    if len(sys.argv) < 2:
        print("Usage: python summarizer.py <filename.jsonl>")
    else:
        target_file = sys.argv[1]
        summarize_jsonl_results(target_file)