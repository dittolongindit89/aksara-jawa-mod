import pandas as pd
import ast
from fuzzywuzzy import fuzz, process

# Load Javanese word dataset
dataset_file = "datasets/javanese.csv"
df = pd.read_csv(dataset_file)

# Standardize column names
df.columns = ["hanacaraka", "javanese_words"]

# Convert "javanese_words" column from string representation to actual lists
df["javanese_words"] = df["javanese_words"].apply(ast.literal_eval)

# Flatten the list of valid words
valid_words_list = set([word for sublist in df["javanese_words"] for word in sublist])


def segment_text(text, valid_words):
    """
    Uses dynamic programming to segment text into the most probable words
    using fuzzy matching with Javanese words.
    """
    n = len(text)
    dp = [None] * (n + 1)  # Stores the best segmentation at each index
    dp[0] = []  # Base case: no words yet

    for i in range(1, n + 1):
        best_segmentation = None
        best_score = -1

        # Try all possible segmentations up to this point
        for j in range(max(0, i - 8), i):  # Consider up to 8-character words
            segment = text[j:i]

            # First, check for exact matches
            if segment in valid_words:
                match, score = segment, 100
            else:
                match, score = (
                    process.extractOne(segment, valid_words, scorer=fuzz.ratio)
                    if segment
                    else (None, 0)
                )

            if match and score > 70:  # Increased threshold for a valid match
                if dp[j] is not None:
                    candidate = dp[j] + [
                        match.strip()
                    ]  # Ensure words are stripped and stored correctly

                    # Ensure we calculate score properly
                    candidate_score = sum(
                        fuzz.ratio(w, max(valid_words, key=lambda v: fuzz.ratio(w, v)))
                        for w in candidate
                    ) / len(candidate)

                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_segmentation = candidate

        dp[i] = (
            best_segmentation
            if best_segmentation is not None
            else (dp[i - 1] if dp[i - 1] is not None else [text])
        )

    return dp[n] if dp[n] else [text]  # Ensure fallback returns a list


def reconstruct_javanese_text(ocr_lines, valid_words=valid_words_list):
    """
    Takes OCR-detected characters, reconstructs words using probabilistic segmentation.
    Returns a dictionary with segmented words grouped per line.
    """
    reconstructed_result = {}

    for line_key, ocr_result in ocr_lines.items():
        continuous_text = "".join(ocr_result)  # Javanese script is continuous
        reconstructed_words = segment_text(continuous_text, valid_words)
        reconstructed_result[line_key] = " ".join(
            reconstructed_words
        )  # Join words per line

    # Fix: Join the values (reconstructed text) instead of the keys
    final_sentence = " ".join(reconstructed_result.values())

    return {"reconstruct": reconstructed_result, "final_sentence": final_sentence}
