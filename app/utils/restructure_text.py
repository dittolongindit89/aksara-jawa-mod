import pandas as pd
import ast
from fuzzywuzzy import fuzz, process


def load_javanese_dictionary(dataset_file):
    """
    Load and prepare the Javanese dictionary from CSV file.
    """
    try:
        df = pd.read_csv(dataset_file)

        # Standardize column names
        df.columns = ["hanacaraka", "javanese_words"]

        # Convert "javanese_words" column from string representation to actual lists
        df["javanese_words"] = df["javanese_words"].apply(ast.literal_eval)

        # Create two dictionary structures:
        # 1. A set of all valid words (flattened)
        # 2. A direct mapping from OCR result to possible valid words
        valid_words_set = set(
            [word for sublist in df["javanese_words"] for word in sublist]
        )
        ocr_to_words_map = dict(zip(df["hanacaraka"], df["javanese_words"]))

        return valid_words_set, ocr_to_words_map
    except Exception as e:
        print(f"Error loading dictionary: {e}")
        return set(), {}


def segment_text_improved(text, valid_words, ocr_to_words_map):
    n = len(text)
    dp = [(0, [], True)] * (n + 1)
    dp[0] = (0, [], True)  # Base case

    MAX_WORD_LENGTH = 12
    MATCH_THRESHOLD = 85  # Increased threshold

    for i in range(1, n + 1):
        best_score = -1
        best_words = []
        best_coverage = False

        for j in range(max(0, i - MAX_WORD_LENGTH), i):
            segment = text[j:i]
            prev_score, prev_words, prev_coverage = dp[j]

            if not segment:
                continue

            current_score = prev_score
            current_words = prev_words.copy()
            matched = False

            # Strategy 1: Check for exact match in OCR mapping
            if segment in ocr_to_words_map:
                potential_matches = ocr_to_words_map[segment]
                if potential_matches:
                    match = potential_matches[0]  # Take first match
                    current_score += 100  # Highest priority
                    current_words.append(match)
                    matched = True

            # Strategy 2: Check for exact match in valid words
            elif segment in valid_words:
                current_score += 90
                current_words.append(segment)
                matched = True

            # Strategy 3: Use fuzzy matching as fallback
            else:
                best_match, match_score = process.extractOne(
                    segment, valid_words, scorer=fuzz.ratio
                )
                if match_score >= MATCH_THRESHOLD:
                    current_score += match_score
                    current_words.append(best_match)
                    matched = True

            # If we found a match, consider this segmentation
            if matched and current_score > best_score:
                best_score = current_score
                best_words = current_words
                best_coverage = True

        # Fallback: If no good segmentation found at this position
        if best_score <= -1:
            # Just add the current character as its own token
            prev_score, prev_words, prev_coverage = dp[i - 1]
            dp[i] = (prev_score, prev_words + [text[i - 1 : i]], prev_coverage)
        else:
            dp[i] = (best_score, best_words, best_coverage)

    return dp[n][1]


def find_substring_matches(text, valid_words, ocr_to_words_map):
    """
    Find recurring patterns and substrings that match dictionary entries.
    This helps when dealing with repeating character sequences.
    """
    results = []

    # First check for direct matches in OCR map
    for length in range(min(12, len(text)), 1, -1):
        for i in range(len(text) - length + 1):
            substr = text[i : i + length]
            if substr in ocr_to_words_map:
                word = ocr_to_words_map[substr][0]  # Take first match
                results.append((i, i + length, word, 100))

    # Then check for valid words (exact matches)
    for length in range(min(12, len(text)), 1, -1):
        for i in range(len(text) - length + 1):
            substr = text[i : i + length]
            if substr in valid_words:
                results.append((i, i + length, substr, 90))

    # Look for repeating patterns
    for length in range(min(6, len(text)), 1, -1):
        for i in range(len(text) - length * 2 + 1):
            pattern = text[i : i + length]
            next_segment = text[i + length : i + length * 2]
            if pattern == next_segment:
                # We found a repeating pattern
                if pattern in valid_words:
                    results.append((i, i + length * 2, pattern + " " + pattern, 80))

    # Sort by score and position
    results.sort(key=lambda x: (-x[3], x[0]))
    return results


def reconstruct_javanese_text(ocr_lines, dataset_file="datasets/javanese.csv"):
    """
    Improved reconstruction function that avoids repeated words and preserves original characters.
    """
    # Load dictionary data
    valid_words, ocr_to_words_map = load_javanese_dictionary(dataset_file)

    if not valid_words:
        return {"error": "Failed to load dictionary"}

    reconstructed_result = {}

    for line_key, ocr_result in ocr_lines.items():
        if not ocr_result:  # Skip empty lines
            reconstructed_result[line_key] = ""
            continue

        # Join OCR characters into continuous text
        continuous_text = "".join(ocr_result)

        # Special handling for repeating patterns
        if len(continuous_text) >= 5:
            pattern_matches = find_substring_matches(
                continuous_text, valid_words, ocr_to_words_map
            )

            # If we found good repeating patterns
            if pattern_matches and pattern_matches[0][3] >= 80:
                # Use pattern-based reconstruction
                covered = [False] * len(continuous_text)
                result_words = []

                for start, end, word, score in pattern_matches:
                    # Check if this segment doesn't overlap with already covered text
                    if not any(covered[start:end]):
                        result_words.append((start, word))
                        for i in range(start, end):
                            covered[i] = True

                # Fill in any gaps with individual characters
                for i in range(len(continuous_text)):
                    if not covered[i]:
                        result_words.append((i, continuous_text[i]))

                # Sort by position and extract words
                result_words.sort(key=lambda x: x[0])
                reconstructed_words = [word for _, word in result_words]
            else:
                # Fall back to dynamic programming approach
                reconstructed_words = segment_text_improved(
                    continuous_text, valid_words, ocr_to_words_map
                )
        else:
            # For very short text, use standard segmentation
            reconstructed_words = segment_text_improved(
                continuous_text, valid_words, ocr_to_words_map
            )

        # Post-process the reconstructed words
        reconstructed_text = post_process_reconstructed_text(
            " ".join([w for w in reconstructed_words if w])
        )
        reconstructed_result[line_key] = reconstructed_text

    # Create final sentence
    final_sentence = " ".join([text for text in reconstructed_result.values() if text])

    # Remove consecutive repeated words from the final sentence
    final_sentence = remove_repeated_words(final_sentence)

    return {"reconstruct": reconstructed_result, "final_sentence": final_sentence}


def post_process_reconstructed_text(text):
    """
    Post-process the reconstructed text to clean up single-character segments.
    """
    words = text.split()
    merged_words = []
    i = 0
    while i < len(words):
        if len(words[i]) == 1 and i + 1 < len(words) and len(words[i + 1]) == 1:
            # Merge single characters (e.g., "n y a" -> "nya")
            merged_word = words[i] + words[i + 1]
            merged_words.append(merged_word)
            i += 2
        else:
            merged_words.append(words[i])
            i += 1
    return " ".join(merged_words)


def remove_repeated_words(text):
    """
    Remove consecutive repeated words from the text.
    """
    words = text.split()
    if not words:
        return ""

    # Remove consecutive duplicates
    cleaned_words = [words[0]]
    for i in range(1, len(words)):
        if words[i] != words[i - 1]:
            cleaned_words.append(words[i])

    return " ".join(cleaned_words)
