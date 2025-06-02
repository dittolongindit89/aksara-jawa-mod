from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
model = load_model("models/hanacaraka_model.keras", compile=False)
print("✅ Model successfully loaded!")

# Define character classes
classes = [
    "ba",
    "ca",
    "da",
    "dha",
    "ga",
    "ha",
    "ja",
    "ka",
    "la",
    "ma",
    "na",
    "nga",
    "nya",
    "pa",
    "ra",
    "sa",
    "ta",
    "tha",
    "wa",
    "ya",
]

# Confidence threshold (change this to adjust filtering)
MIN_VALID_WIDTH = 20  # Minimum character width for valid prediction
MIN_VALID_HEIGHT = 30  # Minimum character height for valid prediction
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to accept prediction (98%)

# Parameters for line detection
LINE_DETECTION_THRESHOLD = 15  # Minimum y-coordinate gap to consider as a new line
MIN_LINE_CHARACTERS = 1  # Minimum characters required to consider a valid line


def add_padding(image, padding=10):
    """Adds padding to an image to avoid tight cropping issues."""
    h, w = image.shape
    padded_image = np.full(
        (h + 2 * padding, w + 2 * padding), 255, dtype=np.uint8
    )  # White background
    padded_image[padding : padding + h, padding : padding + w] = image
    return padded_image


def predict_character(image):
    print("Entering predict_character function")
    """Preprocess and predict a segmented character with padding."""
    image = add_padding(image, padding=10)  # Add margin before resizing
    image = cv2.resize(image, (100, 100))  # Resize to model input size
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict using model
    result = model.predict(image)
    max_confidence = np.max(result)  # Get highest confidence score
    predicted_class = classes[np.argmax(result)]  # Get predicted label

    return predicted_class, max_confidence  # Return both class & score


def filter_out_invalid_boxes(contours, img_width, img_height):
    print("Entering filter_out_invalid_boxes function")
    """
    Advanced filtering of contour bounding boxes for Aksara Jawa character recognition.

    Removes:
    - Extremely large boxes (full text or page)
    - Extremely small boxes (noise or tiny artifacts)
    - Sandhangan (diacritical marks) in specific image regions

    Args:
        contours (list): List of contours detected in the image
        img_width (int): Width of the input image
        img_height (int): Height of the input image

    Returns:
        list: Filtered list of valid bounding boxes (x, y, w, h)
    """
    filtered_contours = []

    # Refined size thresholds for Aksara Jawa detection
    max_width = img_width * 0.4  # Increased to 40% to capture larger characters
    max_height = img_height * 0.8  # Increased to 80% height threshold

    min_width = 10  # Slightly reduced minimum width
    min_height = 20  # Slightly reduced minimum height

    # Aspect ratio constraints for more precise character filtering
    min_aspect_ratio = 0.3  # Minimum width/height ratio
    max_aspect_ratio = 3.0  # Maximum width/height ratio

    # Advanced sandhangan region filtering
    max_sandhangan_y = img_height * 0.2  # Expanded top region for sandhangan
    max_sandhangan_height = img_height * 0.1  # Maximum height for sandhangan

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Calculate aspect ratio
        aspect_ratio = w / h if h > 0 else 0

        # Debug logging with more informative messages
        debug_info = f"Box at (X={x}, Y={y}, W={w}, H={h}, Aspect={aspect_ratio:.2f})"

        # Skip full text or page-level boxes
        if w > max_width or h > max_height:
            print(f"⚠️ Skipping large box: {debug_info}")
            continue

        # Skip very small boxes or noise
        if w < min_width or h < min_height:
            print(f"⚠️ Skipping small box: {debug_info}")
            continue

        # Skip boxes with unusual aspect ratios
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            print(f"⚠️ Skipping unusual aspect ratio box: {debug_info}")
            continue

        # Additional sandhangan filtering
        if y < max_sandhangan_y and h > max_sandhangan_height:
            print(f"⚠️ Skipping potential sandhangan: {debug_info}")
            continue

        # Optional: Area-based filtering for more precision
        area = w * h
        if (
            area < img_width * img_height * 0.001
        ):  # Skip if area is less than 0.1% of image
            print(f"⚠️ Skipping tiny area box: {debug_info}")
            continue

        filtered_contours.append((x, y, w, h))

    return filtered_contours


def expand_and_resize_character(cropped_char, canvas_size=150):
    print("Entering expand_and_resize_character function")
    """
    Places the segmented character on a larger blank canvas and resizes it to fit properly.
    Ensures the character remains WHITE on a BLACK background.
    """
    h, w = cropped_char.shape

    # **Check if the character is valid before processing**
    if w < MIN_VALID_WIDTH or h < MIN_VALID_HEIGHT:
        print(f"⚠️ Character too small (W={w}, H={h}), skipping prediction.")
        return None  # Skip prediction

    # Create a blank BLACK canvas
    expanded_char = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # Find scale factor to fit character in new canvas (keeping aspect ratio)
    scale_factor = min(
        (canvas_size - 20) / h, (canvas_size - 20) / w
    )  # Leave some margin
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)

    # Resize character while maintaining aspect ratio
    resized_char = cv2.resize(
        cropped_char, (new_w, new_h), interpolation=cv2.INTER_AREA
    )

    # Find the center position in the new canvas
    x_offset = (canvas_size - new_w) // 2
    y_offset = (canvas_size - new_h) // 2

    # Place the resized character in the center of the black canvas
    expanded_char[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
        resized_char
    )

    # **INVERT COLORS so the character is WHITE on BLACK**
    expanded_char = cv2.bitwise_not(expanded_char)

    return expanded_char


def detect_line_boundaries(filtered_contours):
    print("Entering detect_line_boundaries function")
    """
    Detects multiple lines of text based on y-coordinate clustering.
    Returns contours grouped by line.
    """
    if not filtered_contours:
        return []

    # Extract y-coordinate and height for each contour
    y_positions = [(i, c[1], c[1] + c[3]) for i, c in enumerate(filtered_contours)]
    # Sort by y-coordinate (top to bottom)
    y_positions.sort(key=lambda x: x[1])

    # Group contours into lines using y-coordinate clustering
    line_groups = []
    current_line = [y_positions[0][0]]  # Start with first contour
    current_line_bottom = y_positions[0][2]

    for i in range(1, len(y_positions)):
        idx, y_top, y_bottom = y_positions[i]

        # If this contour overlaps with current line or is close enough, add to current line
        if y_top <= current_line_bottom + LINE_DETECTION_THRESHOLD:
            current_line.append(idx)
            # Update the bottom boundary of current line if needed
            current_line_bottom = max(current_line_bottom, y_bottom)
        else:
            # Start a new line if the contour is far enough below
            if len(current_line) >= MIN_LINE_CHARACTERS:
                line_groups.append(current_line)
            current_line = [idx]
            current_line_bottom = y_bottom

    # Add the last line if it has enough characters
    if len(current_line) >= MIN_LINE_CHARACTERS:
        line_groups.append(current_line)

    # Convert indices back to contours grouped by line
    contours_by_line = []
    for line in line_groups:
        line_contours = [filtered_contours[idx] for idx in line]
        # Sort each line's contours from left to right
        line_contours.sort(key=lambda c: c[0])
        contours_by_line.append(line_contours)

    return contours_by_line


def multi_line_character_segmentation(image, visualize=True):
    """Segments Javanese characters from multiple lines and returns predictions in JSON format."""
    # Load image in grayscale
    print(f"Image type: {type(image)}")

    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    print(f"Converted image shape: {img.shape}")
    print(f"Converted image type: {type(img)}")

    if img is None:
        print("❌ Error: Image conversion failed")
        return {"error": "Failed to convert image"}

    img_height, img_width = img.shape[:2]

    # Step 1: Adaptive Thresholding
    _, binary = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Morphological Operations (Refined for thinner characters)
    kernel_close = np.ones((3, 3), np.uint8)  # Helps close small gaps
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # Step 3: Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter: Remove invalid boxes
    all_filtered_contours = filter_out_invalid_boxes(contours, img_width, img_height)
    print(
        f"✅ Total detected characters (after filtering): {len(all_filtered_contours)}"
    )

    # Group contours by line
    contours_by_line = detect_line_boundaries(all_filtered_contours)
    num_lines = len(contours_by_line)
    print(f"✅ Detected {num_lines} lines of text")

    # Create output dictionary
    results_dict = {}

    # Process each line
    for line_idx, line_contours in enumerate(contours_by_line, start=1):
        line_predictions = []
        line_key = f"line_{line_idx}"

        print(f"\n=== Processing Line {line_idx} ({len(line_contours)} characters) ===")

        # Process each character in this line
        for i, (x, y, w, h) in enumerate(line_contours, start=1):
            # Crop the segmented character
            cropped_char = binary[y : y + h, x : x + w]

            # Expand and resize character to fit properly with correct color format
            expanded_char = expand_and_resize_character(cropped_char, canvas_size=150)

            if expanded_char is None:
                continue  # Skip prediction if the character is too small

            # Predict character with confidence score
            predicted_label, confidence = predict_character(expanded_char)

            # Ensure prediction is only made when confidence is high enough
            if confidence >= CONFIDENCE_THRESHOLD:
                line_predictions.append(predicted_label)
                print(
                    f"✅ Line {line_idx}, Char {i}: {predicted_label} (Confidence: {confidence*100:.2f}%)"
                )

                # # Display the character with prediction & confidence
                # if visualize:
                #     plt.figure(figsize=(2, 2))
                #     plt.imshow(expanded_char, cmap="gray")
                #     plt.title(
                #         f"Line {line_idx}: {predicted_label} ({confidence*100:.1f}%)"
                #     )
                #     plt.axis("off")
                #     plt.show()
            else:
                print(
                    f"⚠️ Line {line_idx}, Char {i}: Below confidence threshold ({confidence*100:.2f}%)"
                )

        # Add line predictions to results dictionary
        results_dict[line_key] = line_predictions

    # If no lines were detected, add an empty result
    if not results_dict:
        print("⚠️ No valid lines detected in the image")
        results_dict = {"line_1": []}

    # Print final JSON results
    json_results = json.dumps(results_dict, indent=2)
    print("\nFinal Results (JSON format):")
    print(json_results)

    return results_dict
