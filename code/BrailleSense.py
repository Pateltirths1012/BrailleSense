import cv2  # Import OpenCV library for image processing
import pytesseract  # Import pytesseract for OCR functionality
import re  # Import regular expressions for text cleaning

# Import defaultdict for efficient dictionary management
from collections import defaultdict
from datetime import datetime  # Import datetime for timestamp generation
from transformers import (
    AutoTokenizer,  # Import AutoTokenizer for tokenizing text
    AutoModelWithLMHead,  # Import pre-trained language model for text generation
)
# Import pybraille for converting text to Braille
from pybraille import convertText
import os


class brailleSense:

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    # Helper function to clean text (remove unwanted characters)
    def clean_text(self, text):
        """
        Clean the given text by removing non-alphanumeric characters, except for common punctuation.
        Args:
            text (str): The input text to be cleaned.
        Returns:
            str: The cleaned text.
        """
        return re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)  # Remove unwanted characters

    # Function to process OCR boxes and extract meaningful text

    def process_ocr_output(self, boxes):
        """
        Process OCR output to extract cleaned and organized text.
        Args:
            boxes (str): OCR data containing detected text and metadata.
        Returns:
            list: List of cleaned and organized sentences.
        """
        line_buffer = defaultdict(
            list
        )  # Buffer to hold text for each line based on Y-coordinate

        # Iterate over OCR data to extract and clean text
        for i, b in enumerate(boxes.splitlines()):
            if i != 0:  # Skip the first line which contains header data
                b = b.split()  # Split the line into individual components
                if len(b) == 12:  # Ensure that the line has the expected number of columns
                    # Extract bounding box and text
                    x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                    # Clean the text part of the OCR box
                    text = self.clean_text(b[11])
                    if text:
                        # Store the text with its X and Y coordinates
                        line_buffer[y].append((x, text))

        # Combine words into meaningful sentences based on sorted Y-coordinates
        sentences = []
        for key in sorted(line_buffer.keys()):
            # Sort words by X-coordinate
            words = sorted(line_buffer[key], key=lambda item: item[0])
            # Concatenate words into a sentence
            line_text = " ".join([word[1] for word in words])
            if len(line_text) > 5:  # Filter out irrelevant short lines
                sentences.append(line_text)
        return sentences

    # Post-processing the collected text fragments (cleaning and deduplication)

    def preprocess_text_file(self, input_file, output_file):
        """
        Preprocess the collected text file to remove duplicates and clean the content.
        Args:
            input_file (str): Path to the input text file containing raw data.
            output_file (str): Path to the output file where processed data will be saved.
        Returns:
            list: List of cleaned, unique text lines.
        """
        with open(input_file, "r") as file:
            lines = file.readlines()  # Read all lines from the input file

        # Clean each line and remove unwanted characters
        text_lines = [
            self.clean_text(line.split("] ", 1)[-1]) for line in lines
        ]  # Extract text after timestamp

        # Remove duplicates and empty lines
        # Use set to remove duplicates and filter empty lines
        unique_lines = list(filter(None, sorted(set(text_lines))))

        # Save the processed lines to the output file
        with open(output_file, "w") as file:
            file.write("\n".join(unique_lines))

        return unique_lines  # Return cleaned and unique text lines

    # Function to generate a sentence based on the provided words

    def gen_sentence(self, words, max_length=32):
        """
        Generate a meaningful sentence using a pre-trained language model.
        Args:
            words (str): The input text to be used as a prompt.
            max_length (int): The maximum length of the generated sentence.
        Returns:
            str: The generated sentence.
        """
        input_text = words  # The input text for sentence generation
        # Tokenize the input text
        features = self.tokenizer([input_text], return_tensors="pt")

        # Generate the output using the model
        output = self.model.generate(
            # Pass tokenized input to the model
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],  # Pass attention mask
            max_length=max_length,  # Set maximum sentence length
        )

        # Decode the output to text
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    # Display versions of important libraries for troubleshooting
    print("OpenCV version:", cv2.__version__)  # Check OpenCV version
    # Check pytesseract version
    print("pytesseract version:", pytesseract.__version__)

    # Set the path to the Tesseract executable for OCR
    # Path to Tesseract executable
    pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

    # Load the pre-trained tokenizer and model for sentence generation
    tokenizer = AutoTokenizer.from_pretrained(
        "mrm8488/t5-base-finetuned-common_gen")
    model = AutoModelWithLMHead.from_pretrained(
        "mrm8488/t5-base-finetuned-common_gen")
    
    # initializing the class variable
    brailleSenseMethod = brailleSense(tokenizer=tokenizer, model=model)

    # Initialize storage for detected and processed data
    detected_data = []  # Store detected text along with timestamps
    processed_sentences = []  # Store unique, processed sentences

    # Start video capture from the default webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():  # Check if webcam opened successfully
        # Raise an error if webcam cannot be accessed
        raise IOError("Cannot open webcam")

    # Infinite loop to capture frames continuously
    while True:
        ret, frame = cap.read()  # Capture frame from webcam
        if not ret:
            # Handle failure in capturing frame
            print("Failed to capture frame")
            break

        # Preprocess frame for better OCR accuracy (convert to grayscale, then binary threshold)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Perform OCR on the preprocessed frame
        boxes = pytesseract.image_to_data(binary)

        # Process the OCR output and extract meaningful sentences
        for line_text in brailleSenseMethod.process_ocr_output(boxes):
            if line_text not in processed_sentences:  # Avoid processing duplicates
                # Add new sentences to the list
                processed_sentences.append(line_text)
                timestamp = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )  # Generate timestamp
                # Store detected data
                detected_data.append((timestamp, line_text))
                # Print detected text with timestamp
                print(f"[{timestamp}] {line_text}")

        # Display the current frame with OCR detection in real-time
        cv2.imshow("Text Detection", frame)

        # Exit condition: press 'q' to stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Save the detected text data to a text file for future use
    with open("detected_text.txt", "w") as f:
        for timestamp, sentence in detected_data:
            f.write(f"[{timestamp}] {sentence}\n")

    # Preprocess the detected text file and save the cleaned output
    brailleSenseMethod.preprocess_text_file("detected_text.txt", "final_output.txt")

    # Read the cleaned text from the final output file
    with open("final_output.txt", "r") as file:
        fragments = file.readlines()

    # Join all fragments into a single string
    flattened_string = " ".join(fragments)

    # Remove duplicate words from the flattened string
    unique_flattened_string = " ".join(dict.fromkeys(flattened_string.split()))
    print("Unique Flattened String:", unique_flattened_string)
    
    # Generate the final sentence from the cleaned, unique words
    final_answer = brailleSenseMethod.gen_sentence(unique_flattened_string)
    print("generated answer:", final_answer)
    
    # Convert the generated sentence to Braille using pybraille
    braille_text = convertText(final_answer)
    print("Braille Text:",braille_text)

    # # Delete the intermediate text files
    # try:
    #     if os.path.exists("detected_text.txt"):  # Check if the file exists
    #         os.remove("detected_text.txt")  # Delete the file
    #         print("Deleted 'detected_text.txt' successfully.")
    #     else:
    #         print("'detected_text.txt' does not exist.")

    #     if os.path.exists("final_output.txt"):  # Check if the file exists
    #         os.remove("final_output.txt")  # Delete the file
    #         print("Deleted 'final_output.txt' successfully.")
    #     else:
    #         print("'final_output.txt' does not exist.")
    # except Exception as e:
    #     # Handle any errors during deletion
    #     print(f"Error occurred while deleting files: {e}")

    return braille_text


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An unexpected error occurred:", e)
