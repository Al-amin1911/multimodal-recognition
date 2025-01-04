# ADDING CAMERA FEATURE
# REGISTERED PERSONS RECOGNITION WITH UPLOAD FUNCTION AND PERSON BASE MATCH
import customtkinter
from customtkinter import CTkImage
import cv2
from opencv.fr import FR
from opencv.fr.search.schemas import SearchRequest
from PIL import Image
from tkinter import filedialog
import os

# Establish connection
BACKEND_URL = "https://eu.opencv.fr"
DEVELOPER_KEY = ""    #Add Developer Key

# Initialize the SDK
sdk = FR(BACKEND_URL, DEVELOPER_KEY)

# Confidence threshold for granting access
# CONFIDENCE_THRESHOLD = 0.8  # 80%

# GUI setup
customtkinter.set_appearance_mode('dark')
customtkinter.set_default_color_theme('dark-blue')
root = customtkinter.CTk()
root.geometry('500x500')
root.title('Face Recognition')


def upload_and_match():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    match_image(file_path)


def match_image(file_path):
    try:
        if not file_path or not os.path.exists(file_path):
            result_label.configure(text="Invalid file selected.", text_color="red")
            return

        # Create a search request with the uploaded image
        search_request = SearchRequest([file_path])
        results = sdk.search.search(search_request)
        print(results)

        if results:
            # Get the best match from the results
            best_match = results[0].person.name
            result_label.configure(
                text=f"Match found: {best_match}", text_color="white"
            )

        else:
            result_label.configure(text="No match found. Access denied.", text_color="red")

        # Display uploaded image
        try:
            pil_image = Image.open(file_path)
            uploaded_image = CTkImage(pil_image, size=(250, 250))
            image_label.configure(image=uploaded_image)
        except Exception as img_error:
            result_label.configure(text="Error loading image: " + str(img_error), text_color="red")
            return

    except Exception as e:
        print(f"Error during search or matching: {e}")
        result_label.configure(
            text=f"An error occurred: {str(e)}", text_color="red"
        )


def capture_and_match():
    # Open the webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        result_label.configure(text="Unable tp access camera.", text_color="red")
        return

    result_label.configure(text="Press 'C' to capture, or 'Q' to quit.", text_color="white")
    captured_file_path = "captured_image.jpg"

    while True:
        ret, frame = cap.read()
        if not ret:
            result_label.configure(text="Failed to capture frame.", text_color="red")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the frame
        cv2.imshow("Camera - Press 'C' to capture, or 'Q' to quit.", gray)

        # Wait for input
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):  # 'c' key to capture
            cv2.imwrite(captured_file_path, frame)
            result_label.configure(text="Image captured. Matching...", text_color="yellow")
            cap.release()
            cv2.destroyAllWindows()
            match_image(captured_file_path)
            break
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            result_label.configure(text="Image captured. Quitting...", text_color="yellow")
            break


# GUI Elements
upload_button = customtkinter.CTkButton(root, text="Upload and Match", command=upload_and_match)
upload_button.pack(pady=20)

camera_button = customtkinter.CTkButton(root, text="Capture from Camera", command=capture_and_match)
camera_button.pack(pady=10)

result_label = customtkinter.CTkLabel(root, text="Click 'Upload' or 'Capture' to start", font=("Arial", 16))
result_label.pack(pady=10)

image_label = customtkinter.CTkLabel(root)
image_label.pack(pady=20)

# Start the Tkinter event loop
root.mainloop()


