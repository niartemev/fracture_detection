#Application for identifying fractures

#Import lbiraries
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# To store the selected image file path
selected_image_path = None

# To store the current confidence slider value
slider_value = 0.0

# Load the pre-trained YOLO model for fracture detection
best_model_path = "best.pt"
trained_model = YOLO(best_model_path)

# ----------------------------- #
#      Image Browse Handler     #
# ----------------------------- #
def browse_image():
    """
    Called when 'Browse Image' button is clicked.
    Opens a file dialog to select an image, then runs YOLO prediction,
    and displays results including class names, coordinates, and confidence scores.
    """
    global selected_image_path, image_label, img_display

    # Open file picker for images
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if file_path:
        selected_image_path = file_path
        print(f"Selected Image: {selected_image_path}")

        # Run prediction using the YOLO model with specified confidence and IOU
        results = trained_model.predict(
            source=selected_image_path, conf=slider_value, iou=0.45, verbose=True
        )

        # Print detected class names
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls)
                predicted_class_name = r.names[class_id]
                print(f"Detected: {predicted_class_name}")

        # Print detailed detection info: class, coordinates, probability
        for r in results:
            boxes = r.boxes
            for box in boxes:
                print(f"Object type: {trained_model.names[int(box.cls[0])]}")  # Class name
                print(f"Coordinates (xyxy): {box.xyxy[0].tolist()}")            # Bounding box
                print(f"Probability: {box.conf[0].item():.2f}")                # Confidence score
                print("-" * 20)

        # Display image with YOLO-predicted bounding boxes
        rendered_image = results[0].plot()  # Image with boxes
        plt.imshow(cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Hide axes
        plt.show()

# ----------------------------- #
#      Slider Update Handler    #
# ----------------------------- #
def on_slider_change(val):
    """
    Called when confidence slider value is changed.
    Updates the global confidence threshold used in detection.
    """
    global slider_value
    slider_value = round(float(val), 2)
    slider_value_label.config(text=f"{slider_value:.2f}")
    print(f"Slider Value: {slider_value}")



# Initialize main application window
root = tk.Tk()
root.title("Image & Confidence Selector")
root.geometry("250x110")
root.resizable(False, False)

# Browse button to load image
browse_button = ttk.Button(root, text="Browse Image", command=browse_image)
browse_button.pack(pady=10)

# Frame to hold confidence slider and label
slider_frame = ttk.Frame(root)
slider_frame.pack(pady=10)

# Slider label (static text)
slider_label = ttk.Label(slider_frame, text="Confidence level:")
slider_label.pack(side="left")

# Label that dynamically shows current slider value
slider_value_label = ttk.Label(slider_frame, text="0.00")
slider_value_label.pack(side="left", padx=5)

# Confidence slider to control detection threshold
slider = ttk.Scale(
    slider_frame, from_=0.0, to=1.0, command=on_slider_change
)
slider.pack(pady=5, fill='x', expand=True)
slider.set(0.07)  # Default value

# Placeholder label for future image display (not used in current layout)
image_label = tk.Label(root)
image_label.pack(pady=10)


# Start the GUI main loop
root.mainloop()
