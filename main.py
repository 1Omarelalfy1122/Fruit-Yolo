from ultralytics import YOLO
import cv2
import numpy as np
import glob
# Load the trained model
model = YOLO("runs/detect/train3/weights/best.pt")

def run_camera_detection():
    """Run real-time object detection using webcam"""
    # Initialize camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        print("Make sure your camera is connected and not being used by another application")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera window opened. Press 'q' to quit, 's' to save current frame")
    print("Make sure to click on the camera window to activate it for key presses")
    
    frame_count = 0
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            try:
                # Run YOLO detection on the frame with smaller image size for better performance
                results = model.predict(frame, conf=0.5, verbose=False, imgsz=640)
                
                # Draw detections on the frame
                annotated_frame = results[0].plot()
                
                # Display detection info
                if len(results[0].boxes) > 0:
                    detection_text = f"Objects detected: {len(results[0].boxes)}"
                else:
                    detection_text = "No objects detected"
                
                # Add text overlay
                cv2.putText(annotated_frame, detection_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add instructions
                cv2.putText(annotated_frame, "Press 'q' to quit, 's' to save", (10, 460), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            except Exception as e:
                print(f"Detection error: {e}")
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, f"Detection Error", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show the frame
            cv2.imshow('YOLO Real-time Detection', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting camera detection...")
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"detection_frame_{frame_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Frame saved as {filename}")
                frame_count += 1
    
    except KeyboardInterrupt:
        print("\nCamera detection interrupted by user")
    except Exception as e:
        print(f"Unexpected error during camera detection: {e}")
    finally:
        # Release camera and close windows
        cap.release()
        cv2.destroyAllWindows()
        print("Camera detection stopped.")

def test_on_images():
    """Test the model on static images"""
    print("Testing model on static images...")
    
    # Test on multiple images - add your own image paths here
    test_images = glob.glob("images/*.[jp][pn]g") + glob.glob("images/*.webp")

    for image_path in test_images:
        print(f"\n=== Testing on {image_path} ===")
        results = model.predict(image_path, conf=0.7, save=True)
        
        # Print results
        for r in results:
            if len(r.boxes) > 0:
                print(f"Detected {len(r.boxes)} objects:")
                print("Boxes:", r.boxes.xyxy)   # [x1, y1, x2, y2]
                print("Conf:", r.boxes.conf)   # confidence scores
                print("Class:", r.boxes.cls)   # class IDs
            else:
                print("No objects detected above confidence threshold")

if __name__ == "__main__":
    print("YOLO Detection Application")
    print("1. Press Enter to start camera detection")
    print("2. Type 'test' to run on static images")
    print("3. Type 'quit' to exit")
    
    while True:
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == "" or choice == "camera":
            run_camera_detection()
        elif choice == "test":
            test_on_images()
        elif choice == "quit" or choice == "exit":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.") 