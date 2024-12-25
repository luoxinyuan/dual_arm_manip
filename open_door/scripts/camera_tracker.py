import cv2
import pupil_apriltags as apriltag
import time
import math  # Import math module for angle calculation
import os
import json

def detect_and_record1(detector, cap, recorded_positions):
    # Capture a single frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame from the camera.")
        return

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags in the frame
    detections = detector.detect(gray)
    if detections:
        for detection in detections:
            # Corners of the tag (clockwise from top-left)
            center = detection.center.tolist()  # Center of the tag
            corners = detection.corners  # 4x2 array of corner points

            # Calculate the angle of rotation (based on the top edge)
            top_edge_vector = corners[1] - corners[0]  # Vector from top-left to top-right
            angle_rad = math.atan2(top_edge_vector[1], top_edge_vector[0])  # Rotation angle in radians
            angle_deg = math.degrees(angle_rad)  # Convert to degrees if needed

            # Append tag ID, center, and rotation angle to the recorded list
            # recorded_positions.append({
            #     "id": detection.tag_id,
            #     "center": detection.center.tolist(),
            #     "rotation_angle_rad": angle_rad,
            #     "rotation_angle_deg": angle_deg
            # })
            recorded_positions.append([center[0], center[1], 0, 0, 0, angle_rad])

        print(f"Recorded {len(detections)} tag(s): {[(det.tag_id, det.center, math.degrees(math.atan2((det.corners[1]-det.corners[0])[1], (det.corners[1]-det.corners[0])[0]))) for det in detections]}")
    else:
        print("No tags detected.")

def detect_and_record(detector, cap, recorded_positions, num=1, output_folder="calibration"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture a single frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame from the camera.")
        return

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags in the frame
    detections = detector.detect(gray)
    if detections:
        for detection in detections:
            # Corners of the tag (clockwise from top-left)
            center = detection.center.tolist()  # Center of the tag
            corners = detection.corners  # 4x2 array of corner points

            # Calculate the angle of rotation (based on the top edge)
            top_edge_vector = corners[1] - corners[0]
            angle_rad = math.atan2(top_edge_vector[1], top_edge_vector[0])
            angle_deg = math.degrees(angle_rad)

            # Append tag ID, center, and rotation angle to the recorded list
            recorded_positions.append([center[0], center[1], 0, 0, 0, angle_rad])

            for i in range(4):
                pt1 = tuple(corners[i].astype(int))
                pt2 = tuple(corners[(i+1) % 4].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # 画边框

            center_point = tuple(int(c) for c in center)
            cv2.circle(frame, center_point, 5, (0, 0, 255), -1)  # 画中心点
            cv2.putText(frame, f"ID: {detection.tag_id}", center_point, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)  # 显示 ID

        # 保存图片
        output_path = os.path.join(output_folder, f'detected_tags_{num}.jpg')
        cv2.imwrite(output_path, frame)
        print(f"Saved detection image to: {output_path}")

        print(f"Recorded {len(detections)} tag(s): {[(det.tag_id, det.center, angle_deg) for det in detections]}")
        return True
    else:
        print("No tags detected.")
        return False

def record():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        return

    # Set up AprilTag detector
    os.add_dll_directory("C:/Users/arpit/miniconda3/envs/doorbot/lib/site-packages/pupil_apriltags.libs")
    detector = apriltag.Detector(
            families="tag36h11",       
            nthreads=4,               
            quad_decimate=1.0,        
            quad_sigma=0.0,           
            refine_edges=True,        
            decode_sharpening=0.25,   
            debug=False       
        )

    # List to store detected tag information
    recorded_positions = []

    print("Instructions:")
    print(" - Enter 'r' to capture AprilTag data.")
    print(" - Enter 'q' to quit and save recorded data.")

    try:
        while True:
            user_input = input("Enter command (r: record, q: quit): ").strip().lower()
            if user_input == "r":
                detect_and_record(detector, cap, recorded_positions)
            elif user_input == "q":
                print("Exiting...")
                break
            else:
                print("Invalid command. Please enter 'r' or 'q'.")
    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        # Release the camera
        cap.release()

        # Save the recorded positions to a text file
        if recorded_positions:
            with open("recorded_positions.json", "w") as file:
               json.dump(recorded_positions, file, indent=4)
            print(f"Recorded positions saved to 'recorded_positions.txt'.")
        else:
            print("No positions recorded.")

def find_camera_index():
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera found at index {index}")
            cap.release()
        else:
            print(f"No camera at index {index}")
        index += 1
        if index > 10:
            break

def test_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        return
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture a frame from the camera.")
        cap.release()
        return
    cv2.imwrite("captured_image.jpg", frame)
    print(f"Photo captured")

# if __name__ == "__main__":
#     record()

    
    
