import cv2
import pygame
import numpy as np
import pupil_apriltags as apriltag
import json
import os
import math

recording = False
recorded_positions = []
recorded_positions_left = []
recorded_positions_right = []
background_image = None
drawing = False
curve_points = []
curve_points_left = []
curve_points_right = []
sampled_points = []
sampled_points_left = []
sampled_points_right = []
root_dir = "dmp_traj"

# sample function (not used currently)
def sample_points_on_curve(points, num_samples):
    if len(points) < 2:
        return []

    # 计算每段线的长度
    distances = [0]
    for i in range(1, len(points)):
        p1 = np.array(points[i-1])
        p2 = np.array(points[i])
        distances.append(distances[-1] + np.linalg.norm(p2 - p1))  # 累计距离

    total_length = distances[-1]
    step = total_length / (num_samples - 1)  # 均匀间隔

    sampled = []
    current_distance = 0

    # 在曲线上均匀插值采样
    for i in range(1, len(points)):
        while current_distance <= distances[i]:
            ratio = (current_distance - distances[i-1]) / (distances[i] - distances[i-1])
            p1 = np.array(points[i-1])
            p2 = np.array(points[i])
            new_point = (1 - ratio) * p1 + ratio * p2  # 线性插值
            sampled.append((int(new_point[0]), int(new_point[1])))
            current_distance += step

    sampled.append(points[-1])  # 添加最后一个点
    return sampled

# Detect single tag
def detect_and_record(detector, cap, recorded_positions, curve_points):
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame from the camera.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    if detections:
        for detection in detections:
            center = detection.center.tolist()
            corners = detection.corners
            top_edge_vector = corners[1] - corners[0]
            angle_rad = math.atan2(top_edge_vector[1], top_edge_vector[0])

            recorded_positions.append([center[0], center[1], 0, 0, 0, angle_rad])
            curve_points.append([center[0], center[1]])
            print(f"Recorded tag: {detection.tag_id} at {center} with angle {math.degrees(angle_rad)} degrees.")

            # 绘制 AprilTag
            for i in range(4):
                pt1 = tuple(corners[i].astype(int))
                pt2 = tuple(corners[(i + 1) % 4].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            center_point = tuple(int(c) for c in center)
            cv2.circle(frame, center_point, 5, (0, 0, 255), -1)

        print(f"Recorded {len(detections)} tags.")
    return frame

# Detect left and right at the same time
def detect_and_record_both(detector, cap, recorded_positions_left, curve_points_left, 
                      recorded_positions_right, curve_points_right):
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame from the camera.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    if detections:
        if not recorded_positions_left and not recorded_positions_right and len(detections) == 2:
            sorted_detections = sorted(detections, key=lambda det: det.center[0])  # 按 x 坐标排序
            left_tag = sorted_detections[0]
            right_tag = sorted_detections[1]

            center_left = left_tag.center.tolist()
            corners_left = left_tag.corners
            top_edge_vector_left = corners_left[1] - corners_left[0]
            angle_rad_left = math.atan2(top_edge_vector_left[1], top_edge_vector_left[0])

            recorded_positions_left.append([center_left[0], center_left[1], 0, 0, 0, angle_rad_left])
            curve_points_left.append([center_left[0], center_left[1]])
            print(f"Assigned left tag: {left_tag.tag_id} at {center_left} with angle {math.degrees(angle_rad_left)} degrees.")

            center_right = right_tag.center.tolist()
            corners_right = right_tag.corners
            top_edge_vector_right = corners_right[1] - corners_right[0]
            angle_rad_right = math.atan2(top_edge_vector_right[1], top_edge_vector_right[0])

            recorded_positions_right.append([center_right[0], center_right[1], 0, 0, 0, angle_rad_right])
            curve_points_right.append([center_right[0], center_right[1]])
            print(f"Assigned right tag: {right_tag.tag_id} at {center_right} with angle {math.degrees(angle_rad_right)} degrees.")
        
        elif recorded_positions_left and recorded_positions_right:
            for detection in detections:
                center = detection.center.tolist()
                corners = detection.corners
                top_edge_vector = corners[1] - corners[0]
                angle_rad = math.atan2(top_edge_vector[1], top_edge_vector[0])

                dist_to_left = np.linalg.norm(np.array(center) - np.array(curve_points_left[-1])) if curve_points_left else float('inf')
                dist_to_right = np.linalg.norm(np.array(center) - np.array(curve_points_right[-1])) if curve_points_right else float('inf')

                if dist_to_left <= dist_to_right:
                    recorded_positions_left.append([center[0], center[1], 0, 0, 0, angle_rad])
                    curve_points_left.append([center[0], center[1]])
                    print(f"Added to left: {center} with angle {math.degrees(angle_rad)} degrees.")
                else:
                    recorded_positions_right.append([center[0], center[1], 0, 0, 0, angle_rad])
                    curve_points_right.append([center[0], center[1]])
                    print(f"Added to right: {center} with angle {math.degrees(angle_rad)} degrees.")
    else:
        print("No tags detected.")
    return frame

# save one tag
def save_results(root_dir, recorded_positions, curve_points, sampled_points):
    converted_points = [(x, y, 0, 0, 0, 2.5) for x, y in sampled_points]
    with open(f"{root_dir}/april_tag_points.json", "w") as file:
        json.dump(recorded_positions, file, indent=4)
    print("Sampled points saved to 'april_tag_points.json'.")

# save left and right tags
def save_results_left_right(root_dir, recorded_positions_left, curve_points_left, recorded_positions_right, curve_points_right, sampled_points_left, sampled_points_right):
    with open(f"{root_dir}/april_tag_points_left.json", "w") as file:
        json.dump(recorded_positions_left, file, indent=4)
    with open(f"{root_dir}/april_tag_points_right.json", "w") as file:
        json.dump(recorded_positions_right, file, indent=4)
    print("Sampled points saved to 'april_tag_points_left.json' and 'april_tag_points_right.json'.")

def main():
    global recording, recorded_positions, curve_points, sampled_points, recorded_positions_left, curve_points_left, recorded_positions_right, curve_points_right
    global root_dir
    frames = []

    cap = cv2.VideoCapture(0)
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
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = f"{root_dir}/output_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    frame_rate = 30
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("AprilTag Tracking")

    clock = pygame.time.Clock()
    running = True

    print("Instructions:")
    print(" - Press 'r' to record AprilTag data.")
    print(" - Press 'e' to stop recording and save data.")
    # print(" - Press 's' to save ")
    print(" - Press 'q' to quit.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("Recording started...")
                    recording = True

                elif event.key == pygame.K_e:
                    print("Recording stopped.")
                    recording = False
                    save_results_left_right(root_dir, recorded_positions_left, curve_points_left, recorded_positions_right, curve_points_right, sampled_points_left, sampled_points_right)

                # elif event.key == pygame.K_s:
                #     # sampled_points = sample_points_on_curve(curve_points, num_samples=10)
                #     # print("Sampled Points:", sampled_points)
                #     # save_results(recorded_positions, curve_points, sampled_points)
                #     save_results_left_right(root_dir, recorded_positions_left, curve_points_left, recorded_positions_right, curve_points_right, sampled_points_left, sampled_points_right)

                elif event.key == pygame.K_q:
                    running = False

        ret, frame = cap.read()
        video_writer.write(frame)
        if not ret:
            print("Error: Unable to capture frame.")
            continue

        if recording:
            # detect_and_record(detector, cap, recorded_positions, curve_points)
            detect_and_record_both(detector, cap, recorded_positions_left, curve_points_left, recorded_positions_right, curve_points_right)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = np.rot90(frame)
        frame = frame.swapaxes(0, 1)
        background_image = pygame.surfarray.make_surface(frame)
        screen.blit(background_image, (0, 0))

        # Draw the trajectory and direction recorded
        if len(curve_points) > 1:
            pygame.draw.lines(screen, (0, 255, 0), False, curve_points, 2)
        if len(curve_points_left) > 1:
            pygame.draw.lines(screen, (255, 0, 0), False, curve_points_left, 2)
        if len(curve_points_right) > 1:
            pygame.draw.lines(screen, (0, 0, 255), False, curve_points_right, 2)

        if len(recorded_positions_left) > 1:
            last_point = recorded_positions_left[-1]
            angle = last_point[-1]
            arrow_length = 20
            arrow_end = (last_point[0] + arrow_length * math.cos(angle), last_point[1] + arrow_length * math.sin(angle))
            pygame.draw.lines(screen, (0, 255, 0), False, [last_point[:2], arrow_end], 2)
        if len(recorded_positions_right) > 1:
            last_point = recorded_positions_right[-1]
            angle = last_point[-1]
            arrow_length = 20
            arrow_end = (last_point[0] + arrow_length * math.cos(angle), last_point[1] + arrow_length * math.sin(angle))
            pygame.draw.lines(screen, (0, 255, 0), False, [last_point[:2], arrow_end], 2)

        for point in sampled_points:
            pygame.draw.circle(screen, (255, 0, 0), point, 5)
            font = pygame.font.Font(None, 24)
            text_surface = font.render(f"{point}", True, (255, 255, 255))
            screen.blit(text_surface, (point[0] + 5, point[1] - 5))


        pygame.display.flip()
        clock.tick(30)

    cap.release()
    video_writer.release()
    pygame.quit()
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    main()
