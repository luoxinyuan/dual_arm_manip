import cv2
import pygame
import numpy as np
import json

curve_points = []
sampled_points = []
drawing = False
background_image = None

def sample_points_on_curve(points, num_samples):
    if len(points) < 2:
        return []

    distances = [0]
    for i in range(1, len(points)):
        p1 = np.array(points[i-1])
        p2 = np.array(points[i])
        distances.append(distances[-1] + np.linalg.norm(p2 - p1))

    total_length = distances[-1]
    step = total_length / (num_samples - 1)

    sampled = []
    current_distance = 0

    for i in range(1, len(points)):
        while current_distance <= distances[i]:
            ratio = (current_distance - distances[i-1]) / (distances[i] - distances[i-1])
            p1 = np.array(points[i-1])
            p2 = np.array(points[i])
            new_point = (1 - ratio) * p1 + ratio * p2
            sampled.append((int(new_point[0]), int(new_point[1])))
            current_distance += step

    sampled.append(points[-1])
    return sampled

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        return None

    print("Capturing image...")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Unable to capture frame.")
        return None

    cv2.imwrite("captured_image.jpg", frame)
    print("Image saved as 'captured_image.jpg'")
    return frame


def save_results(curve_points, sampled_points, image, output_image_path="output_with_curve.jpg", output_json_path="curve_points.json"):

    for i in range(1, len(curve_points)):
        cv2.line(image, curve_points[i-1], curve_points[i], (0, 255, 0), 2)

    for point in sampled_points:
        cv2.circle(image, point, 5, (0, 0, 255), -1)
        cv2.putText(image, f"{point}", (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(output_image_path, image)
    print(f"Image with curve and sampled points saved as '{output_image_path}'")

    converted_points = [(x, y, 0, 0, 0, 2.5) for x, y in sampled_points]
    with open(output_json_path, "w") as json_file:
        json.dump(converted_points, json_file, indent=4)
    print(f"Curve points and sampled points saved to '{output_json_path}'")

def main():
    global curve_points, sampled_points, drawing, background_image

    frame = capture_image()
    if frame is None:
        return

    photo_height, photo_width, _ = frame.shape

    print(photo_height, photo_width)

    pygame.init()

    WINDOW_WIDTH = photo_width
    WINDOW_HEIGHT = photo_height
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Image with Curve Drawing")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    background_image = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

    print("Instructions:")
    print(" - Hold the left mouse button to draw a curve.")
    print(" - Press 's' to sample points on the curve.")
    print(" - Press 'q' to quit.")

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drawing = True
                    curve_points = []
                    curve_points.append(event.pos)

            elif event.type == pygame.MOUSEMOTION and drawing:
                curve_points.append(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    sampled_points = sample_points_on_curve(curve_points, num_samples=10)
                    print("Sampled Points:", sampled_points)
                elif event.key == pygame.K_q:
                    running = False

        screen.blit(background_image, (0, 0))

        if len(curve_points) > 1:
            pygame.draw.lines(screen, (0, 255, 0), False, curve_points, 2)

        for point in sampled_points:
            pygame.draw.circle(screen, (255, 0, 0), point, 5)
            font = pygame.font.Font(None, 24)
            text_surface = font.render(f"{point}", True, (255, 255, 255))
            screen.blit(text_surface, (point[0] + 5, point[1] - 5))

        pygame.display.flip()
        clock.tick(30)

    print("Saving results...")
    opencv_frame = cv2.cvtColor(np.flip(np.rot90(pygame.surfarray.array3d(background_image)), axis=0), cv2.COLOR_RGB2BGR)
    save_results(curve_points, sampled_points, opencv_frame)

    pygame.quit()

if __name__ == "__main__":
    main()
