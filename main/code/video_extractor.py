import cv2 as cv
import os

def extract_calibration_frames(video_path, output_folder, extract_every_n_frames=30):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    print(f"Extracting frames from {video_path}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only save 1 frame per second (assuming 30fps)
        if frame_count % extract_every_n_frames == 0:
            # Format the filename with leading zeros (ex: frame_005.jpg)
            filename = os.path.join(output_folder, f"frame_{saved_count:03d}.jpg")
            cv.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done! Saved {saved_count} perfectly spaced images to '{output_folder}'.\n")

if __name__ == '__main__':

    extract_calibration_frames('vids/locking/right-cam-lock.mp4', 'locking_images_right_cam', extract_every_n_frames=30)