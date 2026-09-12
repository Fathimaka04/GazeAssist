"""
src/sensing/camera_capture.py

Sensing Layer. Responsibility: open the webcam and yield frames — nothing
else. This module never knows about MediaPipe, EAR, or gaze. Keeping it
"dumb" lets you swap camera sources later (e.g. a recorded video file for
testing) without touching perception code.
"""

import cv2
import platform

class CameraCapture:
    def __init__(self, source=0, width=640, height=480):     # giving source=0 means using default camera there are 1,2
        
        if platform.system() == "Windows":
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)   #DSHOW direct show for the webcam
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera/source: {source}. "
                f"Check that no other app is using the webcam, and that "
                f"OS camera permissions are granted."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def frames(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)  # mirror to match intuitive "selfie view"
            yield frame

    def release(self):
        self.cap.release()


if __name__ == "__main__":
    # Standalone sanity check — run this file directly to confirm the
    # camera opens BEFORE adding MediaPipe on top.
    cam = CameraCapture(source=0)
    print("Camera opened. Press 'q' to quit.")
    window_name = "Raw Camera Feed"
    for frame in cam.frames():
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cam.release()
    cv2.destroyAllWindows()