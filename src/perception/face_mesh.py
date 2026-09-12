"""
src/perception/face_mesh.py

Perception Layer (first pass). Runs MediaPipe Face Mesh on a frame and
returns raw landmarks + the specific eye-landmark index sets you'll need
for EAR computation later. Does NOT compute EAR, gaze, or head pose yet.
"""

import cv2
import mediapipe as mp

# 6-point EAR eye layout mapped onto MediaPipe's 468-point mesh.
# Order matters: [p1, p2, p3, p4, p5, p6] — p1/p4 are horizontal corners,
# p2/p3 and p5/p6 are vertical top/bottom pairs.
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Iris points — only populated when refine_landmarks=True. Not used until
# gaze estimation later, but defined here now so all eye-related indices
# live in one place.
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [474, 475, 476, 477]


class FaceMeshDetector:
    def __init__(self, max_num_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):  #confident is the threshold
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False # no modification while processing
        results = self.face_mesh.process(frame_rgb)     #information about the detected face.
        return results

    def get_eye_points(self, results, frame_shape):
        if not results.multi_face_landmarks:
            return None, None

        h, w = frame_shape[0], frame_shape[1]
        landmarks = results.multi_face_landmarks[0].landmark

        # convert the normalized coordinates to pixel 
        def to_pixel(idx):
            lm = landmarks[idx]
            return (int(lm.x * w), int(lm.y * h))

        right_eye_pts = [to_pixel(i) for i in RIGHT_EYE]
        left_eye_pts = [to_pixel(i) for i in LEFT_EYE]
        return right_eye_pts, left_eye_pts

    def draw_landmarks(self, frame_bgr, results):
        if not results.multi_face_landmarks:
            return frame_bgr
        for face_landmarks in results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_tesselation_style(),
            )
            self.mp_drawing.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_iris_connections_style(),
            )
        return frame_bgr

    def draw_eye_points(self, frame_bgr, right_eye_pts, left_eye_pts):
        if right_eye_pts:
            for pt in right_eye_pts:
                cv2.circle(frame_bgr, pt, 2, (0, 255, 0), -1)
        if left_eye_pts:
            for pt in left_eye_pts:
                cv2.circle(frame_bgr, pt, 2, (0, 255, 255), -1)
        return frame_bgr