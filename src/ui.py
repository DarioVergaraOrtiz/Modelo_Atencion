import cv2
from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from gaze_mediapipe import estimate_gaze
import mediapipe as mp
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medidor de Atención")
        self.setGeometry(100, 100, 1000, 500)

        # Labels para cámara y video
        self.cam_label = QLabel(self)
        self.cam_label.setGeometry(10, 10, 480, 360)

        self.vid_label = QLabel(self)
        self.vid_label.setGeometry(500, 10, 480, 360)

        # Label de estadísticas
        self.stats_label = QLabel(self)
        self.stats_label.setGeometry(10, 380, 970, 100)
        self.stats_label.setWordWrap(True)

        # Botón para cargar video
        btn = QPushButton("Cargar Video", self)
        btn.setGeometry(10, 320, 120, 30)
        btn.clicked.connect(self.load_video)

        # Inicializar cámara y video
        self.cap_cam = cv2.VideoCapture(0)
        self.cap_vid = None

        # MediaPipe Face Mesh
        self.fm = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Definición de ROI en coordenadas del frame de la cámara (480×360)
        # Ajusta estos valores para encuadrar tu región “atenta”
        self.roi = { 'x1': 100, 'y1': 50, 'x2': 380, 'y2': 310 }

        # Contadores
        self.total_frames = 0
        self.attentive_frames = 0

        # Timer para actualizar
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)  # ~33 FPS

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecciona video")
        if path:
            if self.cap_vid:
                self.cap_vid.release()
            self.cap_vid = cv2.VideoCapture(path)

    def update_frames(self):
        # Lectura de cámara
        ret, frame = self.cap_cam.read()
        if ret:
            gaze_vec, vis_cam = estimate_gaze(frame, self.fm)

            # Dibujar ROI y punto de mirada
            if gaze_vec is not None:
                self.total_frames += 1
                h_cam, w_cam, _ = frame.shape
                # Punto central de mirada en coords de cámara
                cx = w_cam // 2 + int(gaze_vec[0])
                cy = h_cam // 2 + int(gaze_vec[1])

                # Contabilizar si está dentro de ROI
                if (self.roi['x1'] <= cx <= self.roi['x2'] and
                    self.roi['y1'] <= cy <= self.roi['y2']):
                    self.attentive_frames += 1

                # Dibujar para depuración
                cv2.circle(vis_cam, (cx, cy), 4, (0,255,255), -1)
            # Dibujar ROI siempre
            cv2.rectangle(vis_cam,
                          (self.roi['x1'], self.roi['y1']),
                          (self.roi['x2'], self.roi['y2']),
                          (255, 0, 0), 2)

            self._display_image(vis_cam, self.cam_label)

        # Lectura de video
        if self.cap_vid:
            ret2, frame2 = self.cap_vid.read()
            if not ret2:
                self.cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret2, frame2 = self.cap_vid.read()
            if ret2:
                self._display_image(frame2, self.vid_label)

        # Actualizar texto de estadísticas
        if self.total_frames > 0:
            pct = self.attentive_frames / self.total_frames * 100
            total_s = self.total_frames * 0.03
            att_s = self.attentive_frames * 0.03
            self.stats_label.setText(
                f"Frames: {self.total_frames}, Atención dentro de ROI: {self.attentive_frames} ({pct:.1f}%)\n"
                f"Tiempo total: {total_s:.1f}s, Atento: {att_s:.1f}s"
            )

    def _display_image(self, frame, label):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            label.width(), label.height()
        ))

