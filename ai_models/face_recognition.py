"""
Face Recognition Module using CNN and OpenCV

"""
import cv2
import numpy as np
import os
from pathlib import Path
import pickle

from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical


class FaceRecognition:
    _clahe_detect  = None   # CLAHE for detection  (clipLimit=3.0, grid=8x8)
    _clahe_patch   = None   # CLAHE for patch norm (clipLimit=3.0, grid=4x4)
    _hog64         = None   # HOGDescriptor for 64×64 full-face
    _hog32         = None   # HOGDescriptor for 32×32 zone patches

    @classmethod
    def _get_clahe_detect(cls):
        if cls._clahe_detect is None:
            cls._clahe_detect = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return cls._clahe_detect

    @classmethod
    def _get_clahe_patch(cls):
        if cls._clahe_patch is None:
            cls._clahe_patch = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        return cls._clahe_patch

    @classmethod
    def _get_hog64(cls):
        if cls._hog64 is None:
            cls._hog64 = cv2.HOGDescriptor(
                _winSize=(64, 64), _blockSize=(16, 16),
                _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9,
            )
        return cls._hog64

    @classmethod
    def _get_hog32(cls):
        if cls._hog32 is None:
            cls._hog32 = cv2.HOGDescriptor(
                _winSize=(32, 32), _blockSize=(16, 16),
                _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9,
            )
        return cls._hog32

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.label_encoder = {}  # Maps user_id to class index
        self.face_cascade = None
        self.img_size = (128, 128)
        self.load_cascade()

        # Automatically load the latest model if no path specified
        if model_path is None:
            self.auto_load_latest_model()
        elif os.path.exists(model_path):
            self.load_model(model_path)

    def load_cascade(self):
        """Load Haar Cascade for face detection"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            print(f"Error loading cascade: {e}")

    # ─────────────────────────────────────────────────────────
    # Preprocessing helpers
    # ─────────────────────────────────────────────────────────

    def apply_clahe(self, image):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve
        face detection under poor / uneven lighting conditions.
        Works on BGR images; returns BGR.
        Uses the cached CLAHE object to avoid repeated construction overhead.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = self._get_clahe_detect()
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def normalize_brightness(self, image):
        """
        Gamma correction to brighten very dark frames captured by low-quality webcams.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 80:
            # Dark frame — boost brightness
            gamma = 0.5
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                               for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
        return image

    # ─────────────────────────────────────────────────────────
    # Face detection
    # ─────────────────────────────────────────────────────────

    def detect_faces(self, image):
        """
        Detect faces — fast two-pass approach.
        Pass 1 (strict): scaleFactor=1.1, minNeighbors=4, minSize=(40,40)
        Pass 2 (relaxed): scaleFactor=1.05, minNeighbors=2, minSize=(30,30)
        Uses cached CLAHE; tries enhanced then raw gray as fallback.
        Returns a list of (x, y, w, h) bounding boxes.
        """
        if self.face_cascade is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use cached CLAHE — avoids repeated object creation
        gray_enhanced = self._get_clahe_detect().apply(gray)

        # Fast two-pass: strict first (usually succeeds), relaxed as fallback
        detection_configs = [
            (gray_enhanced, 1.1,  4, (40, 40)),   # Pass 1 – strict + enhanced
            (gray_enhanced, 1.05, 2, (30, 30)),   # Pass 2 – relaxed + enhanced
            (gray,          1.1,  3, (40, 40)),   # Pass 3 – raw fallback
        ]
        for detect_gray, scale, neighbors, min_size in detection_configs:
            faces = self.face_cascade.detectMultiScale(
                detect_gray,
                scaleFactor=scale,
                minNeighbors=neighbors,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(faces) > 0:
                return faces

        return []

    def preprocess_face(self, face_img):
        """Preprocess face image for recognition"""
        face_resized = cv2.resize(face_img, self.img_size)
        face_normalized = face_resized.astype('float32') / 255.0
        return face_normalized

    # ─────────────────────────────────────────────────────────
    # Encoding extraction (used for Pearson-correlation matching)
    # ─────────────────────────────────────────────────────────

    def extract_face_encoding(self, image):
        """
        Extract face encoding from image.

        Tries, in order:
          1. Raw image.
          2. CLAHE-enhanced image.
          3. Brightness-normalised image.
          4. Horizontally flipped image (webcam mirror mismatch).

        Returns a float32 (128×128×3) numpy array, or None if no face found.
        """
        candidates = [image]

        # CLAHE variant
        try:
            candidates.append(self.apply_clahe(image))
        except Exception:
            pass

        # Brightness variant
        try:
            candidates.append(self.normalize_brightness(image))
        except Exception:
            pass

        # Horizontally flipped (mirror)
        candidates.append(cv2.flip(image, 1))

        for candidate in candidates:
            faces = self.detect_faces(candidate)
            if len(faces) > 0:
                # Pick the largest detected face
                largest = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest

                # Add a small padding around the face for better context
                pad = int(min(w, h) * 0.10)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(candidate.shape[1], x + w + pad)
                y2 = min(candidate.shape[0], y + h + pad)

                face_img = candidate[y1:y2, x1:x2]
                return self.preprocess_face(face_img)

        return None

    def preprocess_for_matching(self, image):
        """
        Full preprocessing pipeline before Pearson-correlation matching:
          brightness normalisation → CLAHE → face encoding.
        Returns encoding or None.
        """
        try:
            img = self.normalize_brightness(image)
            img = self.apply_clahe(img)
            return self.extract_face_encoding(img)
        except Exception:
            return self.extract_face_encoding(image)

    # ───────────────────────────────────────────────────────────
    # HOG + LBP feature encoding (identity-discriminative)
    # ───────────────────────────────────────────────────────────

    def _crop_face_gray(self, image):
        """
        Detect, crop and return a normalised grayscale face patch (64x64).
        Optimised: tries only 2 image variants (original + CLAHE) before
        falling back to flipped — avoids expensive redundant detections.
        Uses cached CLAHE patch object.
        Returns None if no face can be found.
        """
        # Lean candidate list — original + CLAHE-enhanced (covers 95% of cases);
        # flipped mirror only as last resort.
        candidates = [image]
        try:
            candidates.append(self.apply_clahe(image))
        except Exception:
            pass
        try:
            candidates.append(cv2.flip(image, 1))
        except Exception:
            pass

        for cand in candidates:
            try:
                faces = self.detect_faces(cand)
                if not len(faces):
                    continue
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                # Generous padding — helps when face is slightly off-centre
                pad = int(min(w, h) * 0.15)
                x1 = max(0, x - pad);  y1 = max(0, y - pad)
                x2 = min(cand.shape[1], x + w + pad)
                y2 = min(cand.shape[0], y + h + pad)
                face = cand[y1:y2, x1:x2]
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if face.ndim == 3 else face
                gray = cv2.resize(gray, (64, 64))
                # Use cached CLAHE patch object
                gray = self._get_clahe_patch().apply(gray)
                return gray
            except Exception:
                continue
        return None

    def extract_hog_encoding(self, image):
        """
        Extract HOG descriptor from the largest detected face.
        Uses cached HOGDescriptor to avoid per-call construction overhead.
        Returns a 1-D float64 numpy array, or None if no face is found.
        """
        gray = self._crop_face_gray(image)
        if gray is None:
            return None
        try:
            feats = self._get_hog64().compute(gray)   # shape (1764, 1)
            return feats.flatten().astype(np.float64)
        except Exception as e:
            print(f'[HOG] extraction failed: {e}')
            return None

    def extract_lbp_encoding(self, image):
        """
        Manual LBP (Local Binary Pattern) histogram — no opencv-contrib needed.
        Divides the 64×64 face into 4×4 = 16 blocks and computes a 256-bin
        LBP histogram per block, yielding a 4096-element descriptor.

        LBP measures local texture/micro-structure around each pixel, which
        is highly person-specific.
        """
        gray = self._crop_face_gray(image)
        if gray is None:
            return None

        try:
            h, w = gray.shape
            # Basic uniform LBP
            neighbors = 8
            lbp = np.zeros_like(gray, dtype=np.uint8)
            angles = [0, 45, 90, 135, 180, 225, 270, 315]
            for idx, angle in enumerate(angles):
                rad = np.deg2rad(angle)
                dx = int(np.round(np.cos(rad)))
                dy = int(np.round(-np.sin(rad)))
                shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
                lbp += ((shifted >= gray).astype(np.uint8) << idx)

            # Split into 4×4 grid and build histogram per cell
            cells_y, cells_x = 4, 4
            cy, cx = h // cells_y, w // cells_x
            hists = []
            for i in range(cells_y):
                for j in range(cells_x):
                    cell = lbp[i*cy:(i+1)*cy, j*cx:(j+1)*cx]
                    hist, _ = np.histogram(cell.flatten(), bins=256, range=(0, 256))
                    hist = hist.astype(np.float64)
                    norm = hist.sum()
                    if norm > 0:
                        hist /= norm
                    hists.append(hist)
            return np.concatenate(hists).astype(np.float64)   # 4096 dims
        except Exception as e:
            print(f'[LBP] extraction failed: {e}')
            return None

    def extract_combined_encoding(self, image):
        """
        Combine HOG + LBP into a single concatenated descriptor.
        .
        """
        gray = self._crop_face_gray(image)
        if gray is None:
            return None
        return self._combined_from_gray(gray)

    def _combined_from_gray(self, gray):
        """Compute HOG+LBP from an already-cropped 64×64 grayscale patch."""
        parts = []
        # HOG — cached descriptor
        try:
            feats = self._get_hog64().compute(gray)
            parts.append(feats.flatten().astype(np.float64))
        except Exception:
            pass
        # LBP — manual uniform LBP
        try:
            lbp = np.zeros_like(gray, dtype=np.uint8)
            for idx, angle in enumerate([0, 45, 90, 135, 180, 225, 270, 315]):
                rad = np.deg2rad(angle)
                dx = int(np.round(np.cos(rad)))
                dy = int(np.round(-np.sin(rad)))
                shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
                lbp += ((shifted >= gray).astype(np.uint8) << idx)
            h, w = gray.shape
            cells_y, cells_x = 4, 4
            cy, cx = h // cells_y, w // cells_x
            hists = []
            for i in range(cells_y):
                for j in range(cells_x):
                    cell = lbp[i*cy:(i+1)*cy, j*cx:(j+1)*cx]
                    hist, _ = np.histogram(cell.flatten(), bins=256, range=(0, 256))
                    hist = hist.astype(np.float64)
                    norm = hist.sum()
                    if norm > 0:
                        hist /= norm
                    hists.append(hist)
            parts.append(np.concatenate(hists).astype(np.float64))
        except Exception:
            pass
        if not parts:
            return None
        return np.concatenate(parts).astype(np.float64)

    @staticmethod
    def cosine_similarity(a, b):
        """Cosine similarity between two 1-D float64 vectors."""
        dot  = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 1e-10 else 0.0

    # ───────────────────────────────────────────────────────────
    # Glasses-invariant multi-region feature extraction
    # ───────────────────────────────────────────────────────────
    # Glasses ONLY affect the eye-band (rows ~25%–55% of face height).
    # The following zones are stable with/without glasses:
    #   Zone A — forehead (top 25 % of face)
    #   Zone B — nose bridge + nose tip (rows 45–65%)
    #   Zone C — mouth + chin (rows 65–100%)
    #   Zone D — left cheek  (left quarter, rows 35–70%)
    #   Zone E — right cheek (right quarter, rows 35–70%)
    # Extracting HOG+LBP separately from these zones and combining
    # them into one descriptor gives glasses-invariant identity matching.

    def _hog_from_patch(self, patch):
        """Compute HOG descriptor from a pre-cropped grayscale patch.
        Uses cached HOGDescriptor — no object creation on every call."""
        try:
            p = cv2.resize(patch, (32, 32))
            return self._get_hog32().compute(p).flatten().astype(np.float64)
        except Exception:
            return None

    def _lbp_from_patch(self, patch):
        """Compute LBP histogram from a pre-cropped grayscale patch."""
        try:
            p = cv2.resize(patch, (32, 32))
            lbp = np.zeros_like(p, dtype=np.uint8)
            for idx, angle in enumerate([0, 45, 90, 135, 180, 225, 270, 315]):
                rad = np.deg2rad(angle)
                dx = int(np.round(np.cos(rad)))
                dy = int(np.round(-np.sin(rad)))
                shifted = np.roll(np.roll(p, dy, axis=0), dx, axis=1)
                lbp += ((shifted >= p).astype(np.uint8) << idx)
            # 2×2 grid of 256-bin histograms → 1024 dims
            hists = []
            for i in range(2):
                for j in range(2):
                    cell = lbp[i*16:(i+1)*16, j*16:(j+1)*16]
                    h, _ = np.histogram(cell.flatten(), bins=256, range=(0, 256))
                    h = h.astype(np.float64)
                    s = h.sum()
                    if s > 0:
                        h /= s
                    hists.append(h)
            return np.concatenate(hists).astype(np.float64)
        except Exception:
            return None

    def extract_glasses_invariant_encoding(self, image):
        """
        Extract a glasses-invariant face descriptor by combining HOG+LBP
        features from five facial zones that are NOT affected by glasses.

        Zone layout on a 64×64 normalized face:
          A — forehead    rows  0–16  (top 25%)      weight 1.5
          B — nose area   rows 29–42  (45%–65%)      weight 2.0
          C — lower face  rows 42–64  (65%–100%)     weight 2.5
          D — left cheek  cols  0–16, rows 22–45     weight 1.0
          E — right cheek cols 48–64, rows 22–45     weight 1.0

        Eye band (rows 16–29, 25%–45%) is intentionally excluded.

        Returns a float64 vector, or None if no face detected.
        """
        gray = self._crop_face_gray(image)   # 64×64 equalised grayscale
        if gray is None:
            return None

        try:
            parts = []

            # Zone A — forehead (glasses-free)
            zone_a = gray[0:16, :]
            hog_a = self._hog_from_patch(zone_a)
            lbp_a = self._lbp_from_patch(zone_a)
            if hog_a is not None:
                parts.append(hog_a * 1.5)
            if lbp_a is not None:
                parts.append(lbp_a * 1.5)

            # Zone B — nose bridge / tip (glasses-free)
            zone_b = gray[29:42, 16:48]
            hog_b = self._hog_from_patch(zone_b)
            lbp_b = self._lbp_from_patch(zone_b)
            if hog_b is not None:
                parts.append(hog_b * 2.0)
            if lbp_b is not None:
                parts.append(lbp_b * 2.0)

            # Zone C — mouth + chin (most discriminative, always glasses-free)
            zone_c = gray[42:64, :]
            hog_c = self._hog_from_patch(zone_c)
            lbp_c = self._lbp_from_patch(zone_c)
            if hog_c is not None:
                parts.append(hog_c * 2.5)
            if lbp_c is not None:
                parts.append(lbp_c * 2.5)

            # Zone D — left cheek
            zone_d = gray[22:45, 0:16]
            hog_d = self._hog_from_patch(zone_d)
            if hog_d is not None:
                parts.append(hog_d * 1.0)

            # Zone E — right cheek
            zone_e = gray[22:45, 48:64]
            hog_e = self._hog_from_patch(zone_e)
            if hog_e is not None:
                parts.append(hog_e * 1.0)

            if not parts:
                return None

            combined = np.concatenate(parts).astype(np.float64)
            # L2-normalise so cosine similarity is well-scaled
            norm = np.linalg.norm(combined)
            if norm > 1e-10:
                combined /= norm
            return combined

        except Exception as e:
            print(f'[GlassesInvariant] extraction failed: {e}')
            return None

    def extract_robust_encoding(self, image):
        """
        Best-of-both-worlds descriptor: concatenates the glasses-invariant
        multi-zone encoding with the standard full-face HOG+LBP encoding.

        The multi-zone part gives stability across glasses on/off.
        The full-face part gives discriminative power to reject non-users.

        Returns a float64 vector, or None if no face detected.
        """
        gi  = self.extract_glasses_invariant_encoding(image)
        std = self.extract_combined_encoding(image)

        if gi is None and std is None:
            return None
        if gi is None:
            return std
        if std is None:
            return gi

        # Weight: glasses-invariant zones carry 60%, standard full-face 40%
        gi_norm  = gi  / (np.linalg.norm(gi)  + 1e-10)
        std_norm = std / (np.linalg.norm(std)  + 1e-10)
        combined = np.concatenate([gi_norm * 0.60, std_norm * 0.40])
        return combined.astype(np.float64)


    @staticmethod
    def extract_cnn_embedding(cnn_model, face_batch):
        """
        Extract penultimate Dense-512 layer embeddings from a loaded Keras CNN.

        Unlike softmax (forced to sum=1 across N known classes, so any face gets
        assigned to the nearest class), the 512-dim embedding lives in a learned
        metric space where:
          • same-person pairs   → cosine similarity ~ 0.85-0.99
          • different-person    → cosine similarity ~ 0.20-0.65
        This enables open-set rejection of non-registered users even when the
        softmax layer is fooled by a 2-class closed-set classifier.

        Args:
            cnn_model  – loaded keras.Model (with classification head intact)
            face_batch – np.ndarray shape (N, H, W, C)
        Returns:
            np.ndarray shape (N, 512) or None on failure.
        """
        try:
            from tensorflow.keras import Model as _KModel
            feat = _KModel(inputs=cnn_model.inputs,
                           outputs=cnn_model.layers[-2].output)
            return feat.predict(face_batch, verbose=0)
        except Exception as e:
            print(f'[EmbeddingExtract] {e}')
            return None

    # ─────────────────────────────────────────────────────────
    # CNN model-based recognition (legacy — used by teacher dataset pipeline)
    # ─────────────────────────────────────────────────────────

    def recognize_face(self, image, threshold=0.85):
        """Recognize face in image using CNN model (requires trained model)"""
        if self.model is None:
            return None, 0.0

        face_encoding = self.extract_face_encoding(image)

        if face_encoding is None:
            return None, 0.0

        # Predict
        face_input = np.expand_dims(face_encoding, axis=0)
        predictions = self.model.predict(face_input, verbose=0)

        max_prob = np.max(predictions[0])
        predicted_class = np.argmax(predictions[0])

        if max_prob >= threshold:
            user_id = None
            for uid, class_idx in self.label_encoder.items():
                if class_idx == predicted_class:
                    user_id = uid
                    break

            if user_id == -1:
                return None, float(max_prob)

            return user_id, float(max_prob)

        return None, float(max_prob)

    # ─────────────────────────────────────────────────────────
    # CNN training / persistence helpers
    # ─────────────────────────────────────────────────────────

    def create_model(self, num_classes):
        """Create CNN model for face recognition"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """Train the face recognition model"""
        num_classes = len(np.unique(y_train))

        if self.model is None:
            self.create_model(num_classes)
        else:
            existing_classes = self.model.output_shape[-1]
            if existing_classes != num_classes:
                self.create_model(num_classes)

        y_train_cat = to_categorical(y_train, num_classes=num_classes)

        if X_val is not None and y_val is not None:
            y_val_cat = to_categorical(y_val, num_classes=num_classes)
            validation_data = (X_val, y_val_cat)
        else:
            validation_data = None

        history = self.model.fit(
            X_train, y_train_cat,
            epochs=epochs,
            batch_size=32,
            validation_data=validation_data,
            verbose=1
        )

        return history

    def save_model(self, path):
        """Save trained model"""
        if self.model:
            self.model.save(path)

    def auto_load_latest_model(self):
        """Automatically load the latest trained model"""
        try:
            from django.conf import settings
            models_dir = os.path.join(settings.MEDIA_ROOT, 'models')

            if not os.path.exists(models_dir):
                return False

            model_files = [f for f in os.listdir(models_dir) if f.startswith('face_') and f.endswith('.h5')]

            if not model_files:
                return False

            latest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(models_dir, f)))
            model_path = os.path.join(models_dir, latest_model)

            success = self.load_model(model_path)

            if success:
                encoder_path = model_path.replace('.h5', '_labels.pkl')
                if os.path.exists(encoder_path):
                    with open(encoder_path, 'rb') as f:
                        self.label_encoder = pickle.load(f)

                print(f"Loaded face recognition model: {latest_model}")
                return True
        except Exception as e:
            print(f"Error auto-loading model: {e}")

        return False

    def load_model(self, path):
        """Load trained model"""
        if os.path.exists(path):
            self.model = load_model(path)
            self.model_path = path
            return True
        return False

    def capture_from_camera(self, duration=5):
        """Capture face images from camera"""
        cap = cv2.VideoCapture(0)
        images = []

        start_time = cv2.getTickCount()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.detect_faces(frame)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_img = frame[y:y+h, x:x+w]
                images.append(face_img)

            cv2.imshow('Face Capture', frame)

            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed >= duration:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return images

