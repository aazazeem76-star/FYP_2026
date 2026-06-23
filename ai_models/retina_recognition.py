"""
Retina Recognition Module using Pattern Matching
"""
import cv2
import numpy as np
import os
from pathlib import Path

try:
    from tensorflow import keras
    from keras.models import Sequential, load_model
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from keras.utils import img_to_array
except ImportError:
    pass


class RetinaRecognition:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.eye_cascade = None
        self.img_size = (128, 128)
        self.load_cascade()
    
    def load_cascade(self):
        """Load Haar Cascade for eye detection"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            print(f"Error loading eye cascade: {e}")
    
    def create_model(self, num_classes):
        """Create CNN model for retina recognition"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
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
    
    def detect_eyes(self, image):
        """Detect eyes in an image"""
        if self.eye_cascade is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        return eyes
    
    def preprocess_retina(self, retina_img):
        """Preprocess retina/eye image"""
        # Convert to grayscale
        if len(retina_img.shape) == 3:
            gray = cv2.cvtColor(retina_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = retina_img
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Resize
        resized = cv2.resize(enhanced, self.img_size)
        
        # Convert back to 3 channels for model
        colored = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        
        # Normalize
        normalized = colored.astype('float32') / 255.0
        
        # Convert to array
        retina_array = img_to_array(normalized)
        
        return retina_array
    
    def extract_retina_features(self, image):
        """Extract retina features using image processing"""
        # Circular Hough Transform for pupil detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect circles (pupil)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50
        )
        
        features = {}
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Get the first detected circle (pupil)
            x, y, r = circles[0, 0]
            features['pupil_center'] = (x, y)
            features['pupil_radius'] = r
            
            # Extract region of interest
            roi = gray[max(0, y-r*2):y+r*2, max(0, x-r*2):x+r*2]
            features['roi'] = roi
        
        return features
    
    def extract_retina_encoding(self, image):
        """Extract retina encoding from image"""
        eyes = self.detect_eyes(image)
        
        if len(eyes) == 0:
            return None
        
        # Get the first eye
        x, y, w, h = eyes[0]
        
        # Extract eye region
        eye_img = image[y:y+h, x:x+w]
        
        # Preprocess
        retina_processed = self.preprocess_retina(eye_img)
        
        return retina_processed
    
    def recognize_retina(self, image, threshold=0.75):
        """Recognize retina pattern in image"""
        if self.model is None:
            return None, 0.0
        
        retina_encoding = self.extract_retina_encoding(image)
        
        if retina_encoding is None:
            return None, 0.0
        
        # Predict
        retina_input = np.expand_dims(retina_encoding, axis=0)
        predictions = self.model.predict(retina_input, verbose=0)
        
        # Get best match
        max_prob = np.max(predictions[0])
        predicted_class = np.argmax(predictions[0])
        
        if max_prob >= threshold:
            return predicted_class, float(max_prob)
        
        return None, float(max_prob)
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """Train the retina recognition model"""
        if self.model is None:
            num_classes = len(np.unique(y_train))
            self.create_model(num_classes)
        
        # Convert labels to categorical
        from keras.utils import to_categorical
        y_train_cat = to_categorical(y_train)
        
        # Train
        if X_val is not None and y_val is not None:
            y_val_cat = to_categorical(y_val)
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
    
    def load_model(self, path):
        """Load trained model"""
        if os.path.exists(path):
            self.model = load_model(path)
            return True
        return False
    
    def capture_from_camera(self, duration=5):
        """Capture retina/eye images from camera"""
        cap = cv2.VideoCapture(0)
        images = []
        
        start_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            eyes = self.detect_eyes(frame)
            
            # Draw rectangles around eyes
            for (x, y, w, h) in eyes:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Extract eye
                eye_img = frame[y:y+h, x:x+w]
                images.append(eye_img)
            
            cv2.imshow('Retina Capture', frame)
            
            # Check duration
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed >= duration:
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return images
