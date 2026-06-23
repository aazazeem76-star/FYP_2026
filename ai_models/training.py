"""
Training Module for AI Models
"""
import os
import numpy as np
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from .face_recognition import FaceRecognition
from .retina_recognition import RetinaRecognition


class ModelTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.face_recognizer = FaceRecognition()
        self.retina_recognizer = RetinaRecognition()
    
    def load_dataset(self, dataset_type='face'):
        """Load dataset from directory"""
        X = []
        y = []
        label_map = {}
        
        dataset_dir = self.dataset_path / dataset_type
        
        if not dataset_dir.exists():
            return np.array([]), np.array([]), {}
        
        # Iterate through user directories
        for idx, user_dir in enumerate(dataset_dir.iterdir()):
            if not user_dir.is_dir():
                continue
            
            user_id = user_dir.name
            label_map[idx] = user_id
            
            # Load images
            for img_path in user_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        if dataset_type == 'face':
                            processed = self.face_recognizer.extract_face_encoding(img)
                        else:
                            processed = self.retina_recognizer.extract_retina_encoding(img)
                        
                        if processed is not None:
                            X.append(processed)
                            y.append(idx)
        
        return np.array(X), np.array(y), label_map
    
    def train_face_model(self, epochs=50, test_size=0.2):
        """Train face recognition model"""
        print("Loading face dataset...")
        X, y, label_map = self.load_dataset('face')
        
        if len(X) == 0:
            print("No face data found!")
            return None, None
        
        print(f"Dataset loaded: {len(X)} samples, {len(np.unique(y))} classes")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print("Training face recognition model...")
        history = self.face_recognizer.train_model(
            X_train, y_train, X_val, y_val, epochs=epochs
        )
        
        # Evaluate
        val_loss, val_acc = self.face_recognizer.model.evaluate(
            X_val, 
            self._to_categorical(y_val, len(np.unique(y))),
            verbose=0
        )
        
        print(f"Validation Accuracy: {val_acc * 100:.2f}%")
        
        return history, label_map
    
    def train_retina_model(self, epochs=50, test_size=0.2):
        """Train retina recognition model"""
        print("Loading retina dataset...")
        X, y, label_map = self.load_dataset('retina')
        
        if len(X) == 0:
            print("No retina data found!")
            return None, None
        
        print(f"Dataset loaded: {len(X)} samples, {len(np.unique(y))} classes")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print("Training retina recognition model...")
        history = self.retina_recognizer.train_model(
            X_train, y_train, X_val, y_val, epochs=epochs
        )
        
        # Evaluate
        val_loss, val_acc = self.retina_recognizer.model.evaluate(
            X_val,
            self._to_categorical(y_val, len(np.unique(y))),
            verbose=0
        )
        
        print(f"Validation Accuracy: {val_acc * 100:.2f}%")
        
        return history, label_map
    
    def save_models(self, face_path, retina_path):
        """Save trained models"""
        if self.face_recognizer.model:
            self.face_recognizer.save_model(face_path)
            print(f"Face model saved to {face_path}")
        
        if self.retina_recognizer.model:
            self.retina_recognizer.save_model(retina_path)
            print(f"Retina model saved to {retina_path}")
    
    def _to_categorical(self, y, num_classes):
        """Convert labels to categorical"""
        try:
            from keras.utils import to_categorical
            return to_categorical(y, num_classes)
        except:
            # Manual implementation if TensorFlow not available
            categorical = np.zeros((len(y), num_classes))
            for i, label in enumerate(y):
                categorical[i, label] = 1
            return categorical
    
    def create_sample_dataset(self, user_id, dataset_type='face', num_samples=50):
        """Create sample dataset by capturing from camera"""
        print(f"Creating {dataset_type} dataset for user {user_id}")
        print(f"Capturing {num_samples} samples...")
        
        # Create user directory
        user_dir = self.dataset_path / dataset_type / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Capture images
        if dataset_type == 'face':
            images = self.face_recognizer.capture_from_camera(duration=num_samples // 2)
        else:
            images = self.retina_recognizer.capture_from_camera(duration=num_samples // 2)
        
        # Save images
        for idx, img in enumerate(images[:num_samples]):
            img_path = user_dir / f"{dataset_type}_{idx:04d}.jpg"
            cv2.imwrite(str(img_path), img)
        
        print(f"Saved {len(images[:num_samples])} images to {user_dir}")
        
        return len(images[:num_samples])
