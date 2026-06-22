import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from collections import deque

class ViolenceDetector:
    def __init__(self, model_path, input_size=(224, 224), sequence_length=20, feature_model_path=None):
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.feature_buffer = deque(maxlen=self.sequence_length)
        
        try:
            self.model = load_model(model_path)
            print("Violence model loaded successfully.")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            self.model = None

        try:
            # Use VGG16 conv features (7x7x512) to match model's expected input shape
            vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(self.input_size[0], self.input_size[1], 3))
            self.feature_extractor = Model(inputs=vgg16_base.input, outputs=vgg16_base.output)
            print("Feature extractor (VGG16) loaded successfully.")
        except Exception as e:
            print(f"Error loading VGG16: {e}")
            self.feature_extractor = None

    def buffer_len(self):
        return len(self.feature_buffer)

    def reset(self):
        self.feature_buffer.clear()

    def predict_frame(self, frame):
        if self.model is None or self.feature_extractor is None:
            return None

        try:
            h, w = frame.shape[:2]
            side = min(h, w)
            y0 = (h - side) // 2
            x0 = (w - side) // 2
            frame_cropped = frame[y0:y0+side, x0:x0+side]
            frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.input_size)
            input_tensor = np.expand_dims(frame_resized, axis=0)
            input_tensor = vgg_preprocess(input_tensor)

            features = self.feature_extractor.predict(input_tensor, verbose=0)  # (1, 7, 7, 512)
            self.feature_buffer.append(features[0])  # store (7, 7, 512)

            if len(self.feature_buffer) < self.sequence_length:
                return None

            # Stack to (seq_len, 7, 7, 512) then expand to (1, seq_len, 7, 7, 512)
            sequence = np.stack(list(self.feature_buffer), axis=0)
            sequence_reshaped = np.expand_dims(sequence, axis=0)

            prob = self.model.predict(sequence_reshaped, verbose=0)
            return float(np.ravel(prob)[0])
        except Exception as e:
            print(f"Prediction with sequence failed: {e}")
            return None