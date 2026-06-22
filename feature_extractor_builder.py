# feature_extractor_builder.py
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf
import os

def build_and_save(out_path="models/feature_extractor_model.h5", input_shape=(224,224,3)):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    feat_model = Model(inputs=base.input, outputs=x)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    feat_model.save(out_path)
    print("Saved feature extractor to", out_path)

if __name__ == "__main__":
    build_and_save()
