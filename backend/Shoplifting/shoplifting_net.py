"""
Shoplifting Detection - Gate_Flow_SlowFast 3D-CNN Model
Based on: https://github.com/1amitos1/Shoplifting-Detection

Adapted for TensorFlow 2.20+ / Keras 3.x

Architecture:
  - SlowFast Network with lateral connections
  - Slow path: 4 frames (stride=16), learns deep spatial features
  - Fast path: 64 frames, learns temporal/motion features
  - MobileNet-SSD inspired lightweight Conv3D blocks
  - 3-class softmax output: Bag, Clothes, Normal
"""
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model


# ─── Custom Layers (Keras 3 compatible) ─────────────────────────────────

class ExtractRGB(layers.Layer):
    """Extract first 3 channels (RGB) from input."""
    def call(self, x):
        return x[..., :3]

class TemporalSubsample(layers.Layer):
    """Subsample frames along temporal axis with given stride."""
    def __init__(self, max_frames=64, stride=16, **kwargs):
        super().__init__(**kwargs)
        self.max_frames = max_frames
        self.stride = stride

    def call(self, x):
        indices = tf.range(0, self.max_frames, self.stride)
        return tf.gather(x, indices, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"max_frames": self.max_frames, "stride": self.stride})
        return config

class LateralSample(layers.Layer):
    """Sample frames for lateral connections (fast→slow)."""
    def __init__(self, stride=18, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride

    def call(self, x):
        t = tf.shape(x)[1]
        indices = tf.range(0, t, self.stride)
        return tf.gather(x, indices, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"stride": self.stride})
        return config


class ShopliftingNet:
    """Gate_Flow_SlowFast 3D-CNN for shoplifting detection."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    # ─── Network Building Blocks ─────────────────────────────────────────

    @staticmethod
    def _conv3d_block(x, filters, kernel, strides=(1, 1, 1),
                      activation='relu', padding='same'):
        """Single Conv3D layer with he_normal initialization."""
        return layers.Conv3D(
            filters, kernel_size=kernel, strides=strides,
            kernel_initializer='he_normal', activation=activation,
            padding=padding
        )(x)

    def merging_block(self, x):
        """Merging block: processes combined slow+fast features."""
        x = self._conv3d_block(x, 64, (1, 3, 3))
        x = self._conv3d_block(x, 64, (3, 1, 1))
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = self._conv3d_block(x, 64, (1, 3, 3))
        x = self._conv3d_block(x, 64, (3, 1, 1))
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = self._conv3d_block(x, 128, (1, 3, 3))
        x = self._conv3d_block(x, 128, (3, 1, 1))
        x = layers.MaxPooling3D(pool_size=(2, 3, 3))(x)

        return x

    def get_Flow_gate_fast_path(self, fast_input):
        """
        Fast pathway: processes all 64 frames.
        Extracts temporal features with lateral connections to slow path.
        """
        rgb = ExtractRGB()(fast_input)
        connection_dic = {}

        # ── Stage 1 ──
        rgb = self._conv3d_block(rgb, 16, (1, 3, 3))
        rgb = self._conv3d_block(rgb, 16, (3, 1, 1))
        rgb = layers.MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        rgb = self._conv3d_block(rgb, 16, (1, 3, 3))
        rgb = self._conv3d_block(rgb, 16, (3, 1, 1))
        rgb = layers.MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        # Lateral connection 1
        lateral = LateralSample(stride=18, name="con_1")(rgb)
        connection_dic["con-1"] = lateral

        # ── Stage 2 ──
        rgb = self._conv3d_block(rgb, 32, (1, 3, 3))
        rgb = self._conv3d_block(rgb, 32, (3, 1, 1))
        rgb = layers.MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        rgb = self._conv3d_block(rgb, 32, (1, 3, 3))
        rgb = self._conv3d_block(rgb, 32, (3, 1, 1))
        rgb = layers.MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        # Lateral connection 2
        lateral = LateralSample(stride=18, name="con_2")(rgb)
        connection_dic["con-2"] = lateral

        return rgb, connection_dic

    def get_Flow_gate_slow_path(self, slow_input, connection_dic):
        """
        Slow pathway: processes 4 frames (subsampled by stride=16).
        Receives lateral connections from the fast path.
        """
        rgb = ExtractRGB()(slow_input)
        con_1 = connection_dic.get('con-1')
        con_2 = connection_dic.get('con-2')

        # ── Stage 1 ──
        rgb = self._conv3d_block(rgb, 16, (1, 3, 3))
        rgb = self._conv3d_block(rgb, 16, (3, 1, 1))
        rgb = layers.MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        rgb = self._conv3d_block(rgb, 16, (1, 3, 3))
        rgb = self._conv3d_block(rgb, 16, (3, 1, 1))
        rgb = layers.MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        # Add lateral connection 1
        rgb = layers.Add(name="connection_1_rgb")([rgb, con_1])

        # ── Stage 2 ──
        rgb = self._conv3d_block(rgb, 32, (1, 3, 3))
        rgb = self._conv3d_block(rgb, 32, (3, 1, 1))
        rgb = layers.MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        rgb = self._conv3d_block(rgb, 32, (1, 3, 3))
        rgb = self._conv3d_block(rgb, 32, (3, 1, 1))
        rgb = layers.MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        # Add lateral connection 2
        x = layers.Add(name="connection_2_rgb")([rgb, con_2])
        x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)

        # ── Merging Block (slow path) ──
        x = self._conv3d_block(x, 64, (1, 3, 3))
        x = self._conv3d_block(x, 64, (3, 1, 1))
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = self._conv3d_block(x, 64, (1, 3, 3))
        x = self._conv3d_block(x, 64, (3, 1, 1))
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = self._conv3d_block(x, 128, (1, 3, 3))
        x = self._conv3d_block(x, 128, (3, 1, 1))

        return x

    def gate_flow_slow_fast_network_builder(self):
        """Build the complete Gate_Flow_SlowFast network."""
        clip_shape = (64, 224, 224, 3)
        tau = 16

        clip_input = layers.Input(shape=clip_shape)

        # Slow input: subsample to 4 frames
        slow_input = TemporalSubsample(
            max_frames=64, stride=tau, name='slow_input'
        )(clip_input)

        # Fast input: all 64 frames
        fast_input = clip_input

        # Build pathways
        fast_rgb, connection = self.get_Flow_gate_fast_path(fast_input)
        slow_rgb = self.get_Flow_gate_slow_path(slow_input, connection)

        # Merge: fast merging block + slow path
        merging_block_fast_res = self.merging_block(fast_rgb)
        x = layers.Add(name="ADD_slow_rgb_ans_fast_rgb_opt")(
            [merging_block_fast_res, slow_rgb]
        )

        # Fully connected classification head
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        pred = layers.Dense(3, activation='softmax')(x)

        model = Model(inputs=clip_input, outputs=pred)
        return model

    def load_model_and_weight(self):
        """Build the model and load pre-trained weights."""
        model = self.gate_flow_slow_fast_network_builder()
        model.load_weights(self.weights_path)
        print(f"[+] Model loaded with weights from: {self.weights_path}")
        return model
