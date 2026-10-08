from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import numpy as np
import torch
import tensorflow as tf
import clip
from PIL import Image


NSFW_MODEL_URL = (
    "https://raw.githubusercontent.com/LAION-AI/"
    "CLIP-based-NSFW-Detector/main/"
    "clip_autokeras_binary_nsfw.zip"
)

NSFW_MODEL_DIR = (
    Path.home()
    / ".cache"
    / "laion_nsfw"
    / "clip_autokeras_binary_nsfw"
)


class LAIONNSFWDetector:

    def __init__(self):
        # Device
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Download model if necessary
        self._download_model()

        # Load CLIP
        self.clip_model, self.preprocess = clip.load(
            "ViT-L/14",
            device=self.device,
        )

        self.clip_model.eval()

        # Load TensorFlow SavedModel directly
        model = tf.saved_model.load(
            str(NSFW_MODEL_DIR)
        )

        self.model = model.signatures[
            "serving_default"
        ]

    def _download_model(self):

        if NSFW_MODEL_DIR.exists():
            return

        NSFW_MODEL_DIR.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        zip_path = (
            NSFW_MODEL_DIR.parent
            / "model.zip"
        )

        print("Downloading LAION NSFW model...")

        urlretrieve(
            NSFW_MODEL_URL,
            zip_path,
        )

        print("Extracting...")

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as z:
            z.extractall(
                NSFW_MODEL_DIR.parent
            )

        zip_path.unlink()

    def score_image(self, image_path):

        # Load image
        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.preprocess(
            image
        ).unsqueeze(0).to(
            self.device
        )

        # CLIP embedding
        with torch.no_grad():

            embedding = (
                self.clip_model.encode_image(
                    image
                )
            )

        # Normalize
        embedding = (
            embedding
            / embedding.norm(
                dim=-1,
                keepdim=True,
            )
        )

        # LAION SavedModel expects float64
        embedding = embedding.cpu().numpy()
        embedding = embedding.astype(
            np.float64
        )

        # TensorFlow prediction
        result = self.model(
            tf.constant(
                embedding,
                dtype=tf.float64,
            )
        )

        # SavedModel returns a dictionary
        result = next(
            iter(result.values())
        )

        return float(
            result.numpy().squeeze()
        )


