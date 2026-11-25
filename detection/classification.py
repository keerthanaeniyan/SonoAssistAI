"""
SonoAssist AI - Classification Inference Engine
Wraps EfficientNet-B0 to classify ovarian ultrasound scans into MMOTU dataset categories.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from models.efficientnet import OvarianClassifier, MMOTU_CATEGORIES, predict_category

DEFAULT_CLASSIFIER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "trained_weights", "efficientnet_best.pth"
)


class ClassificationEngine:
    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path or DEFAULT_CLASSIFIER_PATH
        self.is_trained = False
        self.model_mode = "UNTRAINED_DEMO"
        self.model = OvarianClassifier(num_classes=len(MMOTU_CATEGORIES), in_channels=1).to(self.device)

        self._load_weights()

    def _load_weights(self) -> None:
        """Attempt to load trained classifier checkpoint."""
        if os.path.exists(self.weights_path):
            try:
                state_dict = torch.load(self.weights_path, map_location=self.device)
                if "model_state_dict" in state_dict:
                    self.model.load_state_dict(state_dict["model_state_dict"])
                else:
                    self.model.load_state_dict(state_dict)
                self.model.eval()
                self.is_trained = True
                self.model_mode = "TRAINED_MODEL"
                print(f"[ClassificationEngine] Loaded trained classifier from: {self.weights_path}")
            except Exception as e:
                print(f"[ClassificationEngine] Could not load weights ({e}). Running in Demonstration Mode.")
                self.is_trained = False
                self.model_mode = "DEMO_UNTRAINED"
        else:
            self.is_trained = False
            self.model_mode = "DEMO_UNTRAINED"

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Resize to 256x256 and convert to float tensor [1, 1, 256, 256]."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        return torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)

    def predict(self, image: np.ndarray, morphological_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classify ultrasound image into MMOTU category.
        """
        tensor = self.preprocess_image(image)

        if self.is_trained:
            result = predict_category(self.model, tensor, self.device)
            result["is_trained"] = True
            result["model_mode"] = self.model_mode
            return result

        # DEMO / UNTRAINED MODE:
        # Provide heuristic correlation with morphological measurements if available,
        # otherwise provide balanced MMOTU category distribution clearly labeled as DEMO.
        predicted_class = "Simple cyst"
        confidence = 0.72

        if morphological_context and morphological_context.get("has_detection"):
            primary = morphological_context.get("primary_cyst", {})
            circ = primary.get("circularity", 0.8)
            area = primary.get("area_pixels", 2000)
            roughness = primary.get("roughness_index", 1.0)

            if circ > 0.85 and roughness < 1.1:
                predicted_class = "Simple cyst"
                confidence = 0.82
            elif circ > 0.7 and area > 5000:
                predicted_class = "Mucinous cystadenoma"
                confidence = 0.74
            elif roughness > 1.25:
                predicted_class = "Teratoma"
                confidence = 0.69
            else:
                predicted_class = "Chocolate cyst"
                confidence = 0.71
        elif morphological_context and not morphological_context.get("has_detection"):
            predicted_class = "Normal ovary"
            confidence = 0.86

        # Generate realistic softmax probabilities centered around top prediction
        all_probs = {}
        remaining_prob = 1.0 - confidence
        other_cats = [c for c in MMOTU_CATEGORIES if c != predicted_class]
        sub_probs = np.random.dirichlet(np.ones(len(other_cats))) * remaining_prob

        all_probs[predicted_class] = round(confidence, 4)
        for cat, prob in zip(other_cats, sub_probs):
            all_probs[cat] = round(float(prob), 4)

        return {
            "predicted_category": predicted_class,
            "confidence": round(confidence, 4),
            "class_probabilities": all_probs,
            "is_trained": False,
            "model_mode": "DEMO_UNTRAINED",
            "clinical_disclaimer": "DEMO MODE: Statistical prediction shown for software testing. This is NOT a diagnostic prediction and must be confirmed by a radiologist."
        }
