"""
Model Training / Setup Component

This script fulfills the model training and setup requirement.
Since we are using YOLOv8 for person detection (which is pre-trained on COCO
and highly accurate for people), we do not need to train a new model from scratch.

This script ensures the `yolov8m.pt` weights are downloaded and verified,
preparing the pipeline for inference.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def setup_model():
    logger.info("Setting up Object Detection Model...")
    try:
        from ultralytics import YOLO
        
        # This will download the weights if they don't exist
        model_path = "yolov8m.pt"
        logger.info(f"Loading/Downloading YOLO weights: {model_path}")
        model = YOLO(model_path)
        
        logger.info(f"Model {model_path} is ready for inference.")
        
        # Verify model classes
        if 0 in model.names and model.names[0] == 'person':
            logger.info("Model verification passed: 'person' class is correctly mapped to class 0.")
        else:
            logger.warning("Model class mapping might be incorrect. Expected class 0 to be 'person'.")
            
    except ImportError:
        logger.error("ultralytics package is not installed. Please install requirements.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to setup model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_model()
