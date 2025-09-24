#!/usr/bin/env python3
"""
Glove Detection Training Script
===============================
This script trains YOLOv8 on the glove detection dataset to create a custom model
that specifically detects gloved hands and bare hands.

Usage:
    python train_glove_model.py --data dataset/data.yaml --epochs 100

Requirements:
    - Glove dataset in YOLO format (downloaded from Roboflow)
    - YOLOv8 installed (ultralytics package)
"""

import os
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

class GloveModelTrainer:
    def __init__(self, data_yaml_path, model_size="n", project_name="glove_detection"):
        """
        Initialize the glove detection trainer

        Args:
            data_yaml_path: Path to the dataset YAML file
            model_size: Size of YOLOv8 model (n, s, m, l, x)
            project_name: Name for the training project
        """
        self.data_yaml_path = data_yaml_path
        self.model_size = model_size
        self.project_name = project_name

        # Initialize the model
        model_path = f"yolov8{model_size}.pt"
        print(f"🔧 Loading YOLOv8{model_size.upper()} model: {model_path}")
        self.model = YOLO(model_path)

        # Verify dataset
        self._verify_dataset()

    def _verify_dataset(self):
        """Verify the dataset structure and classes"""
        print("🔍 Verifying dataset structure...")

        if not os.path.exists(self.data_yaml_path):
            raise FileNotFoundError(f"Dataset YAML not found: {self.data_yaml_path}")

        # Read the data.yaml file
        with open(self.data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        print(f"✅ Dataset YAML loaded successfully")
        print(f"📊 Number of classes: {data_config.get('nc', 'Unknown')}")
        print(f"📋 Class names: {data_config.get('names', 'Unknown')}")

        # Verify paths exist
        train_path = data_config.get('train', '')
        val_path = data_config.get('val', '')

        if train_path and not os.path.exists(train_path):
            print(f"⚠️  Train path not found: {train_path}")
        else:
            print(f"✅ Train path verified: {train_path}")

        if val_path and not os.path.exists(val_path):
            print(f"⚠️  Validation path not found: {val_path}")
        else:
            print(f"✅ Validation path verified: {val_path}")

    def train(self, epochs=100, imgsz=640, batch=16, patience=10, device='auto'):
        """
        Train the model on glove detection dataset

        Args:
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            patience: Early stopping patience
            device: Training device ('auto', 'cpu', or 'cuda')
        """
        print("=" * 60)
        print("🏋️  STARTING GLOVE DETECTION TRAINING")
        print("=" * 60)

        print(f"📊 Training parameters:")
        print(f"   - Epochs: {epochs}")
        print(f"   - Image size: {imgsz}")
        print(f"   - Batch size: {batch}")
        print(f"   - Patience: {patience}")
        print(f"   - Device: {device}")
        print(f"   - Model: YOLOv8{self.model_size.upper()}")
        print()

        # Start training
        try:
            results = self.model.train(
                data=self.data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                patience=patience,
                device=device,
                project='runs/detect',
                name=self.project_name,
                save=True,
                save_period=10,  # Save checkpoint every 10 epochs
                cache=True,      # Cache images for faster training
                plots=True       # Generate training plots
            )

            print("🎉 Training completed successfully!")

            # Get the best weights path
            best_weights = f"runs/detect/{self.project_name}/weights/best.pt"
            print(f"✅ Best weights saved to: {best_weights}")

            return results, best_weights

        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            return None, None

    def validate(self, weights_path):
        """Validate the trained model"""
        print("\n🔍 Validating trained model...")

        try:
            # Load the trained model
            trained_model = YOLO(weights_path)

            # Run validation
            results = trained_model.val(
                data=self.data_yaml_path,
                imgsz=640,
                batch=16,
                device='auto',
                plots=True
            )

            print("✅ Validation completed!")

            # Print key metrics
            if hasattr(results, 'box'):
                print(f"📊 Validation Results:")
                print(f"   - mAP@0.5: {results.box.map50:.4f}")
                print(f"   - mAP@0.5:0.95: {results.box.map:.4f}")
                print(f"   - Precision: {results.box.mp:.4f}")
                print(f"   - Recall: {results.box.mr:.4f}")

            return results

        except Exception as e:
            print(f"❌ Validation failed: {str(e)}")
            return None

    def test_inference(self, weights_path, test_image_path):
        """Test the trained model on a sample image"""
        print(f"\n🧪 Testing inference on: {test_image_path}")

        try:
            # Load trained model
            trained_model = YOLO(weights_path)

            # Run inference
            results = trained_model(test_image_path, conf=0.3)

            # Display results
            for r in results:
                if r.boxes is not None:
                    print(f"✅ Found {len(r.boxes)} detections")
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        confidence = box.conf[0]
                        print(f"   - Class {class_id}: {confidence:.3f}")
                else:
                    print("❌ No detections found")

            # Save annotated result
            annotated = results[0].plot()
            output_path = f"test_inference_result.jpg"

            import cv2
            cv2.imwrite(output_path, annotated)
            print(f"💾 Annotated result saved: {output_path}")

            return results

        except Exception as e:
            print(f"❌ Inference test failed: {str(e)}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Glove Detection")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to dataset YAML file (e.g., dataset/data.yaml)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size for training")  
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size for training")
    parser.add_argument("--model", type=str, default="n", 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help="YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Training device (auto, cpu, cuda)")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--project", type=str, default="glove_detection",
                       help="Project name for saving results")
    parser.add_argument("--test-image", type=str, default=None,
                       help="Path to test image for inference testing")

    args = parser.parse_args()

    print("=" * 60)
    print("🧤 GLOVE DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"📁 Dataset: {args.data}")
    print(f"🏃 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch}")
    print(f"📏 Image size: {args.imgsz}")
    print(f"🤖 Model size: YOLOv8{args.model.upper()}")
    print(f"💻 Device: {args.device}")
    print(f"⏰ Patience: {args.patience}")
    print()

    # Initialize trainer
    trainer = GloveModelTrainer(
        data_yaml_path=args.data,
        model_size=args.model,
        project_name=args.project
    )

    # Train the model
    results, best_weights = trainer.train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device
    )

    if best_weights:
        # Validate the model
        trainer.validate(best_weights)

        # Test inference if test image provided
        if args.test_image and os.path.exists(args.test_image):
            trainer.test_inference(best_weights, args.test_image)

        print("\n" + "=" * 60)
        print("🎉 TRAINING PIPELINE COMPLETED!")
        print("=" * 60)
        print(f"✅ Trained model weights: {best_weights}")
        print(f"📊 Training results saved in: runs/detect/{args.project}/")
        print(f"\n🔧 NEXT STEPS:")
        print(f"1. Use the trained weights in your detection script:")
        print(f"   python glove_detection_v2.py --model {best_weights}")
        print(f"2. Test on your input images")
        print(f"3. Adjust confidence threshold if needed")

    else:
        print("❌ Training failed. Please check the dataset path and try again.")

if __name__ == "__main__":
    main()
