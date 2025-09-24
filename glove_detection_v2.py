#!/usr/bin/env python3
"""
IMPROVED Gloved vs Bare Hand Detection Script
============================================
This version includes fixes for common detection issues:
- Better preprocessing for difficult lighting
- Improved NMS settings
- Lower confidence thresholds with post-processing
- Enhanced hand region classification

Version 2.0 - Fixed detection issues
"""

import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime

class ImprovedGloveDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.3, iou_threshold=0.4):
        """
        Initialize the improved glove detector

        Args:
            model_path: Path to YOLOv8 model file
            confidence: Lower confidence threshold to catch more hands
            iou_threshold: IoU threshold for Non-Maximum Suppression
        """
        print(f"🔧 Loading YOLOv8 model from: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold

        # Configure model settings for better detection
        self.model.overrides['conf'] = confidence
        self.model.overrides['iou'] = iou_threshold
        self.model.overrides['agnostic_nms'] = True  # Better NMS
        self.model.overrides['max_det'] = 10  # Limit detections per image

        print(f"✅ Model loaded with improved settings!")
        print(f"📊 Confidence threshold: {confidence} (lowered for better detection)")
        print(f"📊 IoU threshold: {iou_threshold} (optimized for NMS)")

    def preprocess_image(self, image):
        """
        Preprocess image for better detection
        """
        # Convert to RGB for better processing
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Apply CLAHE for better contrast
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)

        return enhanced

    def detect_hands_in_image(self, image_path):
        """
        Improved hand detection with better preprocessing and classification
        """
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not read image: {image_path}")
            return [], None

        original_image = image.copy()

        # Preprocess for better detection
        enhanced_image = self.preprocess_image(image)

        # Run YOLO detection on both original and enhanced images
        results_original = self.model(image, conf=self.confidence, iou=self.iou_threshold)
        results_enhanced = self.model(enhanced_image, conf=self.confidence, iou=self.iou_threshold)

        # Combine results from both images
        all_detections = []

        # Process original image results
        for result in results_original:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()

                    # Classify hand type
                    label = self._improved_hand_classification(image, x1, y1, x2, y2)

                    all_detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(confidence),
                        "label": label,
                        "source": "original"
                    })

        # Process enhanced image results
        for result in results_enhanced:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()

                    # Classify hand type
                    label = self._improved_hand_classification(enhanced_image, x1, y1, x2, y2)

                    all_detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(confidence),
                        "label": label,
                        "source": "enhanced"
                    })

        # Apply custom NMS to remove duplicates
        final_detections = self._apply_custom_nms(all_detections)

        # Draw bounding boxes
        annotated_image = self._draw_detections(original_image, final_detections)

        return final_detections, annotated_image

    def _improved_hand_classification(self, image, x1, y1, x2, y2):
        """
        Improved hand classification using multiple features
        """
        # Extract hand region with padding
        padding = 10
        h, w = image.shape[:2]
        x1_pad = max(0, int(x1) - padding)
        y1_pad = max(0, int(y1) - padding)
        x2_pad = min(w, int(x2) + padding)
        y2_pad = min(h, int(y2) + padding)

        hand_region = image[y1_pad:y2_pad, x1_pad:x2_pad]

        if hand_region.size == 0:
            return "bare_hand"  # Default fallback

        # Multiple classification features
        features = []

        # 1. Color analysis (improved)
        hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)

        # Check for white/light colors (gloves)
        white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 30, 255))
        white_ratio = np.sum(white_mask > 0) / hand_region.size
        features.append(white_ratio)

        # Check for blue colors (common glove color)
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        blue_ratio = np.sum(blue_mask > 0) / hand_region.size
        features.append(blue_ratio)

        # 2. Texture analysis
        gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)

        # Calculate texture variance (gloves often have more uniform texture)
        texture_var = np.var(gray)
        features.append(texture_var)

        # 3. Edge density (bare skin typically has more natural edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)

        # 4. Brightness analysis
        brightness = np.mean(gray)
        features.append(brightness)

        # Decision logic (improved rules)
        glove_score = 0

        # White/light color indicates gloves
        if white_ratio > 0.3:
            glove_score += 3

        # Blue color indicates gloves  
        if blue_ratio > 0.1:
            glove_score += 2

        # Low texture variance indicates gloves (smoother surface)
        if texture_var < 200:
            glove_score += 1

        # Very high brightness often indicates white gloves
        if brightness > 180:
            glove_score += 2

        # Lower edge density might indicate gloves
        if edge_density < 0.1:
            glove_score += 1

        return "gloved_hand" if glove_score >= 3 else "bare_hand"

    def _apply_custom_nms(self, detections, overlap_threshold=0.3):
        """
        Apply custom Non-Maximum Suppression to remove duplicate detections
        """
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        final_detections = []

        while detections:
            # Take the highest confidence detection
            current = detections.pop(0)
            final_detections.append(current)

            # Remove overlapping detections
            remaining = []
            for det in detections:
                if self._calculate_iou(current['bbox'], det['bbox']) < overlap_threshold:
                    remaining.append(det)
            detections = remaining

        return final_detections

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Calculate union
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image"""
        annotated = image.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']

            # Choose color based on label
            color = (0, 255, 0) if label == "gloved_hand" else (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Add label with confidence
            label_text = f"{label}: {confidence:.2f}"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Draw background rectangle for text
            cv2.rectangle(annotated, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0], y1), color, -1)

            # Draw text
            cv2.putText(annotated, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated

    def process_folder(self, input_folder, output_folder, logs_folder):
        """Process all images in a folder with improved detection"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        logs_path = Path(logs_folder)

        # Create directories
        output_path.mkdir(parents=True, exist_ok=True)
        logs_path.mkdir(parents=True, exist_ok=True)

        # Find images
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))

        if not image_files:
            print(f"❌ No JPG images found in {input_folder}")
            return

        print(f"🔍 Found {len(image_files)} images to process")

        successful = 0
        failed = 0

        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                # Process image
                detections, annotated_image = self.detect_hands_in_image(image_file)

                # Save annotated image
                if annotated_image is not None:
                    output_image_path = output_path / image_file.name
                    cv2.imwrite(str(output_image_path), annotated_image)

                # Convert detections to required format
                formatted_detections = []
                for det in detections:
                    formatted_detections.append({
                        "label": det["label"],
                        "confidence": det["confidence"], 
                        "bbox": det["bbox"]
                    })

                # Save log file
                log_data = {
                    "filename": image_file.name,
                    "detections": formatted_detections,
                    "processed_at": datetime.now().isoformat(),
                    "model_confidence_threshold": self.confidence,
                    "improvements_applied": [
                        "Enhanced preprocessing with CLAHE",
                        "Dual image processing (original + enhanced)",
                        "Improved hand classification with multiple features",
                        "Custom NMS for better duplicate removal",
                        "Lower confidence threshold with post-processing"
                    ]
                }

                log_file_path = logs_path / f"{image_file.stem}.json"
                with open(log_file_path, 'w') as f:
                    json.dump(log_data, f, indent=2)

                successful += 1
                print(f"✅ {image_file.name}: {len(formatted_detections)} detections")

            except Exception as e:
                failed += 1
                print(f"❌ Error processing {image_file.name}: {str(e)}")

        print(f"\n🎉 Processing complete!")
        print(f"✅ Successful: {successful} images")
        print(f"❌ Failed: {failed} images")

def main():
    parser = argparse.ArgumentParser(description="Improved Glove Detection System")
    parser.add_argument("--input", type=str, default="input_images/", 
                       help="Input folder containing JPG images")
    parser.add_argument("--output", type=str, default="output/", 
                       help="Output folder for annotated images")
    parser.add_argument("--logs", type=str, default="logs/", 
                       help="Folder for JSON detection logs")
    parser.add_argument("--confidence", type=float, default=0.3, 
                       help="Confidence threshold (lowered for better detection)")
    parser.add_argument("--iou", type=float, default=0.4, 
                       help="IoU threshold for NMS")
    parser.add_argument("--model", type=str, default="yolov8n.pt", 
                       help="Path to YOLOv8 model file")

    args = parser.parse_args()

    print("=" * 60)
    print("🧤 IMPROVED GLOVE DETECTION SYSTEM v2.0")
    print("=" * 60)
    print(f"📁 Input folder: {args.input}")
    print(f"📁 Output folder: {args.output}")
    print(f"📁 Logs folder: {args.logs}")
    print(f"📊 Confidence threshold: {args.confidence} (lowered)")
    print(f"📊 IoU threshold: {args.iou} (optimized)")
    print(f"🤖 Model: {args.model}")
    print()
    print("🔧 IMPROVEMENTS IN v2.0:")
    print("- Enhanced image preprocessing with CLAHE")
    print("- Dual processing (original + enhanced images)")
    print("- Better hand classification with multiple features")
    print("- Custom NMS to fix overlapping boxes")
    print("- Lower confidence threshold with smart post-processing")
    print()

    # Initialize improved detector
    detector = ImprovedGloveDetector(
        model_path=args.model, 
        confidence=args.confidence,
        iou_threshold=args.iou
    )

    # Process images
    detector.process_folder(args.input, args.output, args.logs)

if __name__ == "__main__":
    main()
