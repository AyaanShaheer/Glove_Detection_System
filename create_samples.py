
import os
import json
import shutil
from pathlib import Path

def create_sample_files():
    """
    Creates sample files for GitHub demonstration
    """
    print("🏗️ Creating sample files for GitHub...")

    # Create directories with .gitkeep files
    dirs_to_create = [
        "output",
        "logs", 
        "runs/detect/glove_detection/weights"
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        gitkeep_file = Path(dir_path) / ".gitkeep"
        gitkeep_file.touch()
        print(f"✅ Created {dir_path}/.gitkeep")

    # Copy sample output images (if they exist)
    output_dir = Path("output")
    if output_dir.exists():
        jpg_files = list(output_dir.glob("*.jpg"))
        for i, jpg_file in enumerate(jpg_files[:5], 1):
            sample_name = f"sample_output_{i}.jpg"
            sample_path = output_dir / sample_name
            if jpg_file != sample_path:  # Don't copy to itself
                shutil.copy2(jpg_file, sample_path)
                print(f"✅ Created sample: {sample_name}")

    # Create sample JSON logs (if they exist)
    logs_dir = Path("logs")  
    if logs_dir.exists():
        json_files = list(logs_dir.glob("*.json"))
        for i, json_file in enumerate(json_files[:5], 1):
            sample_name = f"sample_log_{i}.json"
            sample_path = logs_dir / sample_name
            if json_file != sample_path:  # Don't copy to itself
                shutil.copy2(json_file, sample_path)
                print(f"✅ Created sample: {sample_name}")

    # Create sample detection log if none exist
    if not list(logs_dir.glob("sample_*.json")):
        sample_log = {
            "filename": "sample_image.jpg",
            "detections": [
                {
                    "label": "gloved_hand",
                    "confidence": 0.92,
                    "bbox": [145, 67, 289, 198]
                },
                {
                    "label": "bare_hand", 
                    "confidence": 0.85,
                    "bbox": [301, 45, 445, 176]
                }
            ],
            "processed_at": "2025-09-24T12:34:15.123456",
            "model_confidence_threshold": 0.3
        }

        with open(logs_dir / "sample_log_1.json", "w") as f:
            json.dump(sample_log, f, indent=2)
        print("✅ Created sample JSON log")

    print("\n🎉 Sample files created for GitHub demonstration!")

if __name__ == "__main__":
    create_sample_files()
