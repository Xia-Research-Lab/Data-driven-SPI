#!/usr/bin/env python3
"""
Integrated Data Preparation Script for Cyto128 Dataset

This script combines the functionality of:
1. cyto_download.py - Downloads the CytoImageNet dataset from Kaggle
2. create_small_dataset.py - Processes and splits the dataset

Features:
- Automatic dataset download from Kaggle
- Hash-based splitting to prevent data leakage
- Resize images to 128x128 and convert to grayscale
- Organize into train/test/valid directories with consistent naming

Usage:
    python data_prep.py [--download] [--process] [--output_dir cyto128]

    # Full pipeline (download + process)
    python data_prep.py

    # Download only
    python data_prep.py --download

    # Process existing dataset
    python data_prep.py --process

    # Custom output directory
    python data_prep.py --output_dir /path/to/cyto128
"""

import os
import sys
import argparse
import hashlib
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class CytoDatasetPrep:
    """Handle Cyto128 dataset download and preparation"""

    def __init__(self, output_dir="cyto128", kaggle_cache=None):
        """
        Initialize dataset preparation

        Args:
            output_dir: Output directory for processed dataset (default: cyto128)
            kaggle_cache: Kaggle cache directory (auto-detected if None)
        """
        self.output_dir = output_dir
        self.kaggle_cache = kaggle_cache or os.path.expanduser(
            "~/.cache/kagglehub/datasets/stanleyhua/cytoimagenet/versions/8"
        )

        # Configuration
        self.train_size = 10000
        self.test_size = 1000
        self.valid_size = 100
        self.target_size = (128, 128)
        self.random_seed = 42

    def download_dataset(self):
        """Download dataset from Kaggle"""
        print("=" * 60)
        print("🔽 Downloading CytoImageNet Dataset from Kaggle")
        print("=" * 60)

        try:
            import kagglehub
        except ImportError:
            print(
                "❌ Error: kagglehub not installed. "
                "Please install it with: pip install kagglehub"
            )
            return False

        try:
            print("Downloading dataset (this may take a while)...")
            path = kagglehub.dataset_download("stanleyhua/cytoimagenet")
            print(f"✅ Dataset downloaded successfully!")
            print(f"   Path: {path}")
            self.kaggle_cache = path
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False

    def create_output_dirs(self):
        """Create output directories"""
        for subdir in ["train", "test", "valid"]:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        print(f"✅ Output directories created: {self.output_dir}/")
        print(f"   - {self.output_dir}/train/")
        print(f"   - {self.output_dir}/test/")
        print(f"   - {self.output_dir}/valid/")

    def collect_all_images(self):
        """Collect all image paths from dataset"""
        if not os.path.exists(self.kaggle_cache):
            print(f"❌ Error: Dataset path does not exist: {self.kaggle_cache}")
            print("Please run with --download flag first")
            return None

        image_paths = []

        # Traverse all subdirectories (each is a category)
        for category_dir in os.listdir(self.kaggle_cache):
            category_path = os.path.join(self.kaggle_cache, category_dir)

            if not os.path.isdir(category_path):
                continue

            # Collect all images in this category
            for filename in os.listdir(category_path):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(category_path, filename))

        print(f"✅ Found {len(image_paths)} images in total")
        return image_paths

    def resize_and_save_image(self, src_path, dst_path):
        """Resize image and save as grayscale PNG"""
        try:
            with Image.open(src_path) as img:
                # Convert to grayscale (L mode)
                if img.mode != "L":
                    img = img.convert("L")

                # Resize image using high-quality resampling
                img_resized = img.resize(self.target_size, Image.LANCZOS)

                # Save as PNG
                img_resized.save(dst_path, "PNG", quality=95)
                return True
        except Exception as e:
            print(f"⚠️  Failed to process {src_path}: {e}")
            return False

    def split_dataset_by_hash(self, image_paths):
        """
        Hash-based dataset splitting to ensure:
        1. Complete reproducibility (using hash instead of random)
        2. No data leakage (same file won't appear in multiple sets)
        3. Train, test, and validation sets are completely independent

        Principle:
        - MD5 hash each file path
        - Map hash values to [0, 10000) range
        - Assign to different sets based on proportion thresholds
        - This ensures the same file is always assigned to the same set with no leakage
        """
        train_set = []
        test_set = []
        valid_set = []

        total_required = self.train_size + self.test_size + self.valid_size
        train_threshold = (self.train_size / total_required) * 10000
        test_threshold = ((self.train_size + self.test_size) / total_required) * 10000

        print(f"\n📊 Using hash-based splitting to ensure no data leakage...")
        print(f"   Split thresholds:")
        print(f"     - Train: [0, {train_threshold:.0f})")
        print(f"     - Test:  [{train_threshold:.0f}, {test_threshold:.0f})")
        print(f"     - Valid: [{test_threshold:.0f}, 10000)")

        for img_path in image_paths:
            # Use file path hash value to determine set assignment
            hash_value = int(hashlib.md5(img_path.encode()).hexdigest(), 16) % 10000

            if hash_value < train_threshold:
                train_set.append(img_path)
            elif hash_value < test_threshold:
                test_set.append(img_path)
            else:
                valid_set.append(img_path)

        print(
            f"\n   Initial split: "
            f"train={len(train_set)}, test={len(test_set)}, valid={len(valid_set)}"
        )

        # Check if quantities are sufficient
        if (
            len(train_set) < self.train_size
            or len(test_set) < self.test_size
            or len(valid_set) < self.valid_size
        ):
            print(
                f"⚠️  Warning: Insufficient quantities after hash split, "
                f"using random fallback..."
            )
            print(
                f"   Required: "
                f"train={self.train_size}, test={self.test_size}, "
                f"valid={self.valid_size}"
            )

            # Fallback: use random split with fixed seed for reproducibility
            random.seed(self.random_seed)
            random.shuffle(image_paths)
            train_set = image_paths[: self.train_size]
            test_set = image_paths[
                self.train_size : self.train_size + self.test_size
            ]
            valid_set = image_paths[
                self.train_size + self.test_size : self.train_size
                + self.test_size
                + self.valid_size
            ]
            print(
                f"   After adjustment: "
                f"train={len(train_set)}, test={len(test_set)}, "
                f"valid={len(valid_set)}"
            )
        else:
            # If enough, truncate to required quantity
            train_set = train_set[: self.train_size]
            test_set = test_set[: self.test_size]
            valid_set = valid_set[: self.valid_size]
            print(
                f"   Final: "
                f"train={len(train_set)}, test={len(test_set)}, "
                f"valid={len(valid_set)}"
            )

        print(f"✅ Dataset split completed (no data leakage guaranteed)\n")

        return train_set, test_set, valid_set

    def process_datasets(self, image_paths):
        """Process and split images into train/test/valid sets"""
        print("=" * 60)
        print("⚙️  Processing Dataset")
        print("=" * 60)

        # Split dataset using hash-based approach
        train_images, test_images, valid_images = self.split_dataset_by_hash(
            image_paths
        )

        # Process training set
        print("Processing training set...")
        train_dir = os.path.join(self.output_dir, "train")
        success_count = 0
        for idx, src_path in enumerate(
            tqdm(train_images, desc="Train", unit="img")
        ):
            dst_filename = f"train_{idx:06d}.png"
            dst_path = os.path.join(train_dir, dst_filename)
            if self.resize_and_save_image(src_path, dst_path):
                success_count += 1
        print(f"✅ Training set: {success_count}/{len(train_images)} processed")

        # Process test set
        print("\nProcessing test set...")
        test_dir = os.path.join(self.output_dir, "test")
        success_count = 0
        for idx, src_path in enumerate(tqdm(test_images, desc="Test", unit="img")):
            dst_filename = f"test_{idx:06d}.png"
            dst_path = os.path.join(test_dir, dst_filename)
            if self.resize_and_save_image(src_path, dst_path):
                success_count += 1
        print(f"✅ Test set: {success_count}/{len(test_images)} processed")

        # Process validation set
        print("\nProcessing validation set...")
        valid_dir = os.path.join(self.output_dir, "valid")
        success_count = 0
        for idx, src_path in enumerate(
            tqdm(valid_images, desc="Valid", unit="img")
        ):
            dst_filename = f"valid_{idx:06d}.png"
            dst_path = os.path.join(valid_dir, dst_filename)
            if self.resize_and_save_image(src_path, dst_path):
                success_count += 1
        print(f"✅ Validation set: {success_count}/{len(valid_images)} processed")

    def print_summary(self):
        """Print dataset preparation summary"""
        print("\n" + "=" * 60)
        print("✅ Dataset Preparation Complete!")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}/")

        # Count images in each directory
        for subset in ["train", "test", "valid"]:
            subset_dir = os.path.join(self.output_dir, subset)
            if os.path.exists(subset_dir):
                count = len(
                    [f for f in os.listdir(subset_dir) if f.endswith(".png")]
                )
                print(f"  - {subset}: {count} images")

        print(f"\nImage specifications:")
        print(f"  - Format: Grayscale PNG (L mode)")
        print(f"  - Resolution: {self.target_size[0]}×{self.target_size[1]} pixels")
        print(f"  - Naming: {{subset}}_{{000000...}}.png (e.g., train_000000.png)")

        print(f"\nData integrity:")
        print(f"  ✓ No data leakage: Hash-based splitting ensures each image")
        print(f"    appears in only one set (train/test/valid)")
        print(f"  ✓ Reproducibility: Same results with same seed ({self.random_seed})")

    def run(self, download=True, process=True):
        """Run the complete pipeline"""
        print("\n" + "=" * 60)
        print("🚀 Cyto128 Dataset Preparation Tool")
        print("=" * 60)
        print(f"Target: {self.train_size} train + {self.test_size} test + "
              f"{self.valid_size} valid = "
              f"{self.train_size + self.test_size + self.valid_size} total\n")

        # Step 1: Download (optional)
        if download:
            if not self.download_dataset():
                print("❌ Dataset download failed. Aborting...")
                return False

        # Step 2: Create output directories
        if process:
            self.create_output_dirs()

            # Step 3: Collect images
            image_paths = self.collect_all_images()
            if not image_paths:
                print("❌ No images found. Aborting...")
                return False

            # Step 4: Process and split
            self.process_datasets(image_paths)

            # Step 5: Print summary
            self.print_summary()

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Integrated Cyto128 Dataset Preparation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (download + process)
  python data_prep.py

  # Download only
  python data_prep.py --download-only

  # Process existing dataset
  python data_prep.py --process-only

  # Custom output directory
  python data_prep.py --output_dir /path/to/cyto128

  # Skip download, just process
  python data_prep.py --skip-download
        """,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="cyto128",
        help="Output directory for processed dataset (default: cyto128)",
    )

    parser.add_argument(
        "--kaggle_cache",
        type=str,
        default=None,
        help="Path to Kaggle cache directory (auto-detected if not provided)",
    )

    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download dataset, skip processing",
    )

    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Only process existing dataset, skip download",
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (assumes dataset already exists)",
    )

    parser.add_argument(
        "--train_size",
        type=int,
        default=10000,
        help="Number of training samples (default: 10000)",
    )

    parser.add_argument(
        "--test_size",
        type=int,
        default=1000,
        help="Number of test samples (default: 1000)",
    )

    parser.add_argument(
        "--valid_size",
        type=int,
        default=100,
        help="Number of validation samples (default: 100)",
    )

    args = parser.parse_args()

    # Initialize dataset preparation
    prep = CytoDatasetPrep(output_dir=args.output_dir, kaggle_cache=args.kaggle_cache)

    # Override dataset sizes if specified
    prep.train_size = args.train_size
    prep.test_size = args.test_size
    prep.valid_size = args.valid_size

    # Determine pipeline steps
    download = not args.process_only and not args.skip_download
    process = not args.download_only

    # Run pipeline
    success = prep.run(download=download, process=process)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
