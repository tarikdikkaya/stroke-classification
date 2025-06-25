"""
Dataset splitting utility for medical image classification.
Splits the main dataset into train/test sets while maintaining class balance.
"""
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def split_dataset(source_dir, output_dir, train_ratio=0.9, random_seed=42):
    """
    Split dataset into train/test sets.
    
    Args:
        source_dir: Path to the main dataset directory
        output_dir: Path where train/test folders will be created
        train_ratio: Proportion of data for training (0.8 = 80% train, 20% test)
        random_seed: Random seed for reproducible splits
    """
    random.seed(random_seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    
    print(f"📁 Creating directories in {output_path}")
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each class folder
    class_stats = defaultdict(dict)
    
    for class_folder in source_path.iterdir():
        if not class_folder.is_dir():
            continue
            
        class_name = class_folder.name
        print(f"\n🔄 Processing class: {class_name}")
        
        # Create class folders in train/test
        (train_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)
        
        # Get all files in class folder
        files = [f for f in class_folder.iterdir() 
                if f.is_file() and f.suffix.lower() in ['.dcm', '.jpg', '.png']]
        
        if not files:
            print(f"⚠️  No supported files found in {class_folder}")
            continue
        
        # Shuffle files
        random.shuffle(files)
        
        # Split files
        num_train = int(len(files) * train_ratio)
        train_files = files[:num_train]
        test_files = files[num_train:]
        
        # Copy files to train folder
        print(f"  📤 Copying {len(train_files)} files to train/{class_name}")
        for file_path in train_files:
            shutil.copy2(file_path, train_dir / class_name / file_path.name)
        
        # Copy files to test folder
        print(f"  📤 Copying {len(test_files)} files to test/{class_name}")
        for file_path in test_files:
            shutil.copy2(file_path, test_dir / class_name / file_path.name)
        
        # Store statistics
        class_stats[class_name] = {
            'total': len(files),
            'train': len(train_files),
            'test': len(test_files)
        }
    
    # Print summary
    print("\n" + "="*50)
    print("📊 Dataset Split Summary:")
    print("="*50)
    
    total_files = 0
    total_train = 0
    total_test = 0
    
    for class_name, stats in class_stats.items():
        print(f"\n{class_name}:")
        print(f"  Total: {stats['total']:,} files")
        print(f"  Train: {stats['train']:,} files ({stats['train']/stats['total']*100:.1f}%)")
        print(f"  Test:  {stats['test']:,} files ({stats['test']/stats['total']*100:.1f}%)")
        
        total_files += stats['total']
        total_train += stats['train']
        total_test += stats['test']
    
    print(f"\n📈 Overall:")
    print(f"  Total: {total_files:,} files")
    print(f"  Train: {total_train:,} files ({total_train/total_files*100:.1f}%)")
    print(f"  Test:  {total_test:,} files ({total_test/total_files*100:.1f}%)")
    
    print(f"\n✅ Dataset split complete!")
    print(f"📁 Train data: {train_dir}")
    print(f"📁 Test data:  {test_dir}")

def main():
    """Main function to run dataset splitting."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Split dataset into train/test sets')
    parser.add_argument('--source_dir', type=str, 
                       default=r'a:\Teknofest SYZ\AŞAMA2\main_dataset',
                       help='Source dataset directory')
    parser.add_argument('--output_dir', type=str, 
                       default=r'a:\Teknofest SYZ\AŞAMA2\split_dataset',
                       help='Output directory for train/test splits')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                       help='Ratio of data for training (default: 0.8)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    
    args = parser.parse_args()
    
    print("🚀 Medical Dataset Splitter")
    print("="*50)
    print(f"Source:      {args.source_dir}")
    print(f"Output:      {args.output_dir}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Random seed: {args.random_seed}")
    
    if not os.path.exists(args.source_dir):
        print(f"❌ Error: Source directory does not exist: {args.source_dir}")
        return
    
    split_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    main()
