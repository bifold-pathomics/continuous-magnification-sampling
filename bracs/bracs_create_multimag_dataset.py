from pathlib import Path
from PIL import Image
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import shutil

# Increase limit for large BRACS images
Image.MAX_IMAGE_PIXELS = 500_000_000

class Downsampler:
    """Class containing downsampling methods images."""
    
    @staticmethod
    def gaussian_antialiased_downsample(img, source_size, target_size, method='cubic'):
        """
        Apply Gaussian anti-aliasing before downsampling. 
        """
        if source_size == target_size:
            return img
        
        img_array = np.array(img)
        assert len(img_array.shape) == 3
        
        # Calculate appropriate sigma for Gaussian filter
        scale_factor = source_size / target_size
        # Nyquist-Shannon theorem suggests roughly sigma = scale_factor/2 -> using slightly less blur 
        sigma = scale_factor / 2.5        
        for c in range(img_array.shape[2]):
            img_array[:,:,c] = ndimage.gaussian_filter(
                    img_array[:,:,c], 
                    sigma=sigma,
                    mode='reflect'  # Better edge handling
                )        
        img_filtered = Image.fromarray(img_array.astype('uint8'))
        
        if method == 'cubic':
            resample = Image.Resampling.BICUBIC
        elif method == 'bilinear':
            resample = Image.Resampling.BILINEAR
            
        return img_filtered.resize((target_size, target_size), resample)
    
    @staticmethod
    def progressive_downsample(img, source_size, target_size):
        """
        Downsample progressively in steps to better preserve features.
        No single downsampling step is larger than 2x.
        """
        if source_size == target_size:
            return img
        
        current_size = source_size
        current_img = img
        
        # Calculate intermediate sizes
        sizes = []
        temp_size = source_size
        while temp_size > target_size:
            temp_size = max(target_size, temp_size // 2)
            sizes.append(temp_size)
        
        # Progressive downsampling
        for next_size in sizes:
            current_img = Downsampler.gaussian_antialiased_downsample(
                current_img, current_size, next_size
            )
            current_size = next_size
        
        return current_img
    

def create_multimag_dataset_enhanced(
    source_path,
    output_path,
    target_mpp,
    source_mpp=0.25,
    patch_size=224,
    min_size_filter=None,
    downsampling_method='progressive',  
):
    """
    Enhanced version with better downsampling methods.
    
    Args:
        downsampling_method: One of:
            - 'gaussian': Gaussian anti-aliasing (with bicubic resizing or linear)
            - 'progressive': Progressive downsampling 
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    
    # Initialize downsampler
    downsampler = Downsampler()    
    downsample_functions = {
        'gaussian': downsampler.gaussian_antialiased_downsample,
        'progressive': downsampler.progressive_downsample,
    }
    downsample_fn = downsample_functions[downsampling_method]
    
    # Create output directory & copy metadata
    output_path.mkdir(parents=True, exist_ok=True)    
    xlsx_source = source_path.parent.parent / "BRACS.xlsx"
    if xlsx_source.exists():
        shutil.copy2(xlsx_source, output_path / "BRACS.xlsx")
    
    # Calculate scaling
    mag_ratio = target_mpp / source_mpp
    source_patch_size = int(patch_size * mag_ratio)
    
    print(f"\n{'='*70}")
    print(f"Enhanced Multi-Magnification BRACS Dataset Creation")
    print(f"{'='*70}")
    print(f"Downsampling method: {downsampling_method}")
    print(f"Source MPP: {source_mpp} (40x)")
    print(f"Target MPP: {target_mpp} ({40 / mag_ratio:.1f}x)")
    print(f"Source patch size: {source_patch_size}x{source_patch_size}")
    print(f"Output patch size: {patch_size}x{patch_size}")
    print(f"{'='*70}\n")
    
    stats = {
        'total_processed': 0,
        'total_skipped': 0,
        'total_filtered': 0,
        'by_class': {}
    }
    
    # Process splits
    for split in ['train', 'val', 'test']:
        split_source = source_path / split

        assert split_source.exists(), f"Source split directory does not exist: {split_source}"
        print(f"\nProcessing {split.upper()} split...")
        
        class_folders = sorted([d for d in split_source.iterdir() if d.is_dir()])
        
        for class_folder in class_folders:
            class_name = class_folder.name
            
            if class_name not in stats['by_class']:
                stats['by_class'][class_name] = {
                    'processed': 0,
                    'skipped': 0,
                    'filtered': 0
                }
            
            image_files = sorted(list(class_folder.glob("*.png")))     
            print(f"  {class_name}: Processing {len(image_files)} images...")
            
            for img_path in tqdm(image_files, desc=f"    {class_name}", leave=False):
                with Image.open(img_path) as img:
                    # Size filtering to ensure images can produce required mpp
                    if min_size_filter and (img.width < min_size_filter or img.height < min_size_filter):
                        stats['by_class'][class_name]['filtered'] += 1
                        stats['total_filtered'] += 1
                        continue
                    
                    if img.width < source_patch_size or img.height < source_patch_size:
                        stats['by_class'][class_name]['skipped'] += 1
                        stats['total_skipped'] += 1
                        continue
                    
                    # Center crop
                    left = (img.width - source_patch_size) // 2
                    top = (img.height - source_patch_size) // 2
                    right = left + source_patch_size
                    bottom = top + source_patch_size  
                    patch = img.crop((left, top, right, bottom))
                    
                    # Apply downsampling method
                    if source_patch_size != patch_size:
                        patch = downsample_fn(patch, source_patch_size, patch_size)
                    
                    # Save result
                    output_file = output_path / img_path.name
                    patch.save(output_file, compress_level=6)  # 6 is standard I believe, just to be sure
                    stats['by_class'][class_name]['processed'] += 1
                    stats['total_processed'] += 1
                        
    return stats



def calculate_min_size_for_mpps(mpps, patch_size=224, source_mpp=0.25):
    """
    Calculate the minimum source image size needed to create patches at all specified MPPs.
    
    Args:
        mpps: List of target MPP values
        patch_size: Output patch size (default: 224)
        source_mpp: Source MPP (default: 0.25)
    
    Returns:
        Minimum size in pixels required
    """
    max_ratio = max([mpp / source_mpp for mpp in mpps])
    min_size = int(patch_size * max_ratio)
    
    for mpp in sorted(mpps):
        ratio = mpp / source_mpp
        required = int(patch_size * ratio)
        print(f"    {mpp}mpp ({source_mpp/mpp:.1f}x): {required}x{required} pixels")
    
    print(f"  Target MPPs: {mpps}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"\n  ➜ Minimum source image size: {min_size}x{min_size} pixels (Images smaller than this will be filtered out)\n")
    
    return min_size



def create_multiple_magnifications_enhanced(
    source_path,
    base_output_path,
    mpps=[0.25, 0.5, 1.0, 2.0],
    downsampling_method='progressive',  
    ensure_all_mags=True,
    **kwargs
):
    """
    Create datasets at multiple magnifications using enhanced downsampling methods.
    Each magnification gets its own flat directory with all images.
    
    Args:
        source_path: Path to source BRACS_RoI/latest_version directory (downloaded form BRACS website)
        base_output_path: Path where datasets will be created
        mpps: List of target MPP values
        downsampling_method: One of:
            - 'progressive': Progressive downsampling
        ensure_all_mags: If True, only process images that can be scaled to all MPPs
        **kwargs: Additional arguments passed to create_multimag_dataset_enhanced
    """
    base_output_path = Path(base_output_path)
    
    # Calculate minimum size if ensuring all magnifications
    if ensure_all_mags:
        min_size = calculate_min_size_for_mpps(mpps, kwargs.get('patch_size', 224))
        kwargs['min_size_filter'] = min_size
        print(f"Minimum source size filter: {min_size}px (to ensure all magnifications)")
    
    results = {}
    for mpp in mpps:
        print(f"\n{'#'*70}")
        print(f"# Creating dataset for {mpp}mpp using {downsampling_method}")
        print(f"{'#'*70}")
        
        output_path = base_output_path / f"BRACS_{mpp}mpp"
        output_path.mkdir(parents=True, exist_ok=True)

        stats = create_multimag_dataset_enhanced(
            source_path=source_path,
            output_path=output_path,
            target_mpp=mpp,
            downsampling_method=downsampling_method,
            **kwargs
        )
        
        results[mpp] = stats

    # Final summary
    print(f"\n{'#'*70}")
    print("ALL MAGNIFICATIONS COMPLETE")
    print(f"{'#'*70}")
  
    for mpp, stats in results.items():
        print(f"\n{mpp}mpp: {stats['total_processed']} images processed")
        if stats['total_filtered'] > 0:
            print(f"  (Filtered: {stats['total_filtered']}, Skipped: {stats['total_skipped']})")
    
    return results


if __name__ == "__main__":
    source_dataset = "/app/BRACS_RoI/latest_version"
    output_base = "/app/BRACS_multimag_enhanced"
    
    print("Creating BRACS-MS via progressive downsampling for all magnifications")
    results = create_multiple_magnifications_enhanced(
        source_path=source_dataset,
        base_output_path=f"{output_base}/progressive",
        mpps=[0.25, 0.375, 0.5,0.75, 1.0, 1.5, 2.0],
        downsampling_method='progressive',
        ensure_all_mags=True,
        patch_size=224
    )