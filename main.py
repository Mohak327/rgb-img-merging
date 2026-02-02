import cv2
import numpy as np
import os
import glob
import logging
from typing import Tuple, Optional, List
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ProkudinGorskiiAligner:
    """
    A class to automate the alignment and restoration of Prokudin-Gorskii 
    triple-exposure glass plates using multi-scale pyramid alignment and 
    advanced image enhancement (Bells & Whistles).
    """
    def __init__(
        self,
        search_range: int = 15,
        pyramid_depth: int = 5,
        crop_ratio: float = 0.15,
        border_threshold_ratio: float = 0.1,
        contrast_low_percentile: float = 2.0,
        contrast_high_percentile: float = 98.0,
        min_pyramid_size: int = 128
    ) -> None:
        """Initialize aligner with configurable parameters.
        
        Args:
            search_range: Range for exhaustive search (±pixels)
            pyramid_depth: Number of pyramid levels for coarse-to-fine alignment
            crop_ratio: Ratio of border to crop during alignment scoring
            border_threshold_ratio: Max ratio of image to search for border removal
            contrast_low_percentile: Lower percentile for contrast stretching
            contrast_high_percentile: Upper percentile for contrast stretching
            min_pyramid_size: Minimum image dimension before stopping pyramid descent
        
        Raises:
            ValueError: If parameters are out of valid ranges
        """
        if search_range <= 0:
            raise ValueError(f"search_range must be positive, got {search_range}")
        if pyramid_depth < 0:
            raise ValueError(f"pyramid_depth must be non-negative, got {pyramid_depth}")
        if not 0 < crop_ratio < 0.5:
            raise ValueError(f"crop_ratio must be in (0, 0.5), got {crop_ratio}")
        if not 0 < border_threshold_ratio < 0.5:
            raise ValueError(f"border_threshold_ratio must be in (0, 0.5), got {border_threshold_ratio}")
        if not 0 < contrast_low_percentile < contrast_high_percentile < 100:
            raise ValueError("Percentiles must satisfy 0 < low < high < 100")
        if min_pyramid_size <= 0:
            raise ValueError(f"min_pyramid_size must be positive, got {min_pyramid_size}")
            
        self.search_range = search_range
        self.pyramid_depth = pyramid_depth
        self.crop_ratio = crop_ratio
        self.border_threshold_ratio = border_threshold_ratio
        self.contrast_low_percentile = contrast_low_percentile
        self.contrast_high_percentile = contrast_high_percentile
        self.min_pyramid_size = min_pyramid_size
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_features(self, img: np.ndarray) -> np.ndarray:
        """Extracts edge features using Sobel gradients to focus on structure.
        
        Args:
            img: Input grayscale image
            
        Returns:
            Magnitude of gradients (edge strength map)
        """
        img_f = img.astype(np.float32)
        dx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(dx**2 + dy**2)

    def _get_ncc_score(self, ref: np.ndarray, target: np.ndarray) -> float:
        """Calculates the Normalized Cross-Correlation score.
        
        Args:
            ref: Reference image (flattened internally)
            target: Target image to compare
            
        Returns:
            NCC score in [-1, 1], higher is better alignment
        """
        ref_f = ref.flatten()
        target_f = target.flatten()
        
        ref_f -= np.mean(ref_f)
        target_f -= np.mean(target_f)
        
        norm = np.sqrt(np.sum(ref_f**2) * np.sum(target_f**2))
        if norm == 0:
            return 0
        return np.sum(ref_f * target_f) / norm

    def _align_exhaustive(self, ref: np.ndarray, target: np.ndarray, search_range: int) -> Tuple[int, int]:
        """Exhaustive search for best alignment in a local window.
        
        Args:
            ref: Reference image (after feature extraction)
            target: Target image to align
            search_range: Search radius in pixels
            
        Returns:
            Tuple of (dy, dx) representing optimal offset
        """
        best_offset = (0, 0)
        max_score = -1.0
        
        h, w = ref.shape
        ch, cw = int(h * self.crop_ratio), int(w * self.crop_ratio)
        ref_cropped = ref[ch:-ch, cw:-cw]
        
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                shifted = np.roll(target, shift=(dy, dx), axis=(0, 1))
                target_cropped = shifted[ch:-ch, cw:-cw]
                
                score = self._get_ncc_score(ref_cropped, target_cropped)
                if score > max_score:
                    max_score = score
                    best_offset = (dy, dx)
                    
        return best_offset

    def _pyramid_align(self, ref: np.ndarray, target: np.ndarray, depth: int) -> Tuple[int, int]:
        """Recursive multi-scale alignment using coarse-to-fine strategy.
        
        Args:
            ref: Reference image
            target: Target image to align
            depth: Current pyramid depth (0 = finest level)
            
        Returns:
            Tuple of (dy, dx) representing optimal offset at this scale
        """
        if depth == 0 or ref.shape[0] < self.min_pyramid_size:
            ref_feat = self._get_features(ref)
            target_feat = self._get_features(target)
            return self._align_exhaustive(ref_feat, target_feat, self.search_range)
        
        # Downsample
        ref_small = cv2.resize(ref, (0, 0), fx=0.5, fy=0.5)
        target_small = cv2.resize(target, (0, 0), fx=0.5, fy=0.5)
        
        # Coarse alignment
        coarse_dy, coarse_dx = self._pyramid_align(ref_small, target_small, depth - 1)
        
        # Scale back and refine
        refined_dy, refined_dx = coarse_dy * 2, coarse_dx * 2
        ref_feat = self._get_features(ref)
        target_shifted = np.roll(target, shift=(refined_dy, refined_dx), axis=(0, 1))
        target_feat = self._get_features(target_shifted)
        
        adj_dy, adj_dx = self._align_exhaustive(ref_feat, target_feat, search_range=2)
        
        return refined_dy + adj_dy, refined_dx + adj_dx

    def _load_and_preprocess(self, img_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads image, handles grayscale conversion, and normalization.
        
        Args:
            img_path: Path to input image
            
        Returns:
            Tuple of (original_image, normalized_8bit_image)
            
        Raises:
            FileNotFoundError: If image cannot be loaded
            ValueError: If image has unexpected format
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image at {img_path} - possibly unsupported format")

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Normalize to 8-bit for internal alignment calculations
        img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img, img_8u

    def _split_channels(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Splits the vertical plate into Blue, Green, and Red channels.
        
        Args:
            img: Input image to split (height must be divisible by 3)
            
        Returns:
            Tuple of (blue, green, red) channel images
            
        Raises:
            ValueError: If image height is not suitable for splitting
        """
        if img.shape[0] < 3:
            raise ValueError(f"Image too small to split: height={img.shape[0]}")
        h = img.shape[0] // 3
        return img[:h, :], img[h:2*h, :], img[2*h:3*h, :]

    def _automatic_crop(self, img: np.ndarray, threshold_ratio: Optional[float] = None) -> np.ndarray:
        """Removes borders by detecting high-variance edges typical of scanner/plate borders.
        
        Args:
            img: Input BGR image
            threshold_ratio: Max ratio of image to search from edges (uses instance default if None)
            
        Returns:
            Cropped image with borders removed
        """
        if threshold_ratio is None:
            threshold_ratio = self.border_threshold_ratio
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Compute horizontal and vertical projections of gradients
        dx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        dy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        
        v_grad = np.mean(dx, axis=0)
        h_grad = np.mean(dy, axis=1)
        
        # Find boundaries where gradient intensity drops (moving from messy border to image)
        def find_cut(arr: np.ndarray, reverse: bool = False) -> int:
            limit = int(len(arr) * threshold_ratio)
            search_area = arr[:limit] if not reverse else arr[-limit:][::-1]
            # Simple heuristic: find first peak or steady region
            threshold = np.mean(arr) * 0.8
            for i, val in enumerate(search_area):
                if val > threshold: return i if not reverse else len(arr) - i
            return 0 if not reverse else len(arr)

        left = find_cut(v_grad)
        right = find_cut(v_grad, True)
        top = find_cut(h_grad)
        bottom = find_cut(h_grad, True)
        
        return img[top:bottom, left:right]

    def _apply_white_balance(self, img: np.ndarray) -> np.ndarray:
        """Applies 'Gray World' white balance adjustment.
        
        Args:
            img: Input BGR image
            
        Returns:
            White-balanced image
        """
        result = img.astype(np.float32)
        avg_b = np.mean(result[:,:,0])
        avg_g = np.mean(result[:,:,1])
        avg_r = np.mean(result[:,:,2])
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        
        result[:,:,0] *= (avg_gray / avg_b)
        result[:,:,1] *= (avg_gray / avg_g)
        result[:,:,2] *= (avg_gray / avg_r)
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_contrast(self, img: np.ndarray, low_perc: Optional[float] = None, high_perc: Optional[float] = None) -> np.ndarray:
        """Automatically rescales contrast using percentile-based clipping.
        
        Args:
            img: Input BGR image
            low_perc: Lower percentile for clipping (uses instance default if None)
            high_perc: Upper percentile for clipping (uses instance default if None)
            
        Returns:
            Contrast-enhanced image
        """
        if low_perc is None:
            low_perc = self.contrast_low_percentile
        if high_perc is None:
            high_perc = self.contrast_high_percentile
        result = np.zeros_like(img, dtype=np.float32)
        for i in range(3):
            low = np.percentile(img[:,:,i], low_perc)
            high = np.percentile(img[:,:,i], high_perc)
            result[:,:,i] = (img[:,:,i].astype(np.float32) - low) / (high - low + 1e-5)
            
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result

    def _finalize_image(self, b: np.ndarray, g: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Merges channels and applies automatic enhancements.
        
        Args:
            b: Blue channel
            g: Green channel
            r: Red channel
            
        Returns:
            Final enhanced BGR image
        """
        # 1. Merge
        merged = cv2.merge([b, g, r])
        
        # 2. Normalize to 8-bit for internal processing
        merged_8u = cv2.normalize(merged, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 3. Apply enhancement pipeline
        self.logger.debug("Applying automatic white balance")
        balanced = self._apply_white_balance(merged_8u)
        
        self.logger.debug("Applying automatic contrast")
        contrasted = self._apply_contrast(balanced)
        
        self.logger.debug("Applying automatic cropping")
        final = self._automatic_crop(contrasted)
        
        return final


    def _align_single_scale(self, ref: np.ndarray, target: np.ndarray) -> Tuple[int, int]:
        """Simple exhaustive search without pyramid (for small images).
        
        Args:
            ref: Reference image
            target: Target image to align
            
        Returns:
            Tuple of (dy, dx) representing optimal offset
        """
        ref_feat = self._get_features(ref)
        target_feat = self._get_features(target)
        return self._align_exhaustive(ref_feat, target_feat, self.search_range)
    
    def process(self, img_path: str, output_path: str = "restored.jpg") -> np.ndarray:
        """Main pipeline: Load, Split, Align, Merge, Enhance, Save.
        
        Args:
            img_path: Path to input glass plate image
            output_path: Path where restored image will be saved
            
        Returns:
            Final processed image as numpy array
            
        Raises:
            FileNotFoundError: If input image doesn't exist
            ValueError: If image format is invalid
        """
        self.logger.info(f"Processing: {img_path}")
        
        try:
            # 1. Load data
            original_img, processing_img = self._load_and_preprocess(img_path)

            # 2. Split for both processing (8-bit) and final output (original depth)
            b_proc, g_proc, r_proc = self._split_channels(processing_img)
            b_orig, g_orig, r_orig = self._split_channels(original_img)

            # 3. Align (Calculate offsets using 8-bit features)
            self.logger.info("Aligning channels...")
            
            if img_path.lower().endswith(('.jpg', '.jpeg')):
                self.logger.info("Using single-scale alignment for JPEG")
                g_off = self._align_single_scale(b_proc, g_proc)
                r_off = self._align_single_scale(b_proc, r_proc)
            else:
                self.logger.info("Using multi-scale pyramid alignment")
                g_off = self._pyramid_align(b_proc, g_proc, self.pyramid_depth)
                r_off = self._pyramid_align(b_proc, r_proc, self.pyramid_depth)
            self.logger.info(f"Offsets found: Green{g_off}, Red{r_off}")

            # 4. Reconstruct using offsets applied to original channels
            h, w = g_orig.shape
            g_final = cv2.warpAffine(g_orig.astype(np.float32), 
                                      np.float32([[1, 0, g_off[1]], [0, 1, g_off[0]]]),
                                      (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            r_final = cv2.warpAffine(r_orig.astype(np.float32), 
                                      np.float32([[1, 0, r_off[1]], [0, 1, r_off[0]]]),
                                      (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            g_final = g_final.astype(g_orig.dtype)
            r_final = r_final.astype(r_orig.dtype)
            
            # 5. Finalize (Merge + Auto-Crop + Contrast + WB)
            result = self._finalize_image(b_orig, g_final, r_final)
            
            # 6. Save output
            success = cv2.imwrite(output_path, result)
            if not success:
                raise IOError(f"Failed to save image to {output_path}")
                
            self.logger.info(f"Saved: {output_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {img_path}: {str(e)}")
            raise

    def batch_process(self, input_dir: str, output_dir: str, file_list: Optional[List[str]] = None) -> List[str]:
        """Processes all images in input_dir or a specific subset provided in file_list.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory where restored images will be saved
            file_list: Optional list of specific filenames to process
            
        Returns:
            List of successfully processed output file paths
            
        Raises:
            ValueError: If input_dir doesn't exist
        """
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory does not exist: {input_dir}")
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Created output directory: {output_dir}")

        if file_list:
            files = [os.path.join(input_dir, f) for f in file_list]
        else:
            extensions = ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png']
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(input_dir, ext)))
                files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

        self.logger.info(f"Found {len(files)} files to process")
        
        successful_outputs = []
        failed_count = 0

        for f_path in files:
            if not os.path.isfile(f_path):
                self.logger.warning(f"Skipping: {f_path} is not a valid file")
                continue

            base_name = os.path.basename(f_path)
            name_part, _ = os.path.splitext(base_name)
            out_path = os.path.join(output_dir, f"{name_part}_restored.jpg")
            
            try:
                self.process(f_path, out_path)
                successful_outputs.append(out_path)
            except Exception as e:
                self.logger.error(f"Failed to process {base_name}: {e}")
                failed_count += 1
                
        self.logger.info(f"Batch complete: {len(successful_outputs)} successful, {failed_count} failed")
        return successful_outputs

if __name__ == "__main__":
    aligner = ProkudinGorskiiAligner(
        search_range=15,
        pyramid_depth=5,
        crop_ratio=0.15
    )
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    IMAGES_DIR = "images"
    OUTPUT_DIR = "output"
    
    try:
        # Example 1: Process everything in the folder
        if os.path.exists(IMAGES_DIR):
            aligner.batch_process(IMAGES_DIR, OUTPUT_DIR)
        else:
            logger.warning(f"Directory '{IMAGES_DIR}' not found")
            
        # Example 2: Process a specific limited array of images
        # subset = ["melons.tif"]  # Add more filenames here as needed
        # aligner.batch_process(IMAGES_DIR, OUTPUT_DIR, file_list=subset)
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise