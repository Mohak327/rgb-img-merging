import numpy as np
import os
import glob
import logging
import time
from typing import Tuple
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProkudinGorskiiAligner:
    def __init__(self, search_range: int = 15, pyramid_depth: int = 5):
        self.search_range = search_range
        self.pyramid_depth = pyramid_depth
        self.logger = logging.getLogger(self.__class__.__name__)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Scales image to 0-255 range. Essential for 16-bit TIFF vs 8-bit JPG."""
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return np.zeros_like(img, dtype=np.uint8)

    def _get_features(self, img: np.ndarray) -> np.ndarray:
        """Sobel gradients help align content rather than raw brightness."""
        img_f = img.astype(np.float32)
        dx = ndimage.sobel(img_f, axis=1)
        dy = ndimage.sobel(img_f, axis=0)
        return np.sqrt(dx**2 + dy**2)

    def _get_ncc_score(self, ref: np.ndarray, target: np.ndarray) -> float:
        """Normalized Cross-Correlation score."""
        ref_f = ref.flatten().astype(np.float32)
        target_f = target.flatten().astype(np.float32)
        ref_f -= np.mean(ref_f)
        target_f -= np.mean(target_f)
        norm = np.sqrt(np.sum(ref_f**2) * np.sum(target_f**2))
        return np.sum(ref_f * target_f) / (norm + 1e-6)

    def _shift_image(self, img: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """Linear shift with zero-padding. Prevents the 'wrap-around' color edges."""
        h, w = img.shape
        res = np.zeros_like(img)
        dst_y1, dst_y2 = max(0, dy), min(h, h + dy)
        dst_x1, dst_x2 = max(0, dx), min(w, w + dx)
        src_y1, src_y2 = max(0, -dy), min(h, h - dy)
        src_x1, src_x2 = max(0, -dx), min(w, w - dx)
        if dst_y1 < dst_y2 and dst_x1 < dst_x2:
            res[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
        return res

    def _align_exhaustive(self, ref: np.ndarray, target: np.ndarray, s_range: int) -> Tuple[int, int]:
        """The 'Single-Scale' method: finds best offset in a local window."""
        best_offset = (0, 0)
        max_score = -1.0
        h, w = ref.shape
        # Ignore messy 10% outer edges during scoring
        ch, cw = int(h * 0.1), int(w * 0.1)
        ref_c = ref[ch:-ch, cw:-cw]
        
        for dy in range(-s_range, s_range + 1):
            for dx in range(-s_range, s_range + 1):
                shifted = self._shift_image(target, dy, dx)
                target_c = shifted[ch:-ch, cw:-cw]
                score = self._get_ncc_score(ref_c, target_c)
                if score > max_score:
                    max_score, best_offset = score, (dy, dx)
        return best_offset

    def _pyramid_align(self, ref: np.ndarray, target: np.ndarray, depth: int) -> Tuple[int, int]:
        """The 'Pyramid' method: recursive coarse-to-fine alignment."""
        if depth == 0 or ref.shape[0] < 128:
            return self._align_exhaustive(self._get_features(ref), self._get_features(target), self.search_range)
        
        ref_s = zoom(ref, 0.5, order=1)
        tgt_s = zoom(target, 0.5, order=1)
        c_dy, c_dx = self._pyramid_align(ref_s, tgt_s, depth - 1)
        
        ref_dy, ref_dx = c_dy * 2, c_dx * 2
        target_shifted = self._shift_image(target, ref_dy, ref_dx)
        adj_dy, adj_dx = self._align_exhaustive(self._get_features(ref), self._get_features(target_shifted), 2)
        return ref_dy + adj_dy, ref_dx + adj_dx

    def process(self, img_path: str, output_path: str = "restored.jpg"):
        start_time = time.time()
        self.logger.info(f"Processing: {img_path}")
        img_raw = np.array(Image.open(img_path))
        
        # 1. Pre-process and Split
        if len(img_raw.shape) == 3:
            img_raw = np.dot(img_raw[...,:3], [0.299, 0.587, 0.114])
        
        img = self._normalize(img_raw)
        h_part = img.shape[0] // 3
       
        # Plate order: Blue (Top), Green (Middle), Red (Bottom)
        b, g, r = img[:h_part, :], img[h_part:2*h_part, :], img[2*h_part:3*h_part, :]

        # 2. Conditional Alignment Logic
        is_tiff = img_path.lower().endswith(('.tif', '.tiff'))
        if is_tiff:
            self.logger.info("Using Multi-Scale Pyramid Alignment")
            g_off = self._pyramid_align(b, g, self.pyramid_depth)
            r_off = self._pyramid_align(b, r, self.pyramid_depth)
        else:
            self.logger.info("Using Single-Scale Exhaustive Alignment")
            g_off = self._align_exhaustive(self._get_features(b), self._get_features(g), self.search_range)
            r_off = self._align_exhaustive(self._get_features(b), self._get_features(r), self.search_range)

        self.logger.info(f"Offsets: G{g_off} R{r_off}")

        # 3. Shift and Merge
        g_final = self._shift_image(g, g_off[0], g_off[1])
        r_final = self._shift_image(r, r_off[0], r_off[1])
        result = np.stack([r_final, g_final, b], axis=-1)

        # 4. Clean up the edges (Intersection Crop)
        h, w, _ = result.shape
        top, bottom = max(0, g_off[0], r_off[0]), h + min(0, g_off[0], r_off[0])
        left, right = max(0, g_off[1], r_off[1]), w + min(0, g_off[1], r_off[1])
        
        # Add 5% safety crop to remove physical plate borders
        shave_h, shave_w = int(h * 0.05), int(w * 0.05)
        result = result[top+shave_h : bottom-shave_h, left+shave_w : right-shave_w]

        # 5. Final Contrast Enhancement
        for i in range(3):
            low, high = np.percentile(result[...,i], (2, 98))
            if high > low:
                result[...,i] = np.clip((result[...,i].astype(np.float32) - low) / (high - low) * 255, 0, 255)
        
        result = result.astype(np.uint8)
        Image.fromarray(result).save(output_path, quality=95)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Saved: {output_path} (took {elapsed_time:.1f}s)")
        return result

    def batch_process(self, input_dir: str, output_dir: str, file_list: list = None):
        """Process all images in input_dir, or specific files if file_list is provided."""
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)
        
        if file_list:
            files = [os.path.join(input_dir, f) for f in file_list]
        else:
            exts = ['*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.png']
            files = []
            for e in exts:
                files.extend(glob.glob(os.path.join(input_dir, e)))
                files.extend(glob.glob(os.path.join(input_dir, e.upper())))
        
        for file in files:
            name = os.path.basename(file).rsplit('.', 1)[0]
            out = os.path.join(output_dir, f"{name}_restored.jpg")
            try:
                self.process(file, out)
            except Exception as e:
                self.logger.error(f"Error processing {file}: {e}")

if __name__ == "__main__":
    aligner = ProkudinGorskiiAligner()
    
    # Path configuration
    INPUT_FOLDER = "images"
    OUTPUT_FOLDER = "output"
    
    if os.path.exists(INPUT_FOLDER):
        # Process all images in folder
        aligner.batch_process(INPUT_FOLDER, OUTPUT_FOLDER)
        
        # Or process only specific images (uncomment to use):
        # subset = ["siren.tif", "cathedral.jpg"]
        # aligner.batch_process(INPUT_FOLDER, OUTPUT_FOLDER, file_list=subset)
    else:
        print(f"Folder '{INPUT_FOLDER}' not found. Please update the path in the script.")