import cv2
import numpy as np
import os
import glob

class ProkudinGorskiiAligner:
    """
    A class to automate the alignment and restoration of Prokudin-Gorskii 
    triple-exposure glass plates using multi-scale pyramid alignment and 
    advanced image enhancement (Bells & Whistles).
    """
    def __init__(self, search_range=15, pyramid_depth=5, crop_ratio=0.15):
        self.search_range = search_range
        self.pyramid_depth = pyramid_depth
        self.crop_ratio = crop_ratio

    def _get_features(self, img):
        """Extracts edge features using Sobel gradients to focus on structure."""
        img_f = img.astype(np.float32)
        dx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(dx**2 + dy**2)

    def _get_ncc_score(self, ref, target):
        """Calculates the Normalized Cross-Correlation score."""
        ref_f = ref.flatten()
        target_f = target.flatten()
        
        ref_f -= np.mean(ref_f)
        target_f -= np.mean(target_f)
        
        norm = np.sqrt(np.sum(ref_f**2) * np.sum(target_f**2))
        if norm == 0:
            return 0
        return np.sum(ref_f * target_f) / norm

    def _align_exhaustive(self, ref, target, search_range):
        """Exhaustive search for best alignment in a local window."""
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

    def _pyramid_align(self, ref, target, depth):
        """Recursive multi-scale alignment."""
        if depth == 0 or ref.shape[0] < 128:
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

    def _load_and_preprocess(self, img_path):
        """Loads image, handles grayscale conversion, and normalization."""
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Normalize to 8-bit for internal alignment calculations
        img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img, img_8u

    def _split_channels(self, img):
        """Splits the vertical plate into Blue, Green, and Red channels."""
        h = img.shape[0] // 3
        return img[:h, :], img[h:2*h, :], img[2*h:3*h, :]

    def _automatic_crop(self, img, threshold_ratio=0.1):
        """
        Removes borders by detecting high-variance edges typical of scanner/plate borders.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Compute horizontal and vertical projections of gradients
        dx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        dy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        
        v_grad = np.mean(dx, axis=0)
        h_grad = np.mean(dy, axis=1)
        
        # Find boundaries where gradient intensity drops (moving from messy border to image)
        def find_cut(arr, reverse=False):
            limit = int(len(arr) * 0.1) # Max 10% search from edge
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

    def _apply_white_balance(self, img):
        """Applies 'Gray World' white balance adjustment."""
        result = img.astype(np.float32)
        avg_b = np.mean(result[:,:,0])
        avg_g = np.mean(result[:,:,1])
        avg_r = np.mean(result[:,:,2])
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        
        result[:,:,0] *= (avg_gray / avg_b)
        result[:,:,1] *= (avg_gray / avg_g)
        result[:,:,2] *= (avg_gray / avg_r)
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_contrast(self, img, low_perc=2, high_perc=98):
        """Automatically rescales contrast using percentile-based clipping."""
        result = np.zeros_like(img, dtype=np.float32)
        for i in range(3):
            low = np.percentile(img[:,:,i], low_perc)
            high = np.percentile(img[:,:,i], high_perc)
            result[:,:,i] = (img[:,:,i].astype(np.float32) - low) / (high - low + 1e-5)
            
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result

    def _finalize_image(self, b, g, r):
        """Merges channels and applies automatic enhancements."""
        # 1. Merge
        merged = cv2.merge([b, g, r])
        
        # 2. Normalize to 8-bit float for internal processing
        merged_8u = cv2.normalize(merged, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 3. Bells & Whistles
        print("  Applying Automatic White Balance...")
        balanced = self._apply_white_balance(merged_8u)
        
        print("  Applying Automatic Contrast...")
        contrasted = self._apply_contrast(balanced)
        
        print("  Applying Automatic Cropping...")
        final = self._automatic_crop(contrasted)
        
        return final


    def _align_single_scale(self, ref, target):
        """Simple exhaustive search without pyramid (for small images)"""
        ref_feat = self._get_features(ref)
        target_feat = self._get_features(target)
        return self._align_exhaustive(ref_feat, target_feat, self.search_range)
    
    def process(self, img_path, output_path="restored.jpg"):
        """Main pipeline: Load, Split, Align, Merge, Enhance, Save."""
        print(f"Processing: {img_path}")
        
        # 1. Load data
        original_img, processing_img = self._load_and_preprocess(img_path)

        # 2. Split for both processing (8-bit) and final output (original depth)
        b_proc, g_proc, r_proc = self._split_channels(processing_img)
        b_orig, g_orig, r_orig = self._split_channels(original_img)

        # 3. Align (Calculate offsets using 8-bit features)
        print("  Aligning channels...")
         # Choose alignment method based on file type
        if img_path.lower().endswith(('.jpg', '.jpeg')):
            print("  Using single-scale alignment...")
            g_off = self._align_single_scale(b_proc, g_proc)
            r_off = self._align_single_scale(b_proc, r_proc)
        else:
            print("  Using multi-scale pyramid alignment...")
            g_off = self._pyramid_align(b_proc, g_proc, self.pyramid_depth)
            r_off = self._pyramid_align(b_proc, r_proc, self.pyramid_depth)
        print(f"  Offsets found: Green{g_off}, Red{r_off}")

        # 4. Reconstruct using offsets applied to original channels
        # Use translation matrix instead of np.roll to avoid wrap-around artifacts
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
        cv2.imwrite(output_path, result)
        
        print(f"  Saved: {output_path}")
        return result

    def batch_process(self, input_dir, output_dir, file_list=None):
        """
        Processes all images in input_dir or a specific subset provided in file_list.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Determine which files to process
        if file_list:
            files = [os.path.join(input_dir, f) for f in file_list]
        else:
            extensions = ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png']
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(input_dir, ext)))
                files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

        print(f"Found {len(files)} files to process.")

        for f_path in files:
            if not os.path.isfile(f_path):
                print(f"Skipping: {f_path} is not a valid file.")
                continue

            base_name = os.path.basename(f_path)
            name_part, _ = os.path.splitext(base_name)
            out_path = os.path.join(output_dir, f"{name_part}_restored.jpg")
            
            try:
                self.process(f_path, out_path)
            except Exception as e:
                print(f"Failed to process {base_name}: {e}")

if __name__ == "__main__":
    aligner = ProkudinGorskiiAligner()
    
    # Configuration
    IMAGES_DIR = "images"
    OUTPUT_DIR = "output"
    
    # Example 1: Process everything in the folder
    aligner.batch_process(IMAGES_DIR, OUTPUT_DIR)
    
    # Example 2: Process a specific limited array of images
    # subset = ["melons.tif"] # Add more filenames here as needed
    
    # if os.path.exists(IMAGES_DIR):
    #     aligner.batch_process(IMAGES_DIR, OUTPUT_DIR, file_list=subset)
    # else:
    #     # Fallback for root-level emir.tif if folder doesn't exist
    #     if os.path.exists("emir.tif"):
    #         aligner.process("emir.tif", "emir_restored.jpg")
    #     else:
    #         print(f"Directory '{IMAGES_DIR}' not found.")