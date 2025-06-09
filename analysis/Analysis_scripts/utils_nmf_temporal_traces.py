###
# Utility functions for NMF application and muscle traces extraction on pupa data
###

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from pathlib import Path
from scipy.signal import butter, filtfilt
import tifffile
from tifffile import TiffSequence
from tqdm import tqdm
from typing import Optional, Tuple, Literal
import shutil
import os
import subprocess
import cv2
import re

def get_ordered_recording_paths(pupa_folder):
    """Get ordered recording paths based on the folder structure."""
    pattern = re.compile(r"recording(\d+)_\d{8}_\d{4}")
    recordings = []
    for folder in os.listdir(pupa_folder):
        match = pattern.match(folder)
        if match:
            recording_number = int(match.group(1))
            recordings.append((recording_number, folder))
    recordings.sort(key=lambda x: x[0])
    return recordings

def extract_frames_from_mkv(mkv_path, temp_folder, limit_frames=None, verbose=False):
    """Extracts frames from an MKV video into 16-bit grayscale .tif images while ignoring first frame."""
    if verbose:
        print(f"Extracting frames from {mkv_path} to {temp_folder}")
    os.makedirs(temp_folder, exist_ok=True)
    
    command = [
        "ffmpeg", "-i", mkv_path,
        "-vf", r"select='not(eq(n\,0))'", 
        "-pix_fmt", "gray16le",
    ]
    if limit_frames:
        command += ["-frames:v", str(limit_frames)]
    
    command += [os.path.join(temp_folder, "frame_%04d.tif")]
    
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if verbose:    
        print(f"Extracted frames saved to {temp_folder}")

def load_frames_fast(frames_folder, delete_frames=True):
    # create a glob pattern and load all in one shot
    pattern = os.path.join(frames_folder, 'aligned_*.tif*')
    seq     = TiffSequence(pattern)          # will sort by filename
    frames  = seq.asarray()                  # returns a single np.ndarray
    
    print(f"Loaded {frames.shape[0]} frames of shape {frames.shape[1:]}")

    if delete_frames:
        shutil.rmtree(frames_folder)
    return frames

def temporal_filter(frames, high_cutoff=40.0, fs=80.0, order=3):
    """Apply a Butterworth low‑pass filter along the time axis of a 3D stack."""
    # Design filter in normalized frequency (Nyquist = fs/2)
    nyq = 0.5 * fs
    normal_cutoff = min(high_cutoff / nyq, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    frames.astype(np.float32)  # Ensure float for filtering
    # filtfilt can operate along a specified axis.
    # axis=0 means “time” (n_frames dimension), preserving H×W structure.
    filtered = filtfilt(b, a, frames, axis=0).astype(np.float32)
    return filtered

def reduce_frame(frames: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Spatially downsample a stack of frames by an integer factor.
    
    Parameters
    ----------
    frames : np.ndarray
        Input array of shape (T, H, W).
    factor : int, optional
        Downsampling factor in both H and W (default=2).
    Returns
    -------
    reduced : np.ndarray
        Downsampled array of shape (T, H//factor, W//factor), same dtype as input.
    """
    T, H, W = frames.shape
    new_h = H // factor
    new_w = W // factor

    reduced = np.empty((T, new_h, new_w), dtype=frames.dtype)
    for i in range(T):
        reduced[i] = cv2.resize(
            frames[i],
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )
    return reduced

def remove_baseline(frames: np.ndarray, clip: bool = False) -> np.ndarray:
    """
    Subtract the per-pixel minimum intensity (baseline) from every frame.
    
    Parameters
    ----------
    frames : np.ndarray
        Input array of shape (T, H, W).
    clip : bool, optional
        If True, negative values after subtraction are set to zero (default=True).
    
    Returns
    -------
    corrected : np.ndarray
        Baseline-corrected frames of same shape and dtype as input.
    """
    # Compute baseline (min over time) at each pixel
    baseline = frames.min(axis=0)

    # Do subtraction in a safe dtype to avoid underflow
    working = frames.astype(np.float32) - baseline.astype(np.float32)

    if clip:
        working = np.clip(working, 0, None)

    # Cast back to original dtype
    return working.astype(frames.dtype)


def apply_nmf(
    frames: np.ndarray,
    output_folder: Path | str | None = None,
    n_components: int = 49,
    max_iter: int = 200,
    tol: float = 5e-2,
    alpha_H: float = 0.0,
    save: bool = True,
    save_name: str = 'nmf_components.pkl',
    verbose: int = 1,
    plot: bool = True,
    temporal: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply NMF to a (T, H, W) stack of frames.

    Parameters
    ----------
    frames : np.ndarray
        Input array of shape (T, H, W).
    output_folder : Path or str, optional
        Directory to save the NMF model if save=True.  Must be provided when save=True.
    n_components : int
        Number of NMF components.
    scaling : bool
        If True, rescale the data to [0, 1] before fitting NMF.
    save : bool
        If True, pickle the fitted (components, W) to output_folder.
    verbose : bool
        Print progress messages.
    plot : bool
        Show spatial component maps.
    temporal : bool
        Show temporal traces of each component.

    Returns
    -------
    components : np.ndarray
        Array of shape (n_components, H, W) with spatial maps.
    W : np.ndarray
        Array of shape (T, n_components) with temporal weights.
    """
    if verbose not in (0, 1, 2):
        raise ValueError("`verbose` must be 0, 1, or 2")
    if verbose == 2:
        verb = 1
    else: verb = 0
    # validate save/output_folder logic
    if save:
        if output_folder is None:
            raise ValueError("`output_folder` must be provided when save=True")
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    # reshape and scale
    n_frames, height, width = frames.shape
    if verbose in [1,2]:
        print(f"Number of frames: {n_frames}, number of pixels: {height * width}")

    images_flat = frames.reshape(n_frames, -1)

    data_rescaled = images_flat

    # fit NMF
    nmf = NMF(
        n_components=n_components,
        init='nndsvd',
        max_iter=max_iter,
        random_state=42,
        verbose=verb,
        tol=tol,
        alpha_H=alpha_H,
    )
    W = nmf.fit_transform(data_rescaled)  # (n_frames, n_components)
    H = nmf.components_                   # (n_components, pixels)

    if verbose in [1,2]:
        print("NMF fitted")

    # reshape H into spatial components
    expected_pixels = height * width
    if H.shape[1] != expected_pixels:
        raise ValueError(
            f"Cannot reshape components of shape {H.shape} "
            f"into ({n_components}, {height}, {width})"
        )
    components = H.reshape(n_components, height, width)

    # save model if requested
    if save:
        save_file = output_path / save_name
        with open(save_file, 'wb') as f:
            pickle.dump((components, W), f)
        if verbose in [1,2]:
            print(f"Saved NMF model to {save_file!r}")

    # spatial plot
    if plot:
        n_cols = 7
        n_rows = int(np.ceil(n_components / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2 * n_rows))
        axes = axes.flatten()
        for i, comp in enumerate(components):
            ax = axes[i]
            ax.imshow(comp, cmap='gray')
            ax.set_title(f"Comp {i+1}")
            ax.axis('off')
        # hide any unused axes
        for ax in axes[n_components:]:
            ax.axis('off')
        fig.suptitle("NMF Spatial Components", fontsize=14)
        plt.tight_layout()
        plt.show()

    # temporal plot
    if temporal:
        W_min = W.min(axis=0, keepdims=True)
        W_max = W.max(axis=0, keepdims=True)
        # avoid division by zero
        denom = np.where(W_max - W_min == 0, 1, W_max - W_min)
        W_plot = (W - W_min) / denom
        n_cols = 7
        n_rows = int(np.ceil(n_components / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2 * n_rows), sharex=True)
        axes = axes.flatten()
        for i in range(n_components):
            ax = axes[i]
            ax.plot(W_plot[:, i])
            ax.set_title(f"C{i+1}", fontsize=8)
            ax.set_xlim(0, n_frames - 1)
            ax.tick_params(labelsize=6)
        for ax in axes[n_components:]:
            ax.set_visible(False)
        fig.suptitle("NMF Temporal Components normalized", fontsize=14)
        plt.tight_layout()
        plt.show()

    return components, W

def make_ring_by_scaling(mask, scale=2.0):
    # mask: uint8 binary image (0 or 255)
    h, w = mask.shape
    ring = np.zeros_like(mask)

    # find all external contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        # compute centroid
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        # scale contour points about (cx, cy)
        cnt_scaled = []
        for [[x, y]] in cnt:
            x_new = cx + scale * (x - cx)
            y_new = cy + scale * (y - cy)
            cnt_scaled.append([[int(round(x_new)), int(round(y_new))]])
        cnt_scaled = np.array(cnt_scaled, dtype=np.int32)

        # fill the enlarged blob
        cv2.fillPoly(ring, [cnt_scaled], 255)

    # subtract the original mask to leave a ring
    ring = cv2.subtract(ring, mask)
    return ring 

def apply_motion_correction(orig_folder, mcorr_folder, delete = True, offset_x = 0, offset_y = 0, correction = 0):
    def find_landmark(image: np.ndarray,
                  landmark: np.ndarray,
                  search_bbox: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                  metric: Literal[cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED,
                                  cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
                                  cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED] = cv2.TM_CCOEFF_NORMED,
                  subpixel_accuracy: bool = True
                  ) -> Tuple[Tuple[int, int], float]:
        """
        Find the region in an image that most resembles a particular landmark.

        Implementation leverages OpenCV functions, following tutorial at
        https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

        Parameters
        ----------
        image : np.ndarray
            The image in which to search for the landmark.

        landmark : np.ndarray
            The landmark to search for in the image.

        search_bbox : ([axis1min, axis1max], [axis2min, axis2max])
            The bounding box in which to search for the landmark. If you know
            the landmark you're searching for is in a subset of the image then
            specifying a search region will save time.
            If None, the entire image will be searched. If not None, search_bbox
            should be a list/tuple with two elements. Each element should be either
            a slice object or a list/tuple of two integers that specify the min and
            max values of the range.
            For example, search_bbox=(slice(0, 100), slice(0, 200)) or
            search_box=([0, 100], [0, 200]) would search the top-left
            100x200 region of the image.

        metric : cv2.TemplateMatchModes, optional
            The metric to use to compare the landmark to the image. Options are
            cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
            cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED. Default is cv2.TM_CCOEFF_NORMED.

        Returns
        -------
        (top_left, match_score) : tuple
            top_left : tuple
                The top-left corner of the bounding box in the image that
                is most similar to the landmark.
            score : float
                The score of the match, from 0 to 1. A score of 1 indicates
                a perfect match.
        """
        
        if search_bbox is not None:
            if all(isinstance(el, slice) for el in search_bbox):
                img_to_search = image[search_bbox[0], search_bbox[1]]
                search_offset = (search_bbox[0].start, search_bbox[1].start)
            elif all(isinstance(el, (list, tuple)) and len(el) == 2
                    for el in search_bbox):
                img_to_search = image[search_bbox[0][0]:search_bbox[0][1],
                                    search_bbox[1][0]:search_bbox[1][1]]
                search_offset = (search_bbox[0][0], search_bbox[1][0])
            else:
                raise ValueError(f"Invalid search_bbox: {search_bbox}. Must "
                                "be two slices or two (min, max) pairs.")
        else:
            img_to_search = image
            search_offset = (0, 0)

        # Ensure the images are in the same format    
        if img_to_search.dtype == np.uint16:
            img_to_search = img_to_search.astype(np.float32) / 65535.0
        if landmark.dtype == np.uint16:
            landmark = landmark.astype(np.float32) / 65535.0

        # Perform template matching
        scores = cv2.matchTemplate(img_to_search, landmark, metric)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(scores)
        if metric in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc[::-1]  # Includes a flip from (x, y) to (y, x)
            match_score = 1 - min_val
        else:
            top_left = max_loc[::-1]  # Includes a flip from (x, y) to (y, x)
            match_score = max_val
        if not subpixel_accuracy:
            return top_left, match_score

        # Subpixel accuracy: Fit a quadratic to a patch around the best score
        patch_size = 5
        patch_top_left = (max(0, top_left[0] - patch_size//2),
                        max(0, top_left[1] - patch_size//2))
        patch_bottom_right = (min(scores.shape[0], top_left[0] + patch_size//2),
                            min(scores.shape[1], top_left[1] + patch_size//2))
        patch = scores[patch_top_left[0]:patch_bottom_right[0],
                    patch_top_left[1]:patch_bottom_right[1]]

        def eval_quadratic(coeffs, x, y):
            """
            Evaluate a quadratic surface at a given point.

            The quadratic surface is defined as:
                quadratic(x, y) = a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
            """
            a, b, c, d, e, f = coeffs
            return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

        def quadratic_interpolate_peak(patch):
            """
            Fit a 2D quadratic surface to a matrix of values, and find
            the peak position and value of the surface.

            Returns
            -------
            tuple containing
            - The coordinate of the peak value of the quadratic fit to the patch
            - The peak value
            """
            A = [[i**2, j**2, i*j, i, j, 1] for i, j in np.ndindex(patch.shape)]
            z = [patch[i, j] for i, j in np.ndindex(patch.shape)]
            A = np.array(A)
            z = np.array(z)

            coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
            a, b, c, d, e, f = coeffs

            # Find peak by solving gradient = 0:
            # df/di = 2a i + c j + d = 0
            # df/dj = 2b j + c i + e = 0
            A_grad = np.array([[2*a, c],
                            [c, 2*b]])
            b_grad = -np.array([d, e])
            try:
                peak_loc = np.linalg.solve(A_grad, b_grad)
            except np.linalg.LinAlgError:
                peak_loc = np.array([np.nan, np.nan])  # singular matrix, fallback

            peak_value = eval_quadratic(coeffs, peak_loc[0], peak_loc[1])
            return peak_loc, peak_value

        patch_peak, peak_value = quadratic_interpolate_peak(patch)
        image_peak = patch_peak + patch_top_left + search_offset
        image_peak = (round(image_peak[0], ndigits=2),
                    round(image_peak[1], ndigits=2))

        return image_peak, peak_value

    def extract_ref_landmark(temp_folder):
        #Select first frame from the video as reference frame
        for frame in os.listdir(temp_folder):
            if frame.endswith('0001.tif'):
                ref_frame_path = os.path.join(temp_folder, frame)
        ref_image = tifffile.imread(ref_frame_path)
        h, w = ref_image.shape
        crop_size = 400
        y0, x0 = h - int(crop_size*1.5) +offset_y, w // 2 - crop_size // 2 + offset_x
        landmark = ref_image[y0:y0+crop_size, x0:x0+crop_size]
        assert (0 < landmark.shape[0] < h
                and 0 < landmark.shape[1] < w), (
            f"Landmark {landmark.shape} is impossible for image {(h,w)}"
        )
        mark_pack = (h, w, y0, x0, landmark)
        return mark_pack

    def align_frames(tiff_files, mark_pack, mcorr_folder):
        skipped_frames = 0
        _, _, y0, x0, landmark = mark_pack
        for i,file in enumerate(tqdm(tiff_files,desc='Aligning Frames')):
            #skip the first frame
            if i == 0:
                continue
            img = tifffile.imread(file)
            h_img, w_img = img.shape

            # Find landmark in the current image
            # print("  ▶️  img.shape =", img.shape,
            # " dtype =", img.dtype)
            top_left, score = find_landmark(img, landmark)
            if score > 0.7:
                dy = y0 - top_left[0]
                dx = x0 - top_left[1] - correction
                # Apply translation
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned = cv2.warpAffine(img, M, (w_img, h_img), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                # Save aligned image
                out_path = mcorr_folder / f"aligned_{i:04d}.tiff"
                tifffile.imwrite(out_path, aligned.astype(np.uint16))  # adjust dtype if needed
            else: skipped_frames += 1
        print(f"Skipped {skipped_frames} frames due to low resemblance score.")
        return skipped_frames
    
    mcorr_folder = Path(mcorr_folder)
    markpak = extract_ref_landmark(orig_folder)
    nb_skipped = align_frames(
        sorted(Path(orig_folder).glob('frame_*.tif*')),
        markpak,
        mcorr_folder
    )
    print(f"Motion correction done, saved to {mcorr_folder}, with {nb_skipped} skipped frames.")
    if delete:
        #delete the original frames
        shutil.rmtree(orig_folder)

    
def fractional_threshold(component, fraction=0.2):
    # Normalize to 0–255
    comp = component.astype(np.float32)
    comp = 255 * (comp - comp.min()) / (comp.max() - comp.min())
    comp = comp.astype(np.uint8)

    # Apply threshold at specified fraction
    thresh_val = int(comp.max() * fraction)
    _, mask = cv2.threshold(comp, thresh_val, 255, cv2.THRESH_BINARY)
    return mask