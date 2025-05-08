import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from pathlib import Path
from scipy.signal import butter, filtfilt
from tifffile import TiffSequence
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
    pattern = os.path.join(frames_folder, 'frame_*.tif*')
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
