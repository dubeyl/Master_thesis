### 
# Utility functions for loading, pre-processing, NMF application, visualisatzion 
# and analysis of calcium muscle imaging in the pupa
###
import os
import subprocess
import numpy as np
import tifffile as tiff
from tifffile import TiffSequence
import tifffile
import shutil
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.decomposition import NMF
from pathlib import Path
from scipy.signal import find_peaks, peak_widths
from skimage.feature import peak_local_max
from tqdm import tqdm
from typing import Optional, Tuple, Literal
import re

##
# Loading, pre-processing and visualization functions for mkv recordings
##

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
    """Extracts frames from an MKV video into 16-bit grayscale .tif images. """
    if verbose:
        print(f"Extracting frames from {mkv_path} to {temp_folder}")
    os.makedirs(temp_folder, exist_ok=True)
    
    command = [
        "ffmpeg", "-i", mkv_path,
        "-vf", r"select='not(eq(n\,0))'", # skip the first frame
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

def save_frames_to_mkv(frames_folder,output_mkv_path, frame_rate=80, frame_template="frame_%04d.tif"):
    """Saves frames to a .mkv video file."""
    ffmpeg_cmd = [
        "ffmpeg", "-framerate", str(frame_rate),
        "-i", os.path.join(frames_folder, frame_template),
        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray16le", "-y", output_mkv_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    # Cleanup temporary directories
    for folder in [frames_folder]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
        os.rmdir(folder)
    
    print(f"Processed video saved to {output_mkv_path}")

def normalize_frame(img, min_val=0, max_val=860):
    return ((img - min_val) / (max_val - min_val) * 255).clip(0, 255).astype(np.uint8)


def crop_video_dynamic(frame, max_width=1000, max_height=800):
    """
    Allows the user to select an ROI interactively.
    Resizes the frame if it's too large for the screen, but keeps original size for cropping.
    """
    frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255  # Normalize
    frame = frame.astype(np.uint8)

    # Get original dimensions
    orig_h, orig_w = frame.shape[:2]
    
    # Compute scaling factor
    scale_w = max_width / orig_w
    scale_h = max_height / orig_h
    scale = min(scale_w, scale_h, 1)  # Scale down but not up

    # Resize only for display
    display_frame = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))

    # Select ROI on resized image
    x_scaled, y_scaled, w_scaled, h_scaled = cv2.selectROI(
        "Select ROI", display_frame, showCrosshair=True, fromCenter=False
    )

    # Convert back to original coordinates
    x, y, w, h = int(x_scaled / scale), int(y_scaled / scale), int(w_scaled / scale), int(h_scaled / scale)

    print(f"Selected ROI : (x={x}, y={y}, w={w}, h={h})")

    # Show cropped original-size frame
    cv2.imshow("Cropped Image", frame[y:y+h, x:x+w])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return x, y, w, h

def reduce_frame(frames: np.ndarray, factor: int = 2, method: str = 'mean') -> np.ndarray:
    """
    Spatially downsample a stack of frames by an integer factor.
    
    Parameters
    ----------
    frames : np.ndarray
        Input array of shape (T, H, W).
    factor : int, optional
        Downsampling factor in both H and W (default=2).
    method : {'mean', 'opencv'}, optional
        'mean'    – block‐wise mean pooling (no deps beyond NumPy).
        'opencv'  – cv2.resize with INTER_AREA interpolation.
    
    Returns
    -------
    reduced : np.ndarray
        Downsampled array of shape (T, H//factor, W//factor), same dtype as input.
    """
    T, H, W = frames.shape
    new_h = H // factor
    new_w = W // factor

    if method == 'mean':
        # crop to an exact multiple of factor
        H_crop = new_h * factor
        W_crop = new_w * factor
        cropped = frames[:, :H_crop, :W_crop]
        # reshape into blocks and average
        reduced = (
            cropped
            .reshape(T, new_h, factor, new_w, factor)
            .mean(axis=(2, 4))
        )
        return reduced.astype(frames.dtype)

    elif method == 'opencv':
        # note: requires `import cv2`
        reduced = np.empty((T, new_h, new_w), dtype=frames.dtype)
        for i in range(T):
            reduced[i] = cv2.resize(
                frames[i],
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )
        return reduced

    else:
        raise ValueError(f"Unknown method {method!r}, choose 'mean' or 'opencv'")


def visualize_two_frames(frame1, frame2, normalize: bool = True, title1: str = "Frame 1", title2: str = "Frame 2"):
    """
    Display two frames side by side and normalize them before showing.
    """
    if normalize:
        frame1 = normalize_frame(frame1)
        frame2 = normalize_frame(frame2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(frame1, cmap='gray')
    axes[0].set_title(title1)
    axes[0].axis('off')

    axes[1].imshow(frame2, cmap='gray')
    axes[1].set_title(title2)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


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

def remove_baseline(frames: np.ndarray, clip: bool = True) -> np.ndarray:
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

##
# NMF application and saving/loading/visualisation functions
##

def apply_nmf(
    frames: np.ndarray,
    output_folder: Path | str | None = None,
    n_components: int = 49,
    max_iter: int = 200,
    tol: float = 1e-4,
    alpha_H: float = 0.0,
    scaling: bool = False,
    save: bool = True,
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

    if scaling:
        # images_flat = images_flat.astype(np.float32) / 65535.0  # scale to [0, 1]
        scaler = MinMaxScaler()
        data_rescaled = scaler.fit_transform(images_flat)
    else: data_rescaled = images_flat

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
        save_file = output_path / 'nmf_components.pkl'
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
            ax.set_title(f"Comp {i}")
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
            ax.set_title(f"C{i}", fontsize=8)
            ax.set_xlim(0, n_frames - 1)
            ax.tick_params(labelsize=6)
        for ax in axes[n_components:]:
            ax.set_visible(False)
        fig.suptitle("NMF Temporal Components normalized", fontsize=14)
        plt.tight_layout()
        plt.show()

    return components, W

def save_nmf(root_folder,components, W, name ="nmf_components.pkl"):
    out_path = os.path.join(root_folder, name)
    with open(out_path, "wb") as f:
        pickle.dump((components, W), f)
    return out_path

def visualize_nmf(spatial_components,temporal_components, n_components=40):
    """
    Plot the spatial and temporal components of an NMF model
    """
    n_cols = 8
    n_rows = int(np.ceil(n_components / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2 * n_rows))
    axes = axes.flatten()
    for i, comp in enumerate(spatial_components):
        ax = axes[i]
        ax.imshow(comp, cmap='gray')
        ax.set_title(f"Comp {i}")
        ax.axis('off')
    # hide any unused axes
    for ax in axes[n_components:]:
        ax.axis('off')
    # fig.suptitle("NMF Spatial Components", fontsize=14)
    plt.tight_layout()
    plt.show()

    # temporal plot
    n_frames, n_components = temporal_components.shape

    # No normalization: plot raw temporal components
    W_plot = temporal_components  # use raw data

    # Layout parameters
    n_cols = 8
    n_rows = int(np.ceil(n_components / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2 * n_rows), sharex=True)
    axes = axes.flatten()

    for i in range(n_components):
        ax = axes[i]
        ax.plot(W_plot[:, i])
        ax.set_title(f"Comp {i}", fontsize=8)
        ax.set_xlim(0, n_frames - 1)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for ax in axes[n_components:]:
        ax.set_visible(False)

    # fig.suptitle("NMF Temporal Components (raw)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

##
# Viualisation, analysis and statistics functions for NMF components
##

def zoom_temporal(start_frame,end_frame,W, component_idx, fps =80):
    """
    Visualise a specific temporal window of a nmf temporal component
    """
    plt.figure(figsize=(10, 5))
    plt.plot(W[start_frame:end_frame, component_idx], label=f"Component {component_idx}")
    plt.title(f"Component {component_idx}")
    #change x axis from frames to seconds
    total_frames = end_frame - start_frame
    fps = fps
    start_seconds = start_frame/fps
    step_seconds = np.clip(total_frames//800,1,10) # Show every 5 seconds

    # Compute positions and labels
    second_ticks = range(0, total_frames // fps + 1, step_seconds)
    frame_positions = [s * fps for s in second_ticks]   
    second_labels = [s + start_seconds for s in second_ticks]       

    # Apply to plot
    plt.xticks(frame_positions, second_labels)
    plt.xlabel("Time (s)")
    plt.show()

def comp_sparsity(h):
    """
    Compute sparsity of a single spatial component h.
    Sparsity metric: (L1_norm^2) / (L2_norm^2)
    """
    arr = h.ravel()
    L1 = np.sum(np.abs(arr))
    L2 = np.linalg.norm(arr)
    return (L1**2) / (L2**2) if L2 > 0 else np.nan

def fractional_threshold(component, fraction=0.2):
    """
    Apply a threshold to a spatial component based on a fraction of its max value
    """
    # Normalize to 0–255
    comp = component.astype(np.float32)
    comp = 255 * (comp - comp.min()) / (comp.max() - comp.min())
    comp = comp.astype(np.uint8)

    # Apply threshold at specified fraction
    thresh_val = int(comp.max() * fraction)
    _, mask = cv2.threshold(comp, thresh_val, 255, cv2.THRESH_BINARY)
    return mask

def participation_ratio(W):
    """
    Calculate participation ratio of the temporal weights matrix W
    """
    cov = W.T @ W
    eigs = np.linalg.eigvalsh(cov)
    eigs = np.clip(eigs, 0, None)
    S = np.sum(eigs)
    return (S * S / np.sum(eigs * eigs)) if S > 0 else 0.0


def mean_ipi(W, fs= 80):
    """
    Calculate the mean inter-peak interval of the temporal components
    """
    all_ipi = []
    T, r = W.shape
    for i in range(r):
        w = W[:, i]
        peaks, _ = find_peaks(w, prominence=2, width=8, rel_height=0.5, distance=8)
        if len(peaks) > 1:
            all_ipi.extend(np.diff(peaks) / fs)
    return np.nan if not all_ipi else np.mean(all_ipi)

def mean_peak_width(W, fs=80, **pkw_kwargs):
    """
    Calculate the mean peak width of the peaks found in the temporal components
    """
    widths = []
    T, r = W.shape
    for i in range(r):
        w = W[:, i]
        peaks, _ = find_peaks(w, prominence=2, width=8, rel_height=0.5, distance=8)
        if len(peaks) >= 1:
            w_widths, _, _, _ = peak_widths(w, peaks, **pkw_kwargs)
            widths.extend((w_widths / fs).tolist())
    return np.nan if not widths else np.mean(widths)

def peak_rate(W, fs=80):
    """
    Calculate the number of peaks per second in the temporal components
    """
    T, r = W.shape
    total_peaks = sum(
        len(find_peaks(W[:, i], prominence=2, width=8, rel_height=0.5, distance=8)[0])
        for i in range(r)
    )
    return total_peaks / (T / fs)

def event_cooccurrence_from_traces(muscle_traces, window=5,
                                   prominence=2, width=8,
                                   rel_height=0.5, distance=8):
    """
    Calculate co-occurrence probability of muscle pairs based on the peak events in their muscle traces.
    """
    n_videos   = len(muscle_traces)
    videos     = np.arange(n_videos)
    pairs_idx  = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    pair_labels = [f"{i+1}-{j+1}" for i,j in pairs_idx]

    cooccurrence = np.zeros((n_videos, len(pairs_idx)))
    for v in range(n_videos):
        traces = muscle_traces[v]
        peaks = [
            find_peaks(tr,
                       prominence=prominence,
                       width=width,
                       rel_height=rel_height,
                       distance=distance
            )[0]
            for tr in traces
        ]
        for k, (i, j) in enumerate(pairs_idx):
            pi, pj = peaks[i], peaks[j]
            if len(pi)==0 or len(pj)==0:
                cooccurrence[v,k] = 0.0
                continue
            diffs = np.abs(pi[:,None] - pj[None,:]) <= window
            p_i2j = np.sum(np.any(diffs,axis=1)) / len(pi)
            p_j2i = np.sum(np.any(diffs.T,axis=1)) / len(pj)
            cooccurrence[v,k] = 0.5*(p_i2j + p_j2i)
    return videos, pair_labels, cooccurrence
