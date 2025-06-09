
## 
# Unused deprecated fuctions (used as exploration in other scripts)
##

def get_pupa_contour(frame, plot_contour=True, fill_holes=True):
    """Applies Otsu thresholding and makes a mask with the largest contour."""
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    _, thresh = cv2.threshold(frame.astype(np.uint16), 0, 65535, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
        if fill_holes:
            # Fill internal holes using flood-fill
            h, w = mask.shape
            mask_inv = cv2.bitwise_not(mask)
            floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(mask_inv, floodfill_mask, (0, 0), 255)
            floodfilled_inv = cv2.bitwise_not(mask_inv)
            mask = cv2.bitwise_or(mask, floodfilled_inv)

        if plot_contour:
            masked_frame = (frame * (mask > 0)).astype(np.uint16)
            # Plot the masked frame
            plt.figure(figsize=(8, 6))
            plt.imshow(masked_frame, cmap='gray')
            plt.title("Masked Average Frame")
            plt.axis('off')
            plt.show()
    return mask

def create_fluorescence_mask(frames, percentile: int = 75, plot_mask = True):
    """
    Create a mask based on fluorescence changes over time.
    Uses boxplot outlier detection to identify muscle pixels.
    """
    std_dev = np.std(frames, axis=0)  # Compute temporal standard deviation per pixel
    std_values = std_dev.flatten()
    
    # Compute boxplot statistics
    # Q1 = np.percentile(std_values, 25)
    # Q3 = np.percentile(std_values, 75)
    # IQR = Q3 - Q1
    # threshold = Q3 + 1.5 * IQR  # Outlier threshold
    threshold = np.percentile(std_values, percentile)
    mask = std_dev > threshold  # Keep high-variance pixels
    
    if plot_mask:
        # Plot distribution of standard deviations with outlier threshold
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=std_values)
        plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
        plt.xlabel("Standard Deviation of Pixels")
        plt.title("Pixel Fluctuation Distribution with Outlier Detection")
        plt.legend()
        plt.show()
    
    return mask.astype(np.uint8), std_dev  # Convert to binary mask (0 or 1)

from itertools import cycle
import tempfile
from IPython.display import Video, display

def save_overlay_video(
    video_path: str,
    output_path: str,
    detections: list[np.ndarray],
    component_idxs: int | list[int] = 0,
    crop_origin: tuple[int, int] = (0, 0),
    resize_factor: float = 0.5,
    radius_scale: float = 1.0,
    colors: list[tuple[int, int, int]] | None = None,
    circle_thickness: int = 2
) -> None:
    """
    Overlay circle detections from one or more NMF components onto each frame
    and save to a new video file.

    Parameters
    ----------
    video_path : str
        Path to the input video (e.g. MP4).
    output_path : str
        Path to save the output video with overlays.
    detections : list of np.ndarray
        List where each element is an (N,3) array of (y, x, r) relative to processed frames.
    component_idxs : int or list of ints
        Index or indices of components whose detections to overlay.
    crop_origin : (x_offset, y_offset)
        Top-left of the cropped region used during detection.
    resize_factor : float
        Factor by which frames were downscaled before detection.
    radius_scale : float
        Multiplier for circle radii.
    colors : list of BGR tuples, optional
        Colors for each component. If None, uses a default cycling palette.
    circle_thickness : int
        Thickness of circle borders.
    """
    # Normalize component indices
    if isinstance(component_idxs, int):
        component_idxs = [component_idxs]

    # Setup color palette
    default_colors = [
        (0, 255, 255),
        (0, 128, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 128, 128)
    ]
    if colors is None:
        colors = [c for c, _ in zip(cycle(default_colors), component_idxs)]
    elif len(colors) < len(component_idxs):
        raise ValueError("Provide at least as many colors as components to overlay.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, 20, (width, height))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Cannot open VideoWriter for file {output_path}")

    x_off, y_off = crop_origin

    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for idx, color in zip(component_idxs, colors):
            for y_c, x_c, r_c in detections[idx]:
                x_full = int(x_c * resize_factor + x_off)
                y_full = int(y_c * resize_factor + y_off)
                r_draw = max(1, int(r_c * resize_factor * radius_scale))
                cv2.circle(frame, (x_full, y_full), r_draw, color, circle_thickness)

        writer.write(frame)

    cap.release()
    writer.release()

import cv2
import numpy as np
from itertools import cycle

def save_overlay_video_with_masks(
    video_path: str,
    output_path: str,
    masks: list[np.ndarray],
    crop_origin: tuple[int, int] = (0, 0),
    resize_factor: float = 1.0,
    colors: list[tuple[int, int, int]] | None = None,
    contour_thickness: int = 2
) -> None:
    """
    Overlay binary‐mask contours onto each frame of the base video,
    correctly mapping from your cropped & resized detection space
    back to the original frames.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    output_path : str
        Where to write the annotated video.
    masks : list of np.ndarray
        Each mask is a 2D array (dtype=bool or uint8) in your *processed* frame space.
    crop_origin : (x_offset, y_offset)
        Top‐left corner (in full‐frame coords) of the crop you used before resizing.
    resize_factor : float
        The same factor you used when downscaling the cropped frame.
    colors : list of BGR tuples, optional
        One color per mask. If None, a cycling default palette is used.
    contour_thickness : int
        Line thickness for the drawn contours.
    """
    # --- prepare colors ---
    default_colors = [
        (0, 255, 255),
        (0, 128, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 128, 128)
    ]
    if colors is None:
        colors = [c for c, _ in zip(cycle(default_colors), masks)]
    elif len(colors) < len(masks):
        raise ValueError("Need at least as many colors as masks.")

    # --- open video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    W_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- validate that masks will map inside the frame ---
    w_proc, h_proc = masks[0].shape[1], masks[0].shape[0]
    x_off, y_off   = crop_origin
    # (optional) check bounds:
    if (x_off + w_proc * resize_factor  > W_full  or
        y_off + h_proc * resize_factor  > H_full):
        raise ValueError("Crop + resize would exceed frame dimensions.")

    # --- setup writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W_full, H_full))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Cannot open VideoWriter for file {output_path}")

    # --- process frames ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for mask, color in zip(masks, colors):
            # find contours in processed‐frame space
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # remap each contour point back to full‐frame coords
            scaled_contours = []
            for cnt in contours:
                # cnt shape: (n_pts,1,2) with [x_proc,y_proc]
                cnt = cnt.astype(np.float32)
                cnt[:,:,0] = cnt[:,:,0] * resize_factor + x_off  # x
                cnt[:,:,1] = cnt[:,:,1] * resize_factor + y_off  # y
                scaled_contours.append(cnt.astype(np.int32))

            # draw them
            cv2.drawContours(frame, scaled_contours, -1, color, contour_thickness)

        writer.write(frame)

    cap.release()
    writer.release()



import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
def clustered_heatmap_corr(W):
    if W.ndim != 2:
        raise ValueError(f"W should be 2D, got shape {W.shape}")

    n_frames, n_components = W.shape

    # Create a DataFrame: columns = component indices
    comp_labels = [f"Comp_{i+1}" for i in range(n_components)]
    df = pd.DataFrame(W, columns=comp_labels)

    # Compute correlation matrix
    corr = df.corr()

    dist = 1.0 - corr.values
    # zero out any tiny negative‐to‐zero artifacts
    dist[dist < 0] = 0.0
    condensed = squareform(dist, checks=False)

    # 4. Hierarchical clustering
    link = linkage(condensed, method="average")

    # 5. Dendrogram to get leaf order (no plotting)
    dendro = dendrogram(link, labels=comp_labels, no_plot=True)
    order = dendro["ivl"]  # list of labels in clustered order

    # 6. Reorder corr matrix
    clustered_corr = corr.loc[order, order]

    # 7. Plot heatmap
    fig, ax = plt.subplots()
    #make the plot bigger
    fig.set_size_inches(10, 10)

    im = ax.imshow(clustered_corr.values, aspect="equal")

    # ticks & labels
    ax.set_xticks(range(n_components))
    ax.set_xticklabels(order, rotation=90)
    ax.set_yticks(range(n_components))
    ax.set_yticklabels(order)

    # colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson r")

    ax.set_title("Clustered Temporal‐Component Correlation")
    plt.tight_layout()
    plt.show()

from scipy.signal import welch
from scipy.stats import ttest_ind

def compare_nmf_spectra(
    W_list,
    fs = 60.0,
    nperseg=None,
    noverlap=None,
    window='hann',
    colors=None,
    show=True,
    save=False,
    out_dir='.',
    labels=None
):
    """
    Compare spectral content of NMF temporal components across multiple videos.

    Parameters
    ----------
    W_list : list of ndarray, shape (n_components, n_timepoints)
        Temporal coefficient matrices from NMF for each video.
    fs : float
        Sampling frequency (Hz) of the time series.
    nperseg : int, optional
        Desired length of each segment for Welch's method. Defaults to 256 or shorter if data is shorter.
    noverlap : int, optional
        Number of points to overlap between segments. Defaults to half of `nperseg`.
    window : str or tuple or array_like, optional
        Desired window to use. Default is 'hann'.
    colors : list of color strings, optional
        Colorblind-friendly list of colors for plotting (one per video).
        Default palette is colorblind-safe.
    show : bool, optional
        If True, display plots and print statistics.
    save : bool, optional
        If True, save figures and statistics to `out_dir`.
    out_dir : str, optional
        Directory to save outputs. Default is current working directory.
    labels : list of str, optional
        Names for each video for labeling plots and stats.

    Returns
    -------
    results : dict
        Dictionary containing:
          - freqs: frequency axis (Hz)
          - psd_list: list of PSD arrays (n_videos x n_active_components x n_freqs)
          - centroids: list of arrays of spectral centroids per component
          - median_freqs: list of arrays of median frequencies per component
          - stats: DataFrame of pairwise t-tests on centroids
    """
    # Validate inputs
    n_videos = len(W_list)
    if labels is None:
        labels = [f"Video {i+1}" for i in range(n_videos)]
    elif len(labels) != n_videos:
        raise ValueError("`labels` must match number of W matrices.")

    # Default colorblind-friendly palette
    default_palette = [
        '#0072B2',  # Blue
        '#D55E00',  # Vermilion
        '#009E73',  # Greenish
        '#CC79A7',  # Reddish purple
        '#E69F00',  # Orange
        '#56B4E9',  # Sky blue
        '#F0E442'   # Yellow
    ]
    if colors is None:
        colors = default_palette[:n_videos]
    elif len(colors) != n_videos:
        raise ValueError("`colors` must match number of W matrices.")

    # Prepare storage
    psd_list = []
    centroids_list = []
    median_list = []
    freqs = None

    # Loop over videos
    for idx, W in enumerate(W_list):
        # Ensure W.shape == (n_components, n_timepoints)
        n_comp, n_time = W.shape
        # Skip first two rows (background components)
        if n_comp <= 2:
            raise ValueError("W must have more than 2 components to skip background.")
        # W_active = W[2:, :]
        W_active = W
        n_active, _ = W_active.shape

        # Determine segment length and overlap
        seg = nperseg if nperseg is not None else min(256, n_time)
        seg = min(seg, n_time)
        ovl = noverlap if noverlap is not None else seg // 2
        ovl = min(ovl, seg - 1)

        psds = []
        cents = []
        meds = []
        for comp in W_active:
            f, Pxx = welch(
                comp,
                fs=fs,
                window=window,
                nperseg=seg,
                noverlap=ovl
            )
            if freqs is None:
                freqs = f
            else:
                if not np.allclose(freqs, f):
                    raise ValueError("Frequency axes mismatch across components.")

            psds.append(Pxx)
            cents.append((f * Pxx).sum() / Pxx.sum())
            cum = np.cumsum(Pxx)
            meds.append(f[np.searchsorted(cum, cum[-1] / 2)])

        psd_list.append(np.vstack(psds))
        centroids_list.append(np.array(cents))
        median_list.append(np.array(meds))

    # --- Plotting ---
    # Mean PSD per video
    fig1, ax1 = plt.subplots()
    for color, psd, lbl in zip(colors, psd_list, labels):
        ax1.semilogy(freqs, psd.mean(axis=0), label=lbl, color=color)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD')
    ax1.set_title('Mean Power Spectral Density')
    ax1.legend()

    # Boxplot of spectral centroids with custom colors
    fig2, ax2 = plt.subplots()
    bp = ax2.boxplot(centroids_list, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel('Spectral Centroid (Hz)')
    ax2.set_title('Distribution of Spectral Centroids')

    # Pairwise t-tests on centroids
    stats = []
    for i in range(n_videos):
        for j in range(i+1, n_videos):
            t_stat, p_val = ttest_ind(centroids_list[i], centroids_list[j])
            stats.append({
                'Video A': labels[i],
                'Video B': labels[j],
                'mean_A': centroids_list[i].mean(),
                'mean_B': centroids_list[j].mean(),
                't_stat': t_stat,
                'p_value': p_val
            })
    stats_df = pd.DataFrame(stats)

    # Display or save
    if save:
        os.makedirs(out_dir, exist_ok=True)
        fig1.savefig(os.path.join(out_dir, 'mean_psd.png'))
        fig2.savefig(os.path.join(out_dir, 'spectral_centroids.png'))
        stats_df.to_csv(os.path.join(out_dir, 'centroid_stats.csv'), index=False)

    if show:
        plt.show()
        print("Pairwise t-test results for spectral centroids:")
        print(stats_df)

    return {
        'freqs': freqs,
        'psd_list': psd_list,
        'centroids': centroids_list,
        'median_freqs': median_list,
        'stats': stats_df
    }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

def analyze_nmf_peaks(
    W_list,
    labels=None,
    fs=80,
    height=None,
    distance=None,
    threshold= None,
    width=None,
    prominence=None,
    width_rel_height=0.5,
    show=True,
    save=False,
    out_dir="peak_analysis"
):
    import os
    os.makedirs(out_dir, exist_ok=True)

    all_stats = []

    if labels is None:
        labels = [f"Recording {i}" for i in range(len(W_list))]

    for W, label in zip(W_list, labels):
        W_active = W[:, 2:].T  # exclude first 2 background components
        recording_stats = []

        for i, comp in enumerate(W_active):
            time = np.arange(len(comp)) / fs
            peaks, properties = find_peaks(comp, height=height, distance=distance,threshold=threshold, prominence=prominence, width=width)

            if len(peaks) == 0:
                recording_stats.append({
                    "Recording": label,
                    "Component": i,
                    "NumPeaks": 0,
                    "MeanRisingSpeed": np.nan,
                    "MeanWidth": np.nan
                })
                continue

            widths, width_heights, left_ips, right_ips = peak_widths(comp, peaks, rel_height=width_rel_height)
            widths_sec = widths / fs

            rising_speeds = []
            for peak_idx, left_ip in zip(peaks, left_ips):
                start_idx = int(np.floor(left_ip))
                if start_idx < 0 or start_idx >= peak_idx:
                    continue
                rise_time = (peak_idx - start_idx) / fs
                rise_amplitude = comp[peak_idx] - comp[start_idx]
                rising_speed = rise_amplitude / rise_time if rise_time > 0 else np.nan
                rising_speeds.append(rising_speed)

            # Stats
            mean_rising_speed = np.nanmean(rising_speeds) if rising_speeds else np.nan
            mean_width = np.mean(widths_sec)

            recording_stats.append({
                "Recording": label,
                "Component": i,
                "NumPeaks": len(peaks),
                "MeanRisingSpeed": mean_rising_speed,
                "MeanWidth": mean_width
            })

            # Plotting
            if show or save:
                plt.figure(figsize=(10, 4))
                plt.plot(time, comp, label="Signal")
                plt.plot(time[peaks], comp[peaks], "x", label="Peaks")
                # for peak, l, r in zip(peaks, left_ips, right_ips):
                #     plt.hlines(comp[peak] - width_heights[0], l / fs, r / fs, color="red", linestyle="--", label="Width")
                plt.title(f"{label} - Component {i}")
                plt.xlabel("Time (s)")
                plt.ylabel("Activation")
                plt.legend()
                plt.tight_layout()

                if save:
                    plt.savefig(os.path.join(out_dir, f"{label.replace(' ', '_')}_comp_{i}.png"))
                if show:
                    plt.show()
                else:
                    plt.close()

        all_stats.extend(recording_stats)

    stats_df = pd.DataFrame(all_stats)
    if save:
        stats_df.to_csv(os.path.join(out_dir, "peak_analysis_stats.csv"), index=False)

    return stats_df


def coverage_ratio(S, thresh=0.1):
    """
    S: np.ndarray, shape (n_comp, H, W)
    thresh: fraction of max to threshold at (e.g. 0.1)
    returns: coverage ratios shape (n_comp,)
    """
    n, H, W = S.shape
    ratios = np.zeros(n)
    for i in range(n):
        M = S[i]
        # normalize to [0,1]
        Mn, Mx = M.min(), M.max()
        if Mx == Mn:
            ratios[i] = 1.0
        else:
            MnM = (M - Mn) / (Mx - Mn)
            ratios[i] = np.mean(MnM >= thresh)
    return ratios

def detect_blobs_via_peak_local_max(
    components: np.ndarray,
    bg_idx: np.ndarray,
    min_distance: int = 10,
    threshold_abs: float | None = None,
    threshold_rel: float | None = None,
    exclude_border: bool = False
) -> list[np.ndarray]:
    """
    Detect blob centers in each NMF spatial component by finding 2D local maxima.

    Parameters
    ----------
    components : np.ndarray
        Array of shape (K, H, W) of spatial component maps.
    min_distance : int, optional
        Minimum number of pixels separating peaks in the 2D space.
    threshold_abs : float or None, optional
        Minimum intensity of peaks. If None, no absolute threshold is applied.
    exclude_border : bool, optional
        If True, peaks found within `min_distance` of the image border are excluded.

    Returns
    -------
    detections : list of np.ndarray
        Length-K list where each element is an (N, 3) array of
        (y, x, r) peak centers and an approximate radius r = min_distance/2.
    """
    detections = []
    approx_radius = min_distance / 2.0

    for i,comp in enumerate(components):
        if i in bg_idx:
            continue
        # find 2D local maxima
        coords = peak_local_max(
            comp,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            threshold_rel=threshold_rel,
            exclude_border=exclude_border   
        )  # shape (N, 2) in (row, col)

        if coords.size == 0:
            detections.append(np.zeros((0, 3)))
            continue

        # build (y, x, r) array
        ys = coords[:, 0].reshape(-1, 1)
        xs = coords[:, 1].reshape(-1, 1)
        rs = np.full((coords.shape[0], 1), approx_radius)
        det = np.hstack([ys, xs, rs])
        detections.append(det)

    return detections


def plot_peaks_on_components(
    components: np.ndarray,
    detections: list[np.ndarray],
    radius_factor: float = 1.0,
    figsize: tuple[int, int] | None = None
) -> None:
    """
    Plot each spatial component with circles at 2D peak detections.

    Parameters
    ----------
    components : np.ndarray
        Array of shape (K, H, W).
    detections : list of np.ndarray
        Detected (y, x, r) per component.
    radius_factor : float, optional
        Multiply the detected radius by this factor for plotting.
    figsize : tuple[int,int], optional
        Overall figure size.
    """
    import matplotlib.pyplot as plt

    K, H, W = components.shape
    cols = int(np.ceil(np.sqrt(K)))
    cols = 1 if cols ==0 else cols
    rows = int(np.ceil(K / cols))
    if figsize is None:
        figsize = (cols * 3, rows * 3)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, ax in enumerate(axes[:K]):
        comp = components[idx]
        blobs = detections[idx]

        ax.imshow(comp, cmap='gray')
        ax.set_title(f"Comp {idx+1}")
        ax.axis('off')

        for y, x, r in blobs:
            circle = plt.Circle((x, y), r * radius_factor,
                                edgecolor='yellow', facecolor='none', linewidth=1.2)
            ax.add_patch(circle)

    for ax in axes[K:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()