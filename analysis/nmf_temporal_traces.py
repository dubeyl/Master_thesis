import utils_nmf_temporal_traces as utils
import os
import numpy as np
import pickle
import cv2

if __name__ == "__main__":
    # Example usage
    #x,y,w,h = utils.crop_video_dynamic(mean_frame)
    x=127
    y=724
    w=403
    h=633
    pupa_folder = "Z:/data/LD/experiments/imaging/output/Automated_experiment/pupa_10"
    recordings = utils.get_ordered_recording_paths(pupa_folder)

    #Start by getting the mask for the heart muscles from NMF of recording 40
    print("Starting the NMF on recording 40 for muscle mask extraction")
    recording_number = 40
    mkv_file_path = os.path.join(pupa_folder,f"{recordings[recording_number][1]}",f"{recordings[recording_number][1]}.mkv")
    temp_folder = os.makedirs("./temp_frames", exist_ok=True)
    utils.extract_frames_from_mkv(mkv_path=mkv_file_path, temp_folder=temp_folder)
    print("Extraction of frames complete")
    frames = utils.load_frames_fast(frames_folder=temp_folder, delete_frames=True)
    masked_frames = frames[:, y:y+h, x:x+w]
    del frames
    temporal_frames = utils.temporal_filter(masked_frames, HIGH_CUTOFF=30.0)
    del masked_frames
    reduced_frames = utils.reduce_frame(temporal_frames, factor = 2)
    del temporal_frames
    no_min_frames = utils.remove_baseline(reduced_frames, clip =False)
    del reduced_frames
    print("Processing of frames complete")
    #Apply NMF with 40 components, stop criterion at 5e-2
    components, W = utils.apply_nmf(no_min_frames, n_components=40, max_iter=1500, alpha_H=1.0, save=False, plot=False, temporal=False)
    otsu_mask_12 = cv2.threshold(components[11].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    otsu_mask_17 = cv2.threshold(components[16].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    otsu_mask_19 = cv2.threshold(components[18].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    otsu_mask_29 = cv2.threshold(components[28].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #save four masks with pickledump
    with open(f"muscle_masks.pkl", "wb") as f:
        pickle.dump([otsu_mask_12,otsu_mask_17,otsu_mask_19,otsu_mask_29], f)
    control_mask_12 = utils.make_ring_by_scaling(otsu_mask_12)
    control_mask_17 = utils.make_ring_by_scaling(otsu_mask_17)
    control_mask_19 = utils.make_ring_by_scaling(otsu_mask_19)
    control_mask_29 = utils.make_ring_by_scaling(otsu_mask_29)
    with open(f"control_masks.pkl", "wb") as f:
        pickle.dump([control_mask_12,control_mask_17,control_mask_19,control_mask_29], f)
    del no_min_frames
    print("Extraction of masks complete")
    print("Starting analysis of all recordings")
    #Apply NMF and get trace of heart muscles and control for all recordings
    for recording_number, folder_name in recordings:
        print(f"Analysing Recording Number: {recording_number}, Folder Name: {folder_name}")
        recording_folder = os.path.join(pupa_folder, folder_name)
        mkv_file_path = os.path.join(recording_folder, f"{folder_name}.mkv")
        #extract frames from the video
        temp_folder = os.makedirs("./temp_frames", exist_ok=True)
        utils.extract_frames_from_mkv(mkv_path=mkv_file_path, temp_folder=temp_folder)
        print("Extracted frames, saved to temp folder: ./temp_frames")
        #load frames and process them
        frames = utils.load_frames_fast(frames_folder=temp_folder, delete_frames=True)
        print("Loaded frames shape:", frames.shape)
        masked_frames = frames[:, y:y+h, x:x+w]
        temporal_frames = utils.temporal_filter(masked_frames, HIGH_CUTOFF=30.0)
        print("Applied temporal filter to frames")
        del masked_frames
        reduced_frames = utils.reduce_frame(temporal_frames, factor = 2)
        print("Applied reduction to frames")
        del temporal_frames
        print("Final frames shape:", reduced_frames.shape)
        no_min_frames = utils.remove_baseline(reduced_frames, clip =False)
        print("Applied baseline removal to frames")
        del reduced_frames

        #Get the trace of the masks for heart muscles
        masked_frames_12 = np.multiply(no_min_frames, otsu_mask_12)
        temporal_trace_mask12 = np.mean(masked_frames_12, axis=(1, 2))
        masked_frames_17 = np.multiply(no_min_frames, otsu_mask_17)
        temporal_trace_mask17 = np.mean(masked_frames_17, axis=(1, 2))
        masked_frames_19 = np.multiply(no_min_frames, otsu_mask_19)
        temporal_trace_mask19 = np.mean(masked_frames_19, axis=(1, 2))
        masked_frames_29 = np.multiply(no_min_frames, otsu_mask_29)
        temporal_trace_mask29 = np.mean(masked_frames_29, axis=(1, 2))
        #save the temporal traces to a file
        save_name= os.path.join(recording_folder,f"temporal_muscle_traces_{recording_number}.pkl")
        with open(save_name, "wb") as f:
            pickle.dump([temporal_trace_mask12, temporal_trace_mask17, temporal_trace_mask19, temporal_trace_mask29], f)
        masked_frames_12 = np.multiply(no_min_frames, control_mask_12)
        temporal_trace_controlmask12 = np.mean(masked_frames_12, axis=(1, 2))
        masked_frames_17 = np.multiply(no_min_frames, control_mask_17)
        temporal_trace_controlmask17 = np.mean(masked_frames_17, axis=(1, 2))
        masked_frames_19 = np.multiply(no_min_frames, control_mask_19)
        temporal_trace_controlmask19 = np.mean(masked_frames_19, axis=(1, 2))
        masked_frames_29 = np.multiply(no_min_frames, control_mask_29)
        temporal_trace_controlmask29 = np.mean(masked_frames_29, axis=(1, 2))
        #save the temporal traces of control masks to a file
        save_name= os.path.join(recording_folder,f"temporal_control_traces_{recording_number}.pkl")
        with open(save_name, "wb") as f:
            pickle.dump([temporal_trace_controlmask12, temporal_trace_controlmask17, temporal_trace_controlmask19, temporal_trace_controlmask29], f)    
        
        components,W = utils.apply_nmf(
            no_min_frames,
            save=True,
            output_folder=recording_folder,
            save_name=f"nmf{recording_number}_components.pkl",
            n_components=40,
            tol=5e-2,
            max_iter=1500,
            alpha_H=1.0,
            plot=False,
            temporal=False)
        del no_min_frames
