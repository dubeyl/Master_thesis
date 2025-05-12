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
    temp_folder = "temp_frames_folder"
    mcorr_folder = "motion_corrected_frames"

    #Start by getting the mask for the heart muscles from NMF of recording 40
    print("Starting the NMF on recording 40 for muscle mask extraction")
    recording_number = 40
    mkv_file_path = os.path.join(pupa_folder,f"{recordings[recording_number][1]}",f"{recordings[recording_number][1]}.mkv")
    utils.extract_frames_from_mkv(mkv_path=mkv_file_path, temp_folder=temp_folder)
    print("Extraction of frames complete")
    os.makedirs(mcorr_folder, exist_ok=True)
    utils.apply_motion_correction(temp_folder=temp_folder, mcorr_folder=mcorr_folder)
    frames = utils.load_frames_fast(frames_folder=mcorr_folder, delete_frames=True)
    masked_frames = frames[:, y:y+h, x:x+w]
    del frames
    reduced_frames = utils.reduce_frame(masked_frames, factor = 2)
    del masked_frames
    temporal_frames = utils.temporal_filter(reduced_frames, high_cutoff=10.0)
    del reduced_frames
    no_min_frames = utils.remove_baseline(temporal_frames, clip =False)
    del temporal_frames
    print("Processing of frames complete")
    #Apply NMF with 40 components, stop criterion at 5e-2
    components, W = utils.apply_nmf(no_min_frames, n_components=40, max_iter=500, alpha_H=1.0, save=False, plot=False, temporal=False)
    otsu_mask_14 = cv2.threshold(components[14].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    otsu_mask_16 = cv2.threshold(components[16].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    otsu_mask_18 = cv2.threshold(components[18].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    otsu_mask_25 = cv2.threshold(components[25].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #save four masks with pickledump
    muscle_path = os.path.join(pupa_folder,"muscle_masks.pkl")
    with open(muscle_path, "wb") as f:
        pickle.dump([otsu_mask_14,otsu_mask_16,otsu_mask_18,otsu_mask_25], f)
    control_mask_14 = utils.make_ring_by_scaling(otsu_mask_14)
    control_mask_16 = utils.make_ring_by_scaling(otsu_mask_16)
    control_mask_18 = utils.make_ring_by_scaling(otsu_mask_18)
    control_mask_25 = utils.make_ring_by_scaling(otsu_mask_25)
    control_path = os.path.join(pupa_folder, "control_masks.pkl")
    with open(control_path, "wb") as f:
        pickle.dump([control_mask_14,control_mask_16,control_mask_18,control_mask_25], f)
    del no_min_frames
    print("Extraction of masks complete")
    print("Starting analysis of all recordings")
    #Apply NMF and get trace of heart muscles and control for all recordings
    for recording_number, folder_name in recordings:
        print(f"Analysing Recording Number: {recording_number}, Folder Name: {folder_name}")
        recording_folder = os.path.join(pupa_folder, folder_name)
        mkv_file_path = os.path.join(recording_folder, f"{folder_name}.mkv")
        #extract frames from the video
        utils.extract_frames_from_mkv(mkv_path=mkv_file_path, temp_folder=temp_folder)
        print("Extracted frames, saved to temp folder: ./temp_frames_folder")
        os.makedirs(mcorr_folder, exist_ok=True)
        utils.apply_motion_correction(temp_folder=temp_folder, mcorr_folder=mcorr_folder)
        #load frames and process them
        frames = utils.load_frames_fast(frames_folder=mcorr_folder, delete_frames=True)
        print("Loaded frames shape:", frames.shape)
        masked_frames = frames[:, y:y+h, x:x+w]
        del frames
        reduced_frames = utils.reduce_frame(masked_frames, factor = 2)
        del masked_frames
        temporal_frames = utils.temporal_filter(reduced_frames, high_cutoff=10.0)
        del reduced_frames
        no_min_frames = utils.remove_baseline(temporal_frames, clip =False)
        del temporal_frames

        #Get the trace of the masks for heart muscles
        masked_frames_14 = np.multiply(no_min_frames, otsu_mask_14)
        temporal_trace_mask14 = np.mean(masked_frames_14, axis=(1, 2))
        masked_frames_16 = np.multiply(no_min_frames, otsu_mask_16)
        temporal_trace_mask16 = np.mean(masked_frames_16, axis=(1, 2))
        masked_frames_18 = np.multiply(no_min_frames, otsu_mask_18)
        temporal_trace_mask18 = np.mean(masked_frames_18, axis=(1, 2))
        masked_frames_25 = np.multiply(no_min_frames, otsu_mask_25)
        temporal_trace_mask25 = np.mean(masked_frames_25, axis=(1, 2))
        #save the temporal traces to a file
        save_name= os.path.join(recording_folder,f"temporal_muscle_traces_{recording_number}_mcorr.pkl")
        with open(save_name, "wb") as f:
            pickle.dump([temporal_trace_mask14, temporal_trace_mask16, temporal_trace_mask18, temporal_trace_mask25], f)
        masked_frames_14 = np.multiply(no_min_frames, control_mask_14)
        temporal_trace_controlmask14 = np.mean(masked_frames_14, axis=(1, 2))
        masked_frames_16 = np.multiply(no_min_frames, control_mask_16)
        temporal_trace_controlmask16 = np.mean(masked_frames_16, axis=(1, 2))
        masked_frames_18 = np.multiply(no_min_frames, control_mask_18)
        temporal_trace_controlmask18 = np.mean(masked_frames_18, axis=(1, 2))
        masked_frames_25 = np.multiply(no_min_frames, control_mask_25)
        temporal_trace_controlmask25 = np.mean(masked_frames_25, axis=(1, 2))
        #save the temporal traces of control masks to a file
        save_name= os.path.join(recording_folder,f"temporal_control_traces_{recording_number}_mcorr.pkl")
        with open(save_name, "wb") as f:
            pickle.dump([temporal_trace_controlmask14, temporal_trace_controlmask16, temporal_trace_controlmask18, temporal_trace_controlmask25], f)    
        
        components,W = utils.apply_nmf(
            no_min_frames,
            save=True,
            output_folder=recording_folder,
            save_name=f"nmf{recording_number}_components_mcorr.pkl",
            n_components=40,
            tol=5e-2,
            max_iter=500,
            alpha_H=1.0,
            plot=False,
            temporal=False)
        del no_min_frames
