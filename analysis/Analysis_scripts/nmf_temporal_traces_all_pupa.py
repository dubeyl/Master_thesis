### 
# This script processes multiple pupae to preprocess the recordings with motion correction,
# perform NMF on their cropped recordings,
# then extract heart muscle masks from defined ROIs and get their temporal traces using NMF. ###

#The crop parameter is where to crop to keep only the SMA
#The mask_video is the recording number that is used to extract the heart muscle masks.
#The mask_components_idx is a list of indices of the components that are used to extract the heart muscle masks.
#The offset is the offset to apply to the motion correction landmark (to fit on the pupas head)
#The correction is a special correction to apply because of noticed movement after at a certain recording number.
import utils_nmf_temporal_traces as utils
import os
import numpy as np
import pickle
import cv2

if __name__ == "__main__":
    pupa_dict = {
        "pupa_5":{
            "folder": "Z:/data/LD/experiments/imaging/output/Automated_experiment/pupa_5",
            "crop": [247,806,343,580],
            "mask_video": 40,
            "mask_components_idx": [21],
            "offset": (0,0),
            "correction": [44,148]
        },
        "pupa_7":{
            "folder": "Z:/data/LD/experiments/imaging/output/Automated_experiment/pupa_7",
            "crop": [393,554,379,621],
            "mask_video": 46,
            "mask_components_idx": [8,9,16,24],
            "offset": (-200,0),
            "correction": [44,142]
        },
        "pupa_8":{
            "folder": "Z:/data/LD/experiments/imaging/output/Automated_experiment/pupa_8",
            "crop": [266,499,326,607],
            "mask_video": 39,
            "mask_components_idx": [36],
            "offset": (-200,0),
            "correction": [44,138]
        },
        "pupa_9":{
            "folder": "Z:/data/LD/experiments/imaging/output/Automated_experiment/pupa_9",
            "crop": [156,537,300,576],
            "mask_video": 41,
            "mask_components_idx": [13,21,27,28],
            "offset": (-200,-100),
            "correction": [43, 135],
            "start": 46
        },
        "pupa_10":{
            "folder": "Z:/data/LD/experiments/imaging/output/Automated_experiment/pupa_10",
            "crop": [127,724,403,633],
            "mask_video": 40,
            "mask_components_idx": [14,16,18,25],
            "offset": (0,0),
            "correction": [43, 132],
            "start": 44
        }
    }
    # Example usage
    #x,y,w,h = utils.crop_video_dynamic(mean_frame)
    for pupa_name, pupa_info in pupa_dict.items():
        x,y,w,h = pupa_info["crop"]
        pupa_folder = pupa_info["folder"]

        recordings = utils.get_ordered_recording_paths(pupa_folder)
        temp_folder = "temp_frames_folder"
        mcorr_folder = "motion_corrected_frames"
        ref_number = pupa_info["mask_video"]
        mask_components_idx = pupa_info["mask_components_idx"]
        offset_y, offset_x = pupa_info["offset"]
        correction_idx, correction = pupa_info["correction"]
        if ref_number > correction_idx:
            offset_x = offset_x + correction
            x = x - correction
        else: correction = 0

        #Start by getting the mask for the heart muscles from NMF of recording 40
        print(f"Starting the NMF on recording {ref_number} for muscle mask extraction")
        mkv_file_path = os.path.join(pupa_folder,f"{recordings[ref_number][1]}",f"{recordings[ref_number][1]}.mkv")
        utils.extract_frames_from_mkv(mkv_path=mkv_file_path, temp_folder=temp_folder)
        print("Extraction of frames complete")
        os.makedirs(mcorr_folder, exist_ok=True)
        utils.apply_motion_correction(orig_folder=temp_folder, mcorr_folder=mcorr_folder, offset_x = offset_x, offset_y = offset_y, correction =correction)
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
        if len(mask_components_idx)==1:
            mask = utils.fractional_threshold(components[mask_components_idx[0]], fraction=0.2)
            num_labels, labels = cv2.connectedComponents(mask)
            areas = [(labels == i).sum() for i in range(1, num_labels)]
            top4 = np.argsort(areas)[-4:] + 1
            otsu_mask_1,otsu_mask_2,otsu_mask_3,otsu_mask_4 = [(labels == lbl).astype(np.uint8)*255 for lbl in top4]
        else:
            otsu_mask_1 = utils.fractional_threshold(components[mask_components_idx[0]], fraction=0.2)
            otsu_mask_2 = utils.fractional_threshold(components[mask_components_idx[1]], fraction=0.2)
            otsu_mask_3 = utils.fractional_threshold(components[mask_components_idx[2]], fraction=0.2)
            otsu_mask_4 = utils.fractional_threshold(components[mask_components_idx[3]], fraction=0.2)
        #save four masks with pickledump
        muscle_path = os.path.join(pupa_folder,"muscle_masks.pkl")
        with open(muscle_path, "wb") as f:
            pickle.dump([otsu_mask_1,otsu_mask_2,otsu_mask_3,otsu_mask_4], f)
        control_mask_1 = utils.make_ring_by_scaling(otsu_mask_1)
        control_mask_2 = utils.make_ring_by_scaling(otsu_mask_2)
        control_mask_3 = utils.make_ring_by_scaling(otsu_mask_3)
        control_mask_4 = utils.make_ring_by_scaling(otsu_mask_4)
        control_path = os.path.join(pupa_folder, "control_masks.pkl")
        with open(control_path, "wb") as f:
            pickle.dump([control_mask_1,control_mask_2,control_mask_3,control_mask_4], f)
        del no_min_frames
        print("Extraction of masks complete")
        print("Starting analysis of all recordings")
        #Apply NMF and get trace of heart muscles and control for all recordings
        for recording_number, folder_name in recordings:
            if recording_number < pupa_info["start"]:
                continue
            print(f"Analysing Recording Number: {recording_number}, Folder Name: {folder_name}")
            recording_folder = os.path.join(pupa_folder, folder_name)
            mkv_file_path = os.path.join(recording_folder, f"{folder_name}.mkv")
            #extract frames from the video
            utils.extract_frames_from_mkv(mkv_path=mkv_file_path, temp_folder=temp_folder)
            print("Extracted frames, saved to temp folder: ./temp_frames_folder")
            os.makedirs(mcorr_folder, exist_ok=True)
            offset_y, offset_x = pupa_info["offset"]
            if pupa_name == "pupa_9":
                offset_y = offset_y - 100
            if recording_number > correction_idx:
                correction_idx, correction = pupa_info["correction"]
                offset_x = offset_x + correction
            else: correction = 0
            utils.apply_motion_correction(orig_folder=temp_folder, mcorr_folder=mcorr_folder, offset_x = offset_x, offset_y = offset_y, correction =correction)
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
            masked_frames_1 = np.multiply(no_min_frames, otsu_mask_1)
            temporal_trace_mask1 = np.mean(masked_frames_1, axis=(1, 2))
            masked_frames_2 = np.multiply(no_min_frames, otsu_mask_2)
            temporal_trace_mask2 = np.mean(masked_frames_2, axis=(1, 2))
            masked_frames_3 = np.multiply(no_min_frames, otsu_mask_3)
            temporal_trace_mask3 = np.mean(masked_frames_3, axis=(1, 2))
            masked_frames_4 = np.multiply(no_min_frames, otsu_mask_4)
            temporal_trace_mask4 = np.mean(masked_frames_4, axis=(1, 2))
            #save the temporal traces to a file
            save_name= os.path.join(recording_folder,f"temporal_muscle_traces_{recording_number}_mcorr.pkl")
            with open(save_name, "wb") as f:
                pickle.dump([temporal_trace_mask1, temporal_trace_mask2, temporal_trace_mask3, temporal_trace_mask4], f)
            masked_frames_1 = np.multiply(no_min_frames, control_mask_1)
            temporal_trace_controlmask1 = np.mean(masked_frames_1, axis=(1, 2))
            masked_frames_2 = np.multiply(no_min_frames, control_mask_2)
            temporal_trace_controlmask2 = np.mean(masked_frames_2, axis=(1, 2))
            masked_frames_3 = np.multiply(no_min_frames, control_mask_3)
            temporal_trace_controlmask3 = np.mean(masked_frames_3, axis=(1, 2))
            masked_frames_4 = np.multiply(no_min_frames, control_mask_4)
            temporal_trace_controlmask4 = np.mean(masked_frames_4, axis=(1, 2))
            #save the temporal traces of control masks to a file
            save_name= os.path.join(recording_folder,f"temporal_control_traces_{recording_number}_mcorr.pkl")
            with open(save_name, "wb") as f:
                pickle.dump([temporal_trace_controlmask1, temporal_trace_controlmask2, temporal_trace_controlmask3, temporal_trace_controlmask4], f)    
            
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
