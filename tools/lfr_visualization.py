import os
import cv2
import numpy as np


def generate_color(id):
    np.random.seed(id)
    return tuple(np.random.randint(0, 256, size=3).tolist())


def visualize_tracking_results(txt_folder_path, dataset_path, output_folder_path, position, frame_gap):
    # Iterate over all txt files in the txt folder
    for txt_file_name in os.listdir(txt_folder_path):
        txt_file_path = os.path.join(txt_folder_path, txt_file_name)
        seq_name = txt_file_name.split(".")[0]
        # Find the corresponding image folder path for the sequence
        seq_image_folder_path = os.path.join(dataset_path, seq_name, "img1")
        # Read the content of the txt file
        with open(txt_file_path, 'r') as f:
            lines = f.read().splitlines()
        frame_data_dict = {}
        for line in lines:
            data = line.split(',')
            frame = int(data[0])
            if frame not in frame_data_dict:
                frame_data_dict[frame] = []
            frame_data_dict[frame].append(data)

        output_seq_folder = os.path.join(output_folder_path, seq_name)

        # Iterate over the data for each frame
        if "dancetrack" in txt_file_name:
            name_zero_num = 8
        elif "MOT" in txt_file_name:
            name_zero_num = 6
        else:
            name_zero_num = 6
        for frame, frame_data in frame_data_dict.items():
            # Construct the image filename
            img_file_name = str(frame).zfill(name_zero_num) + ".jpg"
            img_file_path = os.path.join(seq_image_folder_path, img_file_name)

            # Read the image
            img = cv2.imread(img_file_path)
            if img is None:
                continue

            cv2.putText(img, 'Tracker: %s  frame: %d  frame gap: %d ' % ("GLoMOT", frame, 10),
                        (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

            # Iterate over each target's data in the current frame
            for target_data in frame_data:
                id = int(target_data[1])
                bb_left = int(float(target_data[2]))
                bb_top = int(float(target_data[3]))
                width = int(float(target_data[4]))
                bb_height = int(float(target_data[5]))

                # Get the color for the corresponding id
                color = generate_color(id)

                # Draw the bounding box
                cv2.rectangle(img, (bb_left, bb_top), (bb_left + width, bb_top + bb_height), color, 2)
                # Annotate the id on the bounding box
                cv2.putText(img, str(id), (bb_left, bb_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if position:
                    cv2.putText(img, str(bb_left), (bb_left + 40, bb_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Construct the output image filename and path
            output_img_file_path = os.path.join(output_seq_folder, img_file_name)

            # Ensure the output folder exists
            os.makedirs(os.path.dirname(output_img_file_path), exist_ok=True)

            # Save the resulting image
            cv2.imwrite(output_img_file_path, img)


# --- Script Execution ---
position = False
data_type = "val_lfr"
frame_gap = 10

track_results_txt_folder = r"E:\python\GLoMOT\output\experiments/"
dataset_name = "dancetrack"
dataset_folder = f"E:/python/GLoMOT/datasets/{dataset_name}/{data_type}/"
save_folder = r"E:\python\GLoMOT\output\experiments\vis/"
visualize_tracking_results(track_results_txt_folder, dataset_folder, save_folder, position, frame_gap)