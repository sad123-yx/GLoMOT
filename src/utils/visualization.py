import cv2
import os
import numpy as np

def generate_color(id):
    np.random.seed(id)
    return tuple(np.random.randint(0, 256, size=3).tolist())

def visualize_tracking_results(txt_folder_path, dataset_path, output_folder_path):
    for txt_file_name in os.listdir(txt_folder_path):
        txt_file_path = os.path.join(txt_folder_path, txt_file_name)
        seq_name = txt_file_name.split(".")[0]
        seq_image_folder_path = os.path.join(dataset_path, seq_name, "img1")
        with open(txt_file_path, 'r') as f:
            lines = f.read().splitlines()
        frame_data_dict = {}
        for line in lines:
            data = line.split(',')
            frame = int(data[0])
            if frame not in frame_data_dict:
                frame_data_dict[frame] = []
            frame_data_dict[frame].append(data)

        output_seq_folder=os.path.join(output_folder_path, seq_name)


        if "dancetrack" in txt_file_name:
            name_zero_num=8
        elif "MOT" in txt_file_name:
            name_zero_num = 6
        else:
            name_zero_num = 6
        for frame, frame_data in frame_data_dict.items():

            img_file_name = str(frame).zfill(name_zero_num) + ".jpg"
            img_file_path = os.path.join(seq_image_folder_path, img_file_name)


            img = cv2.imread(img_file_path)
            if img is None:
                continue


            for target_data in frame_data:
                id = int(target_data[1])
                bb_left = int(float(target_data[2]))
                bb_top = int(float(target_data[3]))
                width = int(float(target_data[4]))
                bb_height = int(float(target_data[5]))


                color = generate_color(id)


                cv2.rectangle(img, (bb_left, bb_top), (bb_left + width, bb_top + bb_height), color, 2)


                cv2.putText(img, str(id), (bb_left, bb_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            output_img_file_path = os.path.join(output_seq_folder, img_file_name)


            os.makedirs(os.path.dirname(output_img_file_path), exist_ok=True)


            cv2.imwrite(output_img_file_path, img)