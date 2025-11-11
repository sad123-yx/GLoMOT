import os
import shutil
import argparse
import configparser


def process_sequence(seq_path, n, dataset):
    """
    Processes a single video sequence to generate a low-frame-rate version
    with a new seqinfo.ini and re-indexed frame numbers and IDs.

    Args:
        seq_path (str): Path to the original video sequence folder.
        n (int): The frame sampling interval.
    """
    seq_name = os.path.basename(seq_path)
    # For clarity, we will consistently use a name with the -N- suffix
    new_seq_name = f"{seq_name}-N-{n}"
    root_dir = os.path.dirname(seq_path)
    new_seq_path = os.path.join(root_dir, new_seq_name)
    Frame_rate = 30
    print(f"--- Starting processing for sequence: {seq_name} (interval n={n}) ---")
    print(f"Creating new sequence folder: {new_seq_path}")

    # --- 1. Read the original seqinfo.ini to get image dimensions ---
    original_seqinfo_file = os.path.join(seq_path, 'seqinfo.ini')
    imWidth, imHeight = 1920, 1080  # Default values
    original_framerate = 30  # Default original frame rate

    if os.path.exists(original_seqinfo_file):
        try:
            config = configparser.ConfigParser()
            config.read(original_seqinfo_file)
            imWidth = config.getint('Sequence', 'imWidth')
            imHeight = config.getint('Sequence', 'imHeight')
            original_framerate = config.getint('Sequence', 'frameRate')
            print(f"Read dimensions {imWidth}x{imHeight} and original frame rate {original_framerate}fps from original seqinfo.ini")
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
            print(f"Warning: Failed to read original seqinfo.ini file ({e}). Using default dimensions 1920x1080 and default frame rate 30fps.")
    else:
        print("Warning: Original seqinfo.ini not found. Using default dimensions 1920x1080 and default frame rate 30fps.")

    # --- 2. Create directories and check for required files ---
    new_det_path = os.path.join(new_seq_path, 'det')
    new_gt_path = os.path.join(new_seq_path, 'gt')
    new_img1_path = os.path.join(new_seq_path, 'img1')

    os.makedirs(new_det_path, exist_ok=True)
    os.makedirs(new_gt_path, exist_ok=True)
    os.makedirs(new_img1_path, exist_ok=True)

    original_det_file = os.path.join(seq_path, 'det', 'byte065.txt')
    original_gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
    original_img1_path = os.path.join(seq_path, 'img1')

    if not all([os.path.exists(original_det_file),
                os.path.exists(original_gt_file),
                os.path.isdir(original_img1_path)]):
        print(f"Warning: Sequence {seq_name} is missing required files or directories. Skipping.")
        shutil.rmtree(new_seq_path)
        return

    # --- 3. Determine frame selection and create a frame map ---
    image_files = sorted([f for f in os.listdir(original_img1_path) if f.endswith('.jpg')])
    selected_original_frames = []
    for img_file in image_files:
        try:
            frame_number = int(os.path.splitext(img_file)[0])
            if (frame_number - 1) % n == 0:
                selected_original_frames.append(frame_number)
        except ValueError:
            continue  # Ignore files with non-numeric filenames

    frame_map = {orig_frame: new_frame for new_frame, orig_frame in enumerate(selected_original_frames, 1)}

    if not frame_map:
        print(f"Warning: No valid image frames found for sequence {seq_name}. Skipping.")
        shutil.rmtree(new_seq_path)
        return

    print(f"Selected a total of {len(frame_map)} frames.")

    # --- 4. Process img1, det, and gt files ---
    print("Copying and renaming image frames...")
    for original_frame, new_frame in frame_map.items():
        if dataset == "dancetrack":
            original_filename = f"{original_frame:08d}.jpg"
            new_filename = f"{new_frame:08d}.jpg"
            Frame_rate = 20
        else:
            original_filename = f"{original_frame:06d}.jpg"
            new_filename = f"{new_frame:06d}.jpg"
            Frame_rate = 30
        original_filepath = os.path.join(original_img1_path, original_filename)
        new_filepath = os.path.join(new_img1_path, new_filename)
        if os.path.exists(original_filepath):
            shutil.copy2(original_filepath, new_filepath)

    print("Filtering and updating detection (det) results...")
    new_det_file_path = os.path.join(new_det_path, 'byte065.txt')
    with open(original_det_file, 'r') as f_in, open(new_det_file_path, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split(',')
            original_frame = int(float(parts[0]))
            if original_frame in frame_map:
                parts[0] = str(frame_map[original_frame])
                f_out.write(','.join(parts) + '\n')

    print("Filtering, updating, and re-indexing ground truth (gt) results...")
    filtered_gt_data = []
    original_ids_in_new_seq = set()
    with open(original_gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            original_frame = int(float(parts[0]))
            if original_frame in frame_map:
                parts[0] = str(frame_map[original_frame])
                filtered_gt_data.append(parts)
                original_ids_in_new_seq.add(int(float(parts[1])))

    sorted_original_ids = sorted(list(original_ids_in_new_seq))
    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_original_ids, 1)}

    new_gt_file_path = os.path.join(new_gt_path, 'gt.txt')
    with open(new_gt_file_path, 'w') as f_out:
        for parts in filtered_gt_data:
            original_id = int(float(parts[1]))
            if original_id in id_map:
                parts[1] = str(id_map[original_id])
                f_out.write(','.join(parts) + '\n')

    # --- 5. Create a new seqinfo.ini file ---
    print("Generating new seqinfo.ini file...")
    new_framerate = Frame_rate / n
    # Format the frame rate: integer if it's a whole number, otherwise one decimal place
    if new_framerate == int(new_framerate):
        new_framerate_str = str(int(new_framerate))
    else:
        new_framerate_str = f"{new_framerate:.1f}"

    new_seq_length = len(frame_map)

    ini_content = (
        "[Sequence]\n"
        f"name={new_seq_name}\n"
        f"imDir=img1\n"
        f"frameRate={new_framerate_str}\n"
        f"seqLength={new_seq_length}\n"
        f"imWidth={1920}\n"
        f"imHeight={1080}\n"
        "imExt=.jpg\n"
    )

    new_seqinfo_file = os.path.join(new_seq_path, 'seqinfo.ini')
    with open(new_seqinfo_file, 'w') as f:
        f.write(ini_content)

    print(f"Sequence {seq_name} processing complete.\n")

def main():
    """
    Main function to parse arguments and start the processing workflow.
    """
    parser = argparse.ArgumentParser(
        description="Create low-frame-rate versions of multi-object tracking sequences.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        help="Path to the root directory containing all sequence folders.",
        default="E:\python\GLoMOT\datasets\dancetrack/val_lfr/"
    )
    parser.add_argument('--interval', type=int, default=10, help="Frame sampling interval (e.g., 10 means keep 1 frame every 10).")
    parser.add_argument('--dataset', type=str, default="dancetrack", help="The name of the dataset (e.g., 'dancetrack', 'mot17').")
    args = parser.parse_args()

    root_dir = args.root_dir
    n = args.interval
    dataset = args.dataset

    if not os.path.isdir(root_dir):
        print(f"Error: Root directory '{root_dir}' does not exist.")
        return

    print("Start")
    print(f"Starting to process root directory: {root_dir}")
    print(f"Using frame interval: {n}")
    print("-" * 40)

    # Iterate over all items in the root directory
    for item_name in sorted(os.listdir(root_dir)):
        item_path = os.path.join(root_dir, item_name)
        # Ensure it's a directory and the name does not contain '-N-' to process only original sequences
        if os.path.isdir(item_path) and "-N-" not in item_name:
            process_sequence(item_path, n, dataset)

    print("-" * 40)
    print("All sequences processed!")

if __name__ == '__main__':
    main()