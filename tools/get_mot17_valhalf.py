import os
import sys
import shutil

# --- User Configuration ---

# 1. Set the root directory of the dataset.
#    This directory should contain 'train' and 'val' subdirectories.
ROOT_PATH = r'E:\python\GLoMOT\datasets\xxxx/'

# 2. Name of the detection file in the train set.
TRAIN_DET_FILENAME = 'byte065.txt'


# --- End of Configuration ---


def split_sequence(train_seq_path, val_seq_path, seq_name):
    """
    Processes a single sequence, splitting its second half into the validation set.
    """
    print(f"--- Processing sequence: {seq_name} ---")

    # 1. Define source and target paths
    train_det_path = os.path.join(train_seq_path, 'det')
    train_gt_path = os.path.join(train_seq_path, 'gt')
    train_img1_path = os.path.join(train_seq_path, 'img1')

    # Check if source directories exist
    if not all(os.path.isdir(p) for p in [train_det_path, train_gt_path, train_img1_path]):
        print(f"  -> Error: Training set '{seq_name}' has an incomplete directory structure (missing det, gt, or img1). Skipping this sequence.")
        return

    # 2. Clean and create the validation set directory structure
    print("  -> 1. Cleaning and creating validation set directory structure...")
    if os.path.exists(val_seq_path):
        shutil.rmtree(val_seq_path)
    val_det_path = os.path.join(val_seq_path, 'det')
    val_gt_path = os.path.join(val_seq_path, 'gt')
    val_img1_path = os.path.join(val_seq_path, 'img1')
    os.makedirs(val_det_path)
    os.makedirs(val_gt_path)
    os.makedirs(val_img1_path)

    # 3. Calculate the split point
    print("  -> 2. Calculating the video frame split point...")
    try:
        image_files = sorted([f for f in os.listdir(train_img1_path) if f.lower().endswith('.jpg')])
        total_frames = len(image_files)
        split_frame = total_frames // 2  # e.g., 600 frames -> 300, 601 frames -> 300
        print(f"     Total frames: {total_frames}, split point: frame {split_frame}. Validation set starts from frame {split_frame + 1}.")
        if total_frames == 0:
            print("     Warning: Image folder is empty, cannot process.")
            return
        # Get filename padding length, e.g., '000001.jpg' -> 6
        filename_padding = len(os.path.splitext(image_files[0])[0])
    except Exception as e:
        print(f"     Error: Could not process image folder: {e}")
        return

    # 4. Process the det file
    print("  -> 3. Cropping and converting the det file...")
    source_det_file = os.path.join(train_det_path, TRAIN_DET_FILENAME)
    target_det_file = os.path.join(val_det_path, 'byte065.txt')
    val_det_lines = []
    if os.path.exists(source_det_file):
        with open(source_det_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                if frame_id > split_frame:
                    # Reset the frame number
                    parts[0] = str(frame_id - split_frame)
                    val_det_lines.append(','.join(parts))
        with open(target_det_file, 'w') as f:
            f.write('\n'.join(val_det_lines))
        print(f"     Done. Extracted {len(val_det_lines)} detection entries to {target_det_file}")
    else:
        print(f"     Warning: Source det file not found at {source_det_file}")

    # 5. Copy and rename image files
    print("  -> 4. Copying and renaming image files...")
    val_img_count = 0
    for old_frame_num in range(split_frame + 1, total_frames + 1):
        new_frame_num = old_frame_num - split_frame
        source_img_name = f"{old_frame_num:0{filename_padding}d}.jpg"
        target_img_name = f"{new_frame_num:0{filename_padding}d}.jpg"
        source_img_path = os.path.join(train_img1_path, source_img_name)
        target_img_path = os.path.join(val_img1_path, target_img_name)
        if os.path.exists(source_img_path):
            shutil.copy2(source_img_path, target_img_path)
            val_img_count += 1
    print(f"     Done. Copied {val_img_count} images.")

    # 6. Process the gt file (crop, reset frame numbers, reset IDs)
    print("  -> 5. Cropping and converting the gt file (including ID reset)...")
    source_gt_file = os.path.join(train_gt_path, 'gt.txt')
    target_gt_file = os.path.join(val_gt_path, 'gt.txt')

    # Temporarily store the gt data for the second half
    temp_val_gt_data = []
    if os.path.exists(source_gt_file):
        with open(source_gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                if frame_id > split_frame:
                    temp_val_gt_data.append(parts)

        # Re-map IDs for the second half data
        old_id_to_new_id_map = {}
        next_new_id = 1
        final_val_gt_lines = []

        for parts in temp_val_gt_data:
            old_frame_id = int(parts[0])
            old_target_id = parts[1]

            # Reset frame number
            parts[0] = str(old_frame_id - split_frame)

            # Reset ID
            if old_target_id not in old_id_to_new_id_map:
                old_id_to_new_id_map[old_target_id] = next_new_id
                next_new_id += 1

            parts[1] = str(old_id_to_new_id_map[old_target_id])
            final_val_gt_lines.append(','.join(parts))

        with open(target_gt_file, 'w') as f:
            f.write('\n'.join(final_val_gt_lines))
        print(f"     Done. Extracted and reset {len(final_val_gt_lines)} GT entries to {target_gt_file}")
    else:
        print(f"     Warning: Source gt file not found at {source_gt_file}")

    print(f"--- Sequence {seq_name} processed successfully ---\n")


def main():
    """Main function to iterate through the val directory and split data from the train directory."""
    train_path = os.path.join(ROOT_PATH, 'train')
    val_path = os.path.join(ROOT_PATH, 'val')

    if not os.path.isdir(train_path) or not os.path.isdir(val_path):
        print(f"Error: Please ensure that the root directory '{ROOT_PATH}' contains both 'train' and 'val' folders.")
        sys.exit(1)

    # Process sequences based on the directories found in the 'val' folder
    val_sequences = sorted([d for d in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, d))])

    if not val_sequences:
        print(
            "Info: The 'val' directory is empty. No action is required. If you want to split data, please create corresponding empty directories in the 'val' folder.")
        return

    print(f"Starting to split {len(val_sequences)} sequences...\n")

    for seq_name in val_sequences:
        train_seq_path = os.path.join(train_path, seq_name)
        val_seq_path = os.path.join(val_path, seq_name)

        if not os.path.isdir(train_seq_path):
            print(f"Warning: Corresponding sequence '{seq_name}' not found in the 'train' folder. Skipping.\n")
            continue

        split_sequence(train_seq_path, val_seq_path, seq_name)

    print("=== All sequences have been split successfully ===")


if __name__ == '__main__':
    main()