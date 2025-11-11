import os.path as osp
import pandas as pd
import configparser

# Detection and ground truth file formats for MOT17
DET_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf')
GT_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'vis')


def get_sportsmot_det_df_from_det(seq_name, data_root_path, config):
    """
    Load MOT detection file using detection files
    """
    seq_path = osp.join(data_root_path, seq_name)  # Sequence path
    det_file_path = osp.join(seq_path, "gt/gt.txt")  # Ground truth file
    det_type = config.det_file

    # Number and order of columns is always assumed to be the same
    det_df = pd.read_csv(det_file_path, header=None)
    det_df = det_df[det_df.columns[:len(DET_COL_NAMES)]]
    det_df.columns = DET_COL_NAMES

    # Bytetrack conf threshold
    if config.det_file == 'byte065' or config.det_file == "byte065_trainval":
        det_df = det_df[det_df['conf'].ge(0.65)].copy()

    # Coordinates are 1 based
    det_df['bb_left'] -= 1
    det_df['bb_top'] -= 1

    # If id already contains an ID assignment (e.g. using tracktor output), keep it
    # if len(det_df['id'].unique()) > 1:
    #     det_df['preprocessor_id'] = det_df['id']
    det_df['id'] = -1  # Erase the id column

    # Include frame paths into the dataframe
    det_df['frame_path'] = det_df['frame'].apply(lambda frame_num: osp.join(seq_path, f'img1/{int(frame_num):06}.jpg'))

    assert osp.exists(det_df['frame_path'].iloc[0])  # Sanity check

    # Build scene info dictionary
    info_file_path = osp.join(seq_path, 'seqinfo.ini')
    cp = configparser.ConfigParser()
    cp.read(info_file_path)
    seq_info_dict = {'seq': seq_name,
                     'seq_path': seq_path,
                     'det_file_name': det_type,
                     'frame_height': int(cp.get('Sequence', 'imHeight')),
                     'frame_width': int(cp.get('Sequence', 'imWidth')),
                     'seq_len': int(cp.get('Sequence', 'seqLength')),
                     'fps': int(cp.get('Sequence', 'frameRate')),
                     'has_gt': osp.exists(osp.join(seq_path, 'gt')),
                     'is_gt': False}

    return det_df, seq_info_dict

def get_sportsmot_testset_det_df_from_det(seq_name, data_root_path, config):
    """
    Load MOT detection file using detection files
    """
    det_type = config.det_file
    seq_path = osp.join(data_root_path, seq_name)  # Sequence path
    det_file_path = osp.join(seq_path, f"det/{det_type}.txt")  # Ground truth file

    # Number and order of columns is always assumed to be the same
    det_df = pd.read_csv(det_file_path, header=None)
    det_df = det_df[det_df.columns[:len(DET_COL_NAMES)]]
    det_df.columns = DET_COL_NAMES

    if config.det_thresh is not None:
        det_df = det_df[det_df['conf'].ge(config.det_thresh)].copy()
    else:
    # Bytetrack conf threshold
        if config.det_file == 'byte065' or config.det_file == "byte065_trainval":
            det_df = det_df[det_df['conf'].ge(0.65)].copy()
        elif config.det_file == 'byte050':
            det_df = det_df[det_df['conf'].ge(0.50)].copy()
        elif config.det_file == 'byte040':
            det_df = det_df[det_df['conf'].ge(0.40)].copy()

    # Coordinates are 1 based
    det_df['bb_left'] -= 1
    det_df['bb_top'] -= 1

    # If id already contains an ID assignment (e.g. using tracktor output), keep it
    # if len(det_df['id'].unique()) > 1:
    #     det_df['preprocessor_id'] = det_df['id']
    det_df['id'] = -1  # Erase the id column

    # Include frame paths into the dataframe
    det_df['frame_path'] = det_df['frame'].apply(lambda frame_num: osp.join(seq_path, f'img1/{int(frame_num):06}.jpg'))

    assert osp.exists(det_df['frame_path'].iloc[0])  # Sanity check

    # Build scene info dictionary
    info_file_path = osp.join(seq_path, 'seqinfo.ini')
    cp = configparser.ConfigParser()
    cp.read(info_file_path)
    seq_info_dict = {'seq': seq_name,
                     'seq_path': seq_path,
                     'det_file_name': det_type,
                     'frame_height': int(cp.get('Sequence', 'imHeight')),
                     'frame_width': int(cp.get('Sequence', 'imWidth')),
                     'seq_len': int(cp.get('Sequence', 'seqLength')),
                     'fps': int(cp.get('Sequence', 'frameRate')),
                     'has_gt': osp.exists(osp.join(seq_path, 'gt')),
                     'is_gt': False,
                     'det_thresh': config.det_thresh}

    return det_df, seq_info_dict

def get_sportsmot_gt(seq_name, data_root_path, config):
    """
    Load MOT ground truth file
    """
    seq_path = osp.join(data_root_path, seq_name)  # Sequence path
    gt_file_path = osp.join(seq_path, "gt/gt.txt")  # Ground truth file

    # Read the gt file and assign the column names
    gt_df = pd.read_csv(gt_file_path, header=None)
    gt_df = gt_df[gt_df.columns[:len(GT_COL_NAMES)]]
    gt_df.columns = GT_COL_NAMES

    # Coordinates are 1 based
    gt_df['bb_left'] -= 1
    gt_df['bb_top'] -= 1

    # Clean out unnecessary classes
    gt_df = gt_df[gt_df['label'].isin([1, 2, 7, 8, 12])].copy()  # Classes 7, 8, 12 are 'ambiguous' and are not penalized. Let's keep them for now.

    # Extra bbox values that will be used for id matching
    gt_df['bb_bot'] = (gt_df['bb_top'] + gt_df['bb_height']).values
    gt_df['bb_right'] = (gt_df['bb_left'] + gt_df['bb_width']).values

    return gt_df