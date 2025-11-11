import os.path as osp
import os

def get_seqs_from_splits(data_path, train_split=None, val_split=None, test_split=None):
    """
    Get splits that will be used in the experiment
    """
    _SPLITS = {}

    # MOT17 Dets
    mot17_dets = ('SDP', 'FRCNN', 'DPM')
    mot17_only_frcnn = ('FRCNN')

    # Full MOT17-Train
    _SPLITS['mot17-train-all'] = {'mot/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 4, 5, 9, 10, 11, 13) for det in mot17_dets]}
    _SPLITS['mot17-train-split1'] = {'mot/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (4, 5, 9, 11) for det in mot17_dets]}
    _SPLITS['mot17-val-split1'] = {'mot/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 10, 13) for det in mot17_dets]}
    _SPLITS['mot17-train-split2'] = {'mot/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 5, 9, 10, 13) for det in mot17_dets]}
    _SPLITS['mot17-val-split2'] = {'mot/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (4, 11) for det in mot17_dets]}
    _SPLITS['mot17-train-split3'] = {'mot/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 4, 10, 11, 13) for det in mot17_dets]}
    _SPLITS['mot17-val-split3'] = {'mot/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (5, 9) for det in mot17_dets]}

    ########
    _SPLITS['mot17-train'] = {
        'mot/train': [f'MOT17-{seq_num:02}-{mot17_only_frcnn}' for seq_num in (2, 4, 5, 9, 10, 11, 13)]}
    _SPLITS['mot17-debug'] = {
        'mot/train': ["MOT17-02-FRCNN"]}
    _SPLITS['mot17-val'] = {
        'mot/val': [f'MOT17-{seq_num:02}-{mot17_only_frcnn}' for seq_num in (2, 4, 5, 9, 10, 11, 13)]}

    # MOT 17 test set
    _SPLITS['mot17-test-all'] = {
        'mot/test': [f'MOT17-{seq_num:02}-{det}' for seq_num in (1, 3, 6, 7, 8, 12, 14) for det in mot17_dets]}
    _SPLITS['mot17-test'] = {
        'mot/test': ["MOT17-01-FRCNN","MOT17-03-FRCNN","MOT17-06-FRCNN","MOT17-07-FRCNN","MOT17-08-FRCNN","MOT17-12-FRCNN","MOT17-14-FRCNN"]}

    _SPLITS['mot17-val-f2'] = {
        'mot/val': ["MOT17-02-FRCNN-N-2", "MOT17-04-FRCNN-N-2", "MOT17-05-FRCNN-N-2", "MOT17-09-FRCNN-N-2",
                    "MOT17-10-FRCNN-N-2", "MOT17-11-FRCNN-N-2", "MOT17-13-FRCNN-N-2"]}
    _SPLITS['mot17-val-f5'] = {
        'mot/val': ["MOT17-02-FRCNN-N-5", "MOT17-04-FRCNN-N-5", "MOT17-05-FRCNN-N-5", "MOT17-09-FRCNN-N-5",
                    "MOT17-10-FRCNN-N-5", "MOT17-11-FRCNN-N-5", "MOT17-13-FRCNN-N-5"]}
    _SPLITS['mot17-val-f10'] = {
        'mot/val': ["MOT17-02-FRCNN-N-10", "MOT17-04-FRCNN-N-10", "MOT17-05-FRCNN-N-10", "MOT17-09-FRCNN-N-10",
                    "MOT17-10-FRCNN-N-10", "MOT17-11-FRCNN-N-10", "MOT17-13-FRCNN-N-10"]}

    #######
    #######
    # MOT20
    #######
    _SPLITS['mot20-train-all'] = {'mot20/train': [f'MOT20-{seq_num:02}' for seq_num in (1, 2, 3, 5)]}
    _SPLITS['mot20-test-all'] = {'mot20/test': [f'MOT20-{seq_num:02}' for seq_num in (4, 6, 7, 8)]}

    _SPLITS['mot20-train-split1'] = {'mot20/train': [f'MOT20-{seq_num:02}' for seq_num in (1, 2, 3)]}
    _SPLITS['mot20-train-split2'] = {'mot20/train': [f'MOT20-{seq_num:02}' for seq_num in (1, 2, 5)]}
    _SPLITS['mot20-train-split3'] = {'mot20/train': [f'MOT20-{seq_num:02}' for seq_num in (1, 3, 5)]}
    _SPLITS['mot20-train-split4'] = {'mot20/train': [f'MOT20-{seq_num:02}' for seq_num in (2, 3, 5)]}

    _SPLITS['mot20-val-split1'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (5,)]}
    _SPLITS['mot20-val-split2'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (3,)]}
    _SPLITS['mot20-val-split3'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (2,)]}
    _SPLITS['mot20-val-split4'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (1,)]}

    _SPLITS['mot20-train']=_SPLITS['mot20-train-all']
    _SPLITS['mot20-test'] =_SPLITS['mot20-test-all']
    _SPLITS['mot20-test-0608'] = {'mot20/test': ["MOT20-06", "MOT20-08"]}

    #######
    # DanceTrack
    #######
    dancetrack_train_seqs = (1, 2, 6, 8, 12, 15, 16, 20, 23, 24, 27, 29, 32, 33, 37, 39, 44, 45, 49, 51, 52, 53, 55, 57, 61, 62, 66, 68, 69, 72, 74, 75, 80, 82, 83, 86, 87, 96, 98, 99)
    dancetrack_val_seqs = (4, 5, 7, 10, 14, 18, 19, 25, 26, 30, 34, 35, 41, 43, 47, 58, 63, 65, 73, 77, 79, 81, 90, 94, 97)
    dancetrack_test_seqs = (3, 9, 11, 13, 17, 21, 22, 28, 31, 36, 38, 40, 42, 46, 48, 50, 54, 56, 59, 60, 64, 67, 70, 71, 76, 78, 84, 85, 88, 89, 91, 92, 93, 95, 100)
    dancetrack_train_val_seqs =(4, 5, 7, 10, 14, 18, 19, 25, 26, 30, 34, 35, 41, 43, 47, 58, 63, 65, 73, 77, 79, 81, 90, 94, 97, 1, 2, 6, 8, 12, 15, 16, 20, 23, 24, 27, 29, 32, 33, 37, 39, 44, 45, 49, 51, 52, 53, 55, 57, 61, 62, 66, 68, 69, 72, 74, 75, 80, 82, 83, 86, 87, 96, 98, 99)
    assert set(dancetrack_train_seqs + dancetrack_val_seqs + dancetrack_test_seqs) == set([x for x in range(1, 101)]), "Missing sequence in the dancetrack splits"
    assert len(dancetrack_train_seqs + dancetrack_val_seqs + dancetrack_test_seqs) == 100, "Missing or duplicate sequence in the dancetrack splits"

    _SPLITS['dancetrack-train-all'] = {'dancetrack/train': [f'dancetrack{seq_num:04}' for seq_num in dancetrack_train_seqs]}
    _SPLITS['dancetrack-val-all'] = {'dancetrack/val': [f'dancetrack{seq_num:04}' for seq_num in dancetrack_val_seqs]}
    _SPLITS['dancetrack-test-all'] = {'dancetrack/test': [f'dancetrack{seq_num:04}' for seq_num in dancetrack_test_seqs]}
    _SPLITS['dancetrack-train-val'] = {
        'dancetrack/train': [f'dancetrack{seq_num:04}' for seq_num in dancetrack_train_val_seqs]}

    _SPLITS['dancetrack-debug'] = {'dancetrack/train': [f'dancetrack{seq_num:04}' for seq_num in (1,)]}
    _SPLITS['dancetrack-val-debug'] = {'dancetrack/val': [f'dancetrack{seq_num:04}' for seq_num in (18,)]}
    _SPLITS['dancetrack-test'] =_SPLITS['dancetrack-test-all']
    _SPLITS['dancetrack_prune_debug'] = {
        'dancetrack/train': ["dancetrack0029","dancetrack0032","dancetrack0033"]}
    _SPLITS['dancetrack-01'] = {
        'dancetrack/train': ["dancetrack0001"]}
    _SPLITS['dancetrack-train'] = _SPLITS['dancetrack-train-all']
    _SPLITS['dancetrack-val'] = _SPLITS['dancetrack-val-all']
    _SPLITS['dancetrack-test-debug'] = {'dancetrack/test': [f'dancetrack{seq_num:04}' for seq_num in (21,)]}

    resume_txt_path = "E:\python\GLoMOT\datasets\dancetrack/dancetrack-resume.txt"
    with open(resume_txt_path, 'r') as f:
        video_sequences = [line.strip() for line in f if line.strip()]
    _SPLITS['dancetrack-resume'] = {
        'dancetrack/test': [f'{seq_name}' for seq_name in video_sequences if seq_name != "name"]}

    _SPLITS['dancetrack-val-f2'] = {'dancetrack/val_lfr': [f'dancetrack{seq_num:04}-{"N-2"}' for seq_num in dancetrack_val_seqs]}
    _SPLITS['dancetrack-val-f5'] = {
        'dancetrack/val_lfr': [f'dancetrack{seq_num:04}-{"N-5"}' for seq_num in dancetrack_val_seqs]}
    _SPLITS['dancetrack-val-f10'] = {
        'dancetrack/val_lfr': [f'dancetrack{seq_num:04}-{"N-10"}' for seq_num in dancetrack_val_seqs]}
    ########
    # BDD
    ########
    _SPLITS['bdd-val-debug'] = {'bdd/val': [f'{seq_name}' for seq_name in ('b1c66a42-6f7d68ca', 'b1c9c847-3bda4659')]}

    #SportsMOT

    sportsmot_train_folder="E:/python/GLOMOT/datasets/sportsmot/train"
    sportsmot_trainval_folder = "E:/python/GLOMOT/datasets/sportsmot/train_val"
    sportsmot_val_folder = "E:/python/GLOMOT/datasets/sportsmot/val"
    sportsmot_test_folder = "E:/python/GLOMOT/datasets/sportsmot/test"

    sportsmot_train_sequence_list=os.listdir(sportsmot_train_folder)
    sportsmot_trainval_sequence_list=os.listdir(sportsmot_trainval_folder)
    sportsmot_val_sequence_list=os.listdir(sportsmot_val_folder)
    sportsmot_test_sequence_list=os.listdir(sportsmot_test_folder)

    _SPLITS['sportsmot-train'] = {'sportsmot/train': [f'{seq_name}' for seq_name in sportsmot_train_sequence_list]}
    _SPLITS['sportsmot-train-val'] = {'sportsmot/train_val': [f'{seq_name}' for seq_name in sportsmot_trainval_sequence_list]}
    _SPLITS['sportsmot-val'] = {'sportsmot/val': [f'{seq_name}' for seq_name in sportsmot_val_sequence_list]}
    _SPLITS['sportsmot-test'] = {'sportsmot/test': [f'{seq_name}' for seq_name in sportsmot_test_sequence_list]}
    _SPLITS['sportsmot-train-debug'] = {'sportsmot/train': ["v_1LwtoLPw2TU_c006"]}
    _SPLITS['sportsmot-test-debug'] = {'sportsmot/test': ["v_1UDUODIBSsc_c001"]}
    _SPLITS['sportsmot-val-debug'] = {'sportsmot/val': ["v_i2_L4qquVg0_c009"]}

    resume_txt_path="E:\python\GLOMOT\datasets\sportsmot/sportsmot-resume-test.txt"
    with open(resume_txt_path, 'r') as f:
        video_sequences = [line.strip() for line in f if line.strip()]
    _SPLITS['sportsmot-resume-test'] = {'sportsmot/test': [f'{seq_name}' for seq_name in video_sequences if seq_name != "name"]}

    # Ensure that split is valid
    assert train_split in _SPLITS.keys() or train_split is None, "Training split is not valid!"
    assert val_split in _SPLITS.keys() or val_split is None, "Validation split is not valid!"
    assert test_split in _SPLITS.keys() or test_split is None, "Test split is not valid!"


    # Get the sequences to use in the experiment
    seqs = {}
    if train_split is not None:
        # if train_split.split('-')[-1] == 'val':
        #     seqs['train'] = {osp.join(data_path, seqs_path): seq_list for seqs_path, seq_list in
        #                      _SPLITS[train_split].items()}
        # else:
            seqs['train'] = {osp.join(data_path, seqs_path): seq_list for seqs_path, seq_list in
                         _SPLITS[train_split].items()}
    if val_split is not None:
        seqs['val'] = {osp.join(data_path, seqs_path): seq_list for seqs_path, seq_list in
                       _SPLITS[val_split].items()}
    if test_split is not None:
        seqs['test'] = {osp.join(data_path, seqs_path): seq_list for seqs_path, seq_list in
                        _SPLITS[test_split].items()}
    return seqs, (train_split, val_split, test_split)
