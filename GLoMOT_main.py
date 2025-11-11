from configs.config import get_arguments, change_dataset_config
from src.utils.deterministic import make_deterministic
from src.tracker.glomot_tracker import GLoMOT_Tracker
from src.tracker.glomot_tracker_with_motion import GLoMOT_Tracker_Motion
from src.data.splits import get_seqs_from_splits
import os.path as osp
from TrackEval.scripts.run_mot_challenge import evaluate_mot17,evaluate_dancetrack,evaluate_sportsmot
from src.utils.visualization import visualize_tracking_results

if __name__ == "__main__":
    config = get_arguments()

    config = change_dataset_config(config)

    make_deterministic(config.seed)

    print("Experiment ID:", config.experiment_path)
    print("Experiment Mode:", config.experiment_mode)
    print("------------------")

    if "smpnet" in config.run_id:
        config.use_smpnet = True
    else:
        config.use_smpnet = False

    # TRAINING
    if config.experiment_mode == 'train':
        seqs, splits = get_seqs_from_splits(data_path=config.data_path, train_split=config.train_splits[0], val_split=config.val_splits[0])
        config.symmetric_edges=False
        glomot_tracker = GLoMOT_Tracker(config=config, seqs=seqs, splits=splits)
        if config.load_train_ckpt:
            print("Loading checkpoint from ", config.model_path)
            glomot_tracker.model = glomot_tracker.load_pretrained_model()
        glomot_tracker.train()
    elif config.experiment_mode == 'train-cval':
        for train_split, val_split in zip(config.train_splits, config.val_splits):
            seqs, splits = get_seqs_from_splits(data_path=config.data_path, train_split=train_split, val_split=val_split)
            glomot_tracker = GLoMOT_Tracker(config=config, seqs=seqs, splits=splits)
            glomot_tracker.train()
            print("####################")
        evaluate_mot17(tracker_path=osp.join(config.experiment_path, 'oracle'), split=config.cval_seqs, data_path=config.data_path, tracker_sub_folder=config.mot_sub_folder, output_sub_folder=config.mot_sub_folder)
        for e in range(1, config.num_epoch+1):
            evaluate_mot17(tracker_path=osp.join(config.experiment_path, 'Epoch' + str(e)), split=config.cval_seqs, data_path=config.data_path, tracker_sub_folder=config.mot_sub_folder, output_sub_folder=config.mot_sub_folder)
    elif config.experiment_mode == 'test':
        seqs, splits = get_seqs_from_splits(data_path=config.data_path, test_split=config.test_splits[0])
        config.symmetric_edges = True
        if config.script_motion =="yes":
            config.use_buffer_motion = True
        elif config.script_motion == "no":
            config.use_buffer_motion = False
        print("Use buffer motion: ", config.use_buffer_motion)
        if config.use_kal_or_diff_motion: #only track
            if config.script_kal_diff_motion=="yes":
                use_kalman_filter=True
            elif config.script_kal_diff_motion=="no":
                use_kalman_filter=False
            glomot_tracker = GLoMOT_Tracker_Motion(config=config, seqs=seqs, splits=splits)
        else:
            glomot_tracker = GLoMOT_Tracker(config=config, seqs=seqs, splits=splits)
        glomot_tracker.model = glomot_tracker.load_pretrained_model()
        data_type = config.dataset_type # 'train', 'val', 'test'
        visualization = False

        # Track
        if config.use_kal_or_diff_motion:
            epoch_val_logs, epoc_val_logs_per_depth = glomot_tracker.track_adjacent_frame_with_buffer_motion(dataset=glomot_tracker.test_dataset, output_path=osp.join(glomot_tracker.config.experiment_path, data_type),
                                                                     mode='test',
                                                                     oracle=False,node_buffer=config.use_node_buffer,
                                                                     buffer_len=config.buffer_len)
        else:
            epoch_val_logs, epoc_val_logs_per_depth = glomot_tracker.track_adjacent_frame_with_buffer(dataset=glomot_tracker.test_dataset, output_path=osp.join(glomot_tracker.config.experiment_path, data_type),
                                                                     mode='test',
                                                                     oracle=False,node_buffer=config.use_node_buffer,
                                                                                                buffer_len=config.buffer_len)
        dataset_name = config.dataset
        if visualization:
            txt_folder = osp.join(glomot_tracker.config.experiment_path, data_type)+"/mot_files/"
            data_folder = ""+dataset_name+"/"+data_type+"/"
            save_folder = glomot_tracker.config.experiment_path+"/vis/"
            visualize_tracking_results(txt_folder, data_folder, save_folder)
        if data_type != "test":
            if dataset_name == "mot17":
                res=evaluate_mot17(tracker_path=osp.join(glomot_tracker.config.experiment_path, data_type), split=glomot_tracker.test_split,
                   data_path=glomot_tracker.config.data_path,
                   tracker_sub_folder=glomot_tracker.config.mot_sub_folder,
                   output_sub_folder=glomot_tracker.config.mot_sub_folder)
                save_result_path=list(res[1]['MotChallenge2DBox'].keys())[0]

            elif dataset_name == "dancetrack":
                res=evaluate_dancetrack(tracker_path=osp.join(glomot_tracker.config.experiment_path, data_type),
                               split=glomot_tracker.test_split,
                               data_path=glomot_tracker.config.data_path,
                               tracker_sub_folder=glomot_tracker.config.mot_sub_folder,
                               output_sub_folder=glomot_tracker.config.mot_sub_folder)
                save_result_path = list(res[1]['MotChallenge2DBox'].keys())[0]
            elif dataset_name == "sportsmot":
                res=evaluate_sportsmot(tracker_path=osp.join(glomot_tracker.config.experiment_path, data_type),
                               split=glomot_tracker.test_split,
                               data_path=glomot_tracker.config.data_path,
                               tracker_sub_folder=glomot_tracker.config.mot_sub_folder,
                               output_sub_folder=glomot_tracker.config.mot_sub_folder)
                save_result_path = list(res[1]['MotChallenge2DBox'].keys())[0]