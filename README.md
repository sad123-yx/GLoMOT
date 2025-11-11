<h1 align="center">
  <i>GLoMOT: Efficient Online GNN-based Low-Frame-Rate Multi-Object Tracker</i>
</h1>

# 1. Install Requirements

```
pip3 install -r requirements.txt
```

# 2. Dataset File Structure

```
/datasets/ 
|-- MOT17/
|	|--  train/
|	|--  test/
|	|-- val/
|	     |-- MOT17-XX_FRCNN-N-2
|	     |-- MOT17-XX_FRCNN-N-5
|	     |-- MOT17-XX_FRCNN-N-10
|-- DacneTrack/
|	|--  train/
|	|--  val/
|	|--  test/
|	|-- val_lfr/
|	     |--  dancetrack00XX-N-2
|	     |--  dancetrack00XX-N-5
|	     |--  dancetrack00XX-N-10
|-- SportsMOT/
|	|--  train/
|	|--  val/
|	|--  test/
|-- MOT20
|	|--  train/
|	|--  test/
|-- Visdrone-MOT
	|--  train
	|--  val
	|--  test-dev
```

# 3. Data Preparation

This section covers how to prepare the datasets for training and evaluation.

**3.1 Split MOT17 Dataset (train_half / val_half)**

For training and validation on MOT17, you can split the official training set into two halves (train_half and val_half). Use the ```tool/get_mot17_valhalf.py ``` script to automatically perform this split. This script will create a val directory and move the second half of each sequence from the train directory into it, re-indexing the frames and IDs accordingly.

**Usage:**

```
change the dataset ROOT_PATH

python tool/get_mot17_valhalf.py
```

**3.2 Generate Low-Frame-Rate Datasets (MOT17-lfr & DanceTrack-lfr)**

To evaluate the tracker's performance in Low-Frame-Rate scenarios, you first need to generate the corresponding datasets from the original high-frame-rate videos. Use the provided ```tool/get_low_frame_rate_data.py``` script. This script will sample frames at a given interval (--interval) and create new sequence folders with an -N-n suffix.

**Example for MOT17 (frame gap = 2):**

```
python tool/get_low_frame_rate_dataset.py  --root_dir /path/to/your/datasets/mot/val/  --interval 2  --dataset mot17
```

**Example for DanceTrack（frame gap =10）:**

```
python tool/get_low_frame_rate_dataset.py  --root_dir /path/to/your/datasets/DanceTrack/val/  --interval 10  --dataset dancetrack
```

# 4. Model Preparation

**4.1 ReID Model**

This project uses the `sbs-s50` model from the **fast-reid** library to extract appearance features. You can download the pre-trained weights from the official repository or other sources.

​**Download Link**
fast_reid model zoo can download at [here.](https://github.com/Kroery/DiffMOT)

Place the downloaded model weights in `/pretrained` directory and change the model name as `mot17_sbs_S50.pth`and`dancetrack_sbs_S50.pth`.

**4.2 Evaluation Model**

We provide a pre-trained evaluation model specifically for validating tracking performance on Low-Frame-Rate videos.

* ​**Location**​: The model is located in the `/models` directory.
* You can directly use this model with the `--model_path` argument in your evaluation commands.


# 5. Low-Frame-Rate Evaluation

To evaluate the tracker on the generated Low-Frame-Rate datasets, run the main script in `test` mode. Make sure to point the `--data_path` to the directory containing the Low-Frame-Rate sequences (e.g., `dancetrack/val_lfr`).

**5.1 Evaluate MOT17 in Low-Frame-Rate**

```
python GLoMOT_main.py --experiment_mode test --cuda --test_splits mot17-val-f2 --use_node_buffer True --buffer_len 20 --det_file byte065 --data_path "E:/python/GLoMOT/datasets/" --reid_embeddings_dir reid_sbs_s50 --node_embeddings_dir node_sbs_s50 --reid_arch fastreid_mot17_sbs_S50
```

**5.2 Evaluate DanceTrack in Low-Frame-Rate**

```
python GLoMOT_main.py --experiment_mode test --cuda --test_splits dancetrack-val-f2 --use_node_buffer True --buffer_len 10 --det_file byte065 --data_path "E:/python/GLoMOT/datasets/" --reid_embeddings_dir reid_sbs_s50 --node_embeddings_dir node_sbs_s50 --reid_arch fastreid_dancetrack_sbs_S50 --use_smpnet True
```

We use [TrackEval ](https://github.com/JonathonLuiten/TrackEval)to evaluate the results.

# 6. Benchmark Evaluation

* **6.1 Tracking MOT17 test set**

```
python GLoMOT_main.py --experiment_mode test --cuda --test_splits mot17-test --use_node_buffer True --buffer_len 30 --det_file byte065 --data_path "E:/python/GLoMOT/datasets/" --reid_embeddings_dir reid_sbs_s50 --node_embeddings_dir node_sbs_s50 --reid_arch fastreid_mot17_sbs_S50  --use_smpnet True
```

* **6.2 Tracking DanceTrack test set**

```
python GLoMOT_main.py --experiment_mode test --cuda --test_splits dancetrack-test --use_node_buffer True --buffer_len 20 --det_file byte065 --data_path "E:/python/GLoMOT/datasets/" --reid_embeddings_dir reid_sbs_s50 --node_embeddings_dir node_sbs_s50 --reid_arch fastreid_dancetrack_sbs_S50 --use_smpnet True
```

* **6.3 Tracking Visdrone test-dev set**

```
python GLoMOT_main.py --experiment_mode test --cuda --test_splits visdrone-test --use_node_buffer True --buffer_len 60 --det_file byte065 --data_path "E:/python/GLoMOT/datasets/" --reid_embeddings_dir reid_sbs_s50 --node_embeddings_dir node_sbs_s50 --reid_arch fastreid_visdrone_sbs_S50 --use_smpnet True
```

# 7. Model Training

```
python GLoMOT_main.py --experiment_mode train --cuda --train_splits dancetrack-train --val_splits dancetrack val --det_file byte065 --data_path "E:/python/GLoMOT/datasets/" --reid_embeddings_dir reid_sbs_s50 --node_embeddings_dir node_sbs_s50 --reid_arch fastreid_dancetrack_sbs_S50 --use_smpnet True
```

# 8. Acknowledgement

A large part of the code is borrowed from [SUSHI](https://github.com/dvl-tum/SUSHI) and we compared the Low-Frame-Rate tracking results with [DiffMOT](https://github.com/Kroery/DiffMOT). Thanks for their wonderful works!
