import random
import torch
from src.data.seq_processor import MOTSeqProcessor
from src.data.augmentation import GraphAugmentor
import numpy as np
import os.path as osp
from src.data.graph import HierarchicalGraph
from collections import OrderedDict
from torch.nn import functional as F
import pandas as pd

class MOTSceneDataset:
    """
    Main dataset class
    """
    def __init__(self, config, seqs, mode):
        assert mode in ('train', 'val', 'test'), "Dataset mode is not valid!"
        self.config = config
        self.seqs = seqs
        self.mode = mode

        # Load all dataframes
        self.seq_det_dfs, self.seq_info_dicts, self.seq_names = self._load_seq_dfs()

        # Caches for pre-loaded embeddings
        self.reid_embedding_cache = {}
        self.node_embedding_cache = {}

        # Index dataset
        if mode == "train":
            self.seq_and_frames = self._index_dataset(shuffle=True)
        elif mode == "val":
            self.seq_and_frames = self._index_dataset(shuffle=False)
        elif mode =="test":
            self.seq_and_frames = self._index_dataset_test()
        self.sparse_frames_per_seq=self.seq_and_frames
        
    def _load_seq_dfs(self):
        """
        Load the dataframes of the sequences to be used
        """

        # Initialize empty vars
        seq_names, seq_info_dicts, seq_det_dfs = [], {}, {}

        # Loop over the seqs to retrieve
        for dataset_path, seq_list in self.seqs.items():
            for seq_name in seq_list:
                # Process or load the sequence df
                seq_processor = MOTSeqProcessor(dataset_path=dataset_path, seq_name=seq_name, config=self.config)
                seq_det_df = seq_processor.load_or_process_detections()

                # Accumulate
                seq_names.append(seq_name)
                seq_info_dicts[seq_name] = seq_det_df.seq_info_dict
                seq_det_dfs[seq_name] = seq_det_df

        assert len(seq_det_dfs) and len(seq_info_dicts) and len(seq_det_dfs), "No detections to process in the dataset"
        return seq_det_dfs, seq_info_dicts, seq_names

    def preload_embeddings_for_seq(self, seq_name):
        """
        Loads all embeddings for a given sequence into an in-memory cache.
        This should be called once before starting the tracking loop for a sequence.
        """
        print(f"Pre-loading embeddings for sequence: {seq_name}...")
        seq_info_dict = self.seq_info_dicts[seq_name]
        det_df = self.seq_det_dfs[seq_name]

        # Load all embeddings from disk for the entire sequence
        reid_embeddings_full = self._load_precomputed_embeddings(det_df, seq_info_dict, self.config.reid_embeddings_dir)
        node_embeddings_full = self._load_precomputed_embeddings(det_df, seq_info_dict, self.config.node_embeddings_dir)

        # Create a dictionary mapping detection_id to its embedding
        # This is faster than filtering a large tensor at each step
        self.reid_embedding_cache[seq_name] = {int(det_id): emb for det_id, *emb in reid_embeddings_full.tolist()}
        self.node_embedding_cache[seq_name] = {int(det_id): emb for det_id, *emb in node_embeddings_full.tolist()}
        print(f"Finished pre-loading {len(self.reid_embedding_cache[seq_name])} embeddings for {seq_name}.")

    def _get_embeddings_from_cache(self, det_df, seq_name):
        """
        Retrieves pre-loaded embeddings from the cache for the given detections.
        """
        # Ensure the cache is populated for the sequence
        if seq_name not in self.reid_embedding_cache:
            self.preload_embeddings_for_seq(seq_name)

        # Retrieve embeddings using the detection_ids
        det_ids = det_df['detection_id'].values

        # Fast retrieval from dictionary
        x_reid_list = [self.reid_embedding_cache[seq_name][det_id] for det_id in det_ids]
        x_node_list = [self.node_embedding_cache[seq_name][det_id] for det_id in det_ids]

        # Convert to tensor
        x_reid = torch.tensor(x_reid_list, dtype=torch.float32)
        x_node = torch.tensor(x_node_list, dtype=torch.float32)

        return x_reid, x_node


    def _index_dataset_test(self):
        """
        Index the dataset in a form that we can sample
        """
        seq_and_frames = {}
        dataset=self.config.dataset
        dataset_type=self.config.dataset_type
        # Loop over the scenes
        for scene in self.seq_names:
            # Get scene specific dataframe
            scene_df = self.seq_det_dfs[scene]
            frames_per_graph = self.config.frames_per_graph
            # Scene specific values
            frames = list(OrderedDict.fromkeys(scene_df['frame'].to_list()))
            start_frames = []
            end_frames = []
            seq_frames = []
            
            for f in frames:
                if not start_frames or f >= start_frames[-1] + self.config.train_dataset_frame_overlap:
                    valid_frames = np.arange(f, f + frames_per_graph)
                    graph_df = scene_df[scene_df.frame.isin(valid_frames)].copy()   
                    
                    if (graph_df.frame.min() not in start_frames) and (graph_df.frame.max() not in end_frames) and (
                            len(graph_df.frame.unique()) >= 2):
                        seq_frames.append((scene, graph_df.frame.min(), graph_df.frame.max()))
                        start_frames.append(graph_df.frame.min())
                        end_frames.append(graph_df.frame.max())
            
            seq_and_frames[scene]=tuple(seq_frames)
        return seq_and_frames

        

    def _index_dataset(self,shuffle=None):
        """
        Index the dataset in a form that we can sample
        """
        seq_and_frames = []
        order_percent=[]
        # Loop over the scenes
        for scene in self.seq_names:
            seq_frames=[]
            # Get scene specific dataframe
            scene_df = self.seq_det_dfs[scene]
            frames_per_graph = self.config.frames_per_graph

            #####
            frames_per_graph_plus = frames_per_graph+2

            # Scene specific values
            frames = list(OrderedDict.fromkeys(scene_df['frame'].to_list()))
            start_frames = []
            end_frames = []

            #Loop over all frames
            for f in frames:
                if not start_frames or f >= start_frames[-1] + self.config.train_dataset_frame_overlap:
                    valid_frames = np.arange(f, f + frames_per_graph)
                    graph_df = scene_df[scene_df.frame.isin(valid_frames)].copy()
                    if (graph_df.frame.min() not in start_frames) and (graph_df.frame.max() not in end_frames) and (
                            len(graph_df.frame.unique()) >= 2):
                        seq_and_frames.append((scene, graph_df.frame.min(), graph_df.frame.max()))
                        start_frames.append(graph_df.frame.min())
                        end_frames.append(graph_df.frame.max())

                        if graph_df.frame.max()<=frames[-1]*self.config.train_data_percent:
                            order_percent.append((scene, graph_df.frame.min(), graph_df.frame.max()))

            start_frames = []
            end_frames = []
            for f in frames:
                if not start_frames or f >= start_frames[-1] + self.config.train_dataset_frame_overlap:
                    valid_frames = np.arange(f, f + frames_per_graph_plus)
                    graph_df = scene_df[scene_df.frame.isin(valid_frames)].copy()
                    if (graph_df.frame.min() not in start_frames) and (graph_df.frame.max() not in end_frames) and (
                            len(graph_df.frame.unique()) >= 2):
                        seq_and_frames.append((scene, graph_df.frame.min(), graph_df.frame.max()))
                        start_frames.append(graph_df.frame.min())
                        end_frames.append(graph_df.frame.max())

                        if graph_df.frame.max() <= frames[-1] * self.config.train_data_percent:
                            order_percent.append((scene, graph_df.frame.min(), graph_df.frame.max()))

        if shuffle:
            random.shuffle(order_percent)

        #####
        return tuple(order_percent)

    def _sparse_index_dataset(self):
        """
        Overlapping samples used for validation and test. This time we create a dictionary and bookkeep the sequence name
        """
        sparse_frames_per_seq = {}
        frames_per_graph = self.config.frames_per_graph
        overlap_ratio = self.config.evaluation_graph_overlap_ratio
        for scene in self.seq_names:
            scene_df = self.seq_det_dfs[scene]
            sparse_frames = []

            # Scene specific values
            frames = list(OrderedDict.fromkeys(scene_df['frame'].to_list()))
            start_frames = []
            end_frames = []

            min_frame = scene_df.frame.min()  # Initializer

            # Continue until all frames are processed
            while len(frames):
                # Valid regions of the df
                valid_frames = np.arange(min_frame, min_frame + frames_per_graph)
                graph_df = scene_df[scene_df.frame.isin(valid_frames)].copy()

                if (graph_df.frame.min() not in start_frames) and (graph_df.frame.max() not in end_frames) and (
                        len(graph_df.frame.unique()) >= 2):
                    # Include the sample
                    sparse_frames.append((scene, graph_df.frame.min(), graph_df.frame.max()))

                    # Update start and end frames
                    start_frames.append(graph_df.frame.min())
                    end_frames.append(graph_df.frame.max())

                    # Update the min frame
                    current_frames = sorted(list(graph_df.frame.unique()))
                    num_current_frame = len(current_frames)
                    num_overlaps = round(overlap_ratio * num_current_frame)
                    assert num_overlaps < num_current_frame and num_overlaps > 0, "Evaluation overlap ratio leads to either all frames or no frames"
                    min_frame = current_frames[-num_overlaps]

                    # Remove current frames from the remaining frames list
                    frames = [f for f in frames if f not in current_frames]

                else:
                    current_frames = sorted(list(graph_df.frame.unique()))
                    frames = [f for f in frames if f not in current_frames]
                    min_frame = min(frames)

            # To prevent empty lists
            if sparse_frames:

                # Accumulate sparse_frames_per_seq
                sparse_frames_per_seq[scene] = tuple(sparse_frames)

        return sparse_frames_per_seq

    def _load_precomputed_embeddings(self, det_df, seq_info_dict, embeddings_dir):
        """
        Load the embeddings corresponding to the detections specified in the det_df
        """
        # Retrieve the embeddings we need from their corresponding locations
        embeddings_path = osp.join(seq_info_dict['seq_path'], 'processed_data', 'embeddings',
                                   seq_info_dict['det_file_name'],
                                   embeddings_dir)
        # print("EMBEDDINGS PATH IS ", embeddings_path)
        frames_to_retrieve = sorted(det_df.frame.unique())
        embeddings_list = [torch.load(osp.join(embeddings_path, f"{frame_num}.pt")) for frame_num in frames_to_retrieve]
        embeddings = torch.cat(embeddings_list, dim=0)

        # First column in embeddings is the index. Drop the rows of those that are not present in det_df
        ixs_to_drop = list(set(embeddings[:, 0].int().numpy()) - set(det_df['detection_id']))
        embeddings = embeddings[~np.isin(embeddings[:, 0], ixs_to_drop)]  # Not so clean, but faster than a join
        assert_str = "Problems loading embeddings. Indices between query and stored embeddings do not match. BOTH SHOULD BE SORTED!"
        assert (embeddings[:, 0].numpy() == det_df['detection_id'].values).all(), assert_str

        #embeddings = embeddings[:, 1:]  # Get rid of the detection index (MOVED TO OUT OF THIS FUNCTION)

        return embeddings

    def _load_buffer_precomputed_embeddings(self, node_buffer_pool, seq_info_dict, embeddings_dir):
        """
        Load the embeddings corresponding to the detections specified in the det_df
        """
        # Retrieve the embeddings we need from their corresponding locations
        embeddings_path = osp.join(seq_info_dict['seq_path'], 'processed_data', 'embeddings',
                                   seq_info_dict['det_file_name'],
                                   embeddings_dir)
        # print("EMBEDDINGS PATH IS ", embeddings_path)

        node_buffer_pool=sorted(node_buffer_pool, key=lambda x: (x['frame'], x['detection_id']))
        embedding_list=[]
        for node in node_buffer_pool:
            frame=node['frame']
            emb_path=osp.join(embeddings_path, f"{frame}.pt")
            emb_temp=torch.load(emb_path)
            emb_index=node['index_in_missed_frame']
            embedding_list.append(emb_temp[emb_index])
        res_emb=torch.stack(embedding_list, dim=0)
        return res_emb

    def _load_buffer_precomputed_embeddings_in_track(self, node_buffer_pool, seq_info_dict, embeddings_dir):
        """
        Load the embeddings corresponding to the detections specified in the det_df
        """
        embeddings_path = osp.join(seq_info_dict['seq_path'], 'processed_data', 'embeddings',
                                   seq_info_dict['det_file_name'], embeddings_dir)

        # 按 frame 和 detection_id 排序
        node_buffer_pool = sorted(node_buffer_pool,
                                  key=lambda track: (track.buffer_node['frame'], track.buffer_node['detection_id']))
        embedding_list = []
        for track in node_buffer_pool:
            frame = track.buffer_node['frame']
            emb_path = osp.join(embeddings_path, f"{frame}.pt")
            emb_temp = torch.load(emb_path)
            emb_index = track.buffer_node['index_in_missed_frame']
            embedding_list.append(emb_temp[emb_index])

        res_emb = torch.stack(embedding_list, dim=0)
        return res_emb



    def get_df_from_seq_and_frames(self, seq_name, start_frame, end_frame):
        """
        Returns a dataframe and a seq_info_dict belonging to the specified sequence range
        """
        # Load the corresponding part of the dataframe
        seq_det_df = self.seq_det_dfs[seq_name]  # Sequence specific dets
        seq_info_dict = self.seq_info_dicts[seq_name]  # Sequence info dict
        ########
        valid_frames = [start_frame, end_frame]
        #######
        #valid_frames = np.arange(start_frame, end_frame + 1)  # Frames to be processed together
        graph_df = seq_det_df[seq_det_df.frame.isin(valid_frames)].copy()  # Take only valid frames
        graph_df = graph_df.sort_values(by=['frame', 'detection_id']).reset_index(drop=True)

        return graph_df, seq_info_dict

    def add_buffer_pool_node_to_graph(self,graph_df, node_buffer_pool):
        node_buffer_pool=sorted(node_buffer_pool, key=lambda x: (x['frame'], x['detection_id']))
        for buf_node in node_buffer_pool:
            new_row = {
                    'frame_path': buf_node['frame_path'],
                    'frame': graph_df['frame'][0], #
                    'ped_id': buf_node['ped_id'],  #
                    'bb_left': buf_node['bb_left'],
                    'bb_top': buf_node['bb_top'],
                    'bb_width': buf_node['bb_width'],
                    'bb_height': buf_node['bb_height'],
                    'bb_right': buf_node.get('bb_right', 0.0),
                    'conf': buf_node['conf'],
                    'detection_id': buf_node['detection_id'],
                    'feet_x': (buf_node['bb_left']+buf_node['bb_right'])/2,
                    'feet_y': buf_node['bb_top']+buf_node['bb_height'],
                    'id': buf_node.get('ped_id', 0)
                }

            graph_df = pd.concat([pd.DataFrame([new_row]),graph_df], ignore_index=True)

        graph_df.drop_duplicates(subset=['frame', 'detection_id'], keep='last', inplace=True)

        graph_df = graph_df.groupby('frame', as_index=False, group_keys=False).apply(
            lambda x: x.sort_values(by='detection_id', ascending=True))
        return graph_df

    def add_buffer_pool_node_to_graph_in_track(self,graph_df, node_buffer_pool):

        node_buffer_pool = sorted(node_buffer_pool,
                                  key=lambda track: (track.buffer_node['frame'], track.buffer_node['detection_id']))
        for track in node_buffer_pool:
            buf_node = track.buffer_node
            new_row = {
                'frame_path': buf_node.get('frame_path', ''),
                'frame': graph_df['frame'][0],  #
                'ped_id': buf_node.get('ped_id', -1),  #
                'bb_left': buf_node['bb_left'],
                'bb_top': buf_node['bb_top'],
                'bb_width': buf_node['bb_width'],
                'bb_height': buf_node['bb_height'],
                'bb_right': buf_node.get('bb_right', buf_node['bb_left'] + buf_node['bb_width']),
                'conf': buf_node.get('conf', 1.0),
                'detection_id': buf_node['detection_id'],
                'feet_x': (buf_node['bb_left'] + buf_node.get('bb_right',
                                                              buf_node['bb_left'] + buf_node['bb_width'])) / 2,
                'feet_y': buf_node['bb_top'] + buf_node['bb_height'],
                'id': buf_node.get('ped_id', -1)  #
            }

            graph_df = pd.concat([pd.DataFrame([new_row]), graph_df], ignore_index=True)


        graph_df.drop_duplicates(subset=['frame', 'detection_id'], keep='last', inplace=True)
        graph_df = graph_df.groupby('frame', as_index=False, group_keys=False).apply(
            lambda x: x.sort_values(by='detection_id', ascending=True))
        return graph_df

    def get_graph_from_seq_and_frames(self, seq_name, start_frame, end_frame):
        """
        Main dataloading function. Returns a hierarchical graph belonging to the specified sequence range
        """
        graph_df, seq_info_dict = self.get_df_from_seq_and_frames(seq_name=seq_name, start_frame=start_frame, end_frame=end_frame)

        # Ensure that there are at least 2 frames in the sampled graph
        assert len(graph_df['frame'].unique()) > 1, "There aren't enough frames in the sampled graph. Either 0 or 1"

        if self.mode=='train' and self.config.augmentation:
            augmentor = GraphAugmentor(graph_df=graph_df, config=self.config)
            graph_df = augmentor.augment()

        # Load appearance data
        x_reid = self._load_precomputed_embeddings(det_df=graph_df, seq_info_dict=seq_info_dict,
                                                   embeddings_dir=self.config.reid_embeddings_dir)
        
        x_node = self._load_precomputed_embeddings(det_df=graph_df, seq_info_dict=seq_info_dict,
                                                    embeddings_dir=self.config.node_embeddings_dir)


        # Copy node frames and ground truth ids from the dataframe
        x_frame = torch.tensor(graph_df[['detection_id', 'frame']].values)
        x_bbox = torch.tensor(graph_df[['detection_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height']].values)
        x_feet = torch.tensor(graph_df[['detection_id', 'feet_x', 'feet_y']].values)
        y_id = torch.tensor(graph_df[['detection_id', 'id']].values)

        # Assert that order of all the loaded values are the same
        assert (x_reid[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_node[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_frame[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_bbox[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_feet[:, 0].numpy() == y_id[:, 0].numpy()).all(), "Feature and id mismatch while loading"

        # Get rid of the detection id index
        x_reid = x_reid[:, 1:]
        x_node = x_node[:, 1:]
        x_frame = x_frame[:, 1:]
        x_bbox = x_bbox[:, 1:]
        x_center = x_bbox[:, :2] + 0.5* x_bbox[:, 2:]
        x_feet = x_feet[:, 1:]
        y_id = y_id[:, 1:]

        if self.config.l2_norm_reid:
            x_reid = F.normalize(x_reid, dim = -1, p=2)
            x_node = F.normalize(x_node, dim = -1, p=2)

        fps = torch.tensor(seq_info_dict['fps'])
        frames_total = torch.tensor(self.config.frames_per_graph)
        frames_per_level = torch.tensor(self.config.frames_per_level)
        start_frame = torch.tensor(start_frame)
        end_frame = torch.tensor(end_frame)

        # Create the object with float32 and int64 precision and send to the device
        hierarchical_graph = HierarchicalGraph(x_reid=x_reid.float(), x_node=x_node.float(), x_frame=x_frame.long(),
                                               x_bbox=x_bbox.float(), x_feet=x_feet.float(), x_center=x_center.float(), 
                                               y_id=y_id.long(), fps=fps.long(), frames_total=frames_total.long(),
                                               frames_per_level=frames_per_level.long(), 
                                               start_frame=start_frame.long(), end_frame=end_frame.long())

        return hierarchical_graph

    def calculate_pseudo_depth_for_graph_df(self, graph_df, image_size):

        depths_list = []
        for frame, frame_df in graph_df.groupby('frame'):
            detections = []
            det_ids = []

            for _, row in frame_df.iterrows():
                detection = (row['bb_left'], row['bb_top'], row['bb_width'], row['bb_height'])
                detections.append(detection)
                det_ids.append(row['detection_id'])

            if not detections:
                continue

            pseudo_depths = self.calculate_pseudo_depth(image_size, detections)
            for det_id, depth in zip(det_ids, pseudo_depths):
                depths_list.append([det_id, depth])
        if depths_list:
            x_depth = torch.tensor(depths_list)
        else:
            x_depth = torch.zeros((0, 2))

        return x_depth


    def calculate_pseudo_depth(self,image_size, detections_list):

        if not detections_list:
            return torch.empty(0)


        detections = torch.tensor(detections_list, dtype=torch.float32)
        x1, y1, w, h = detections.unbind(dim=1)
        cx = x1 + w / 2
        cy = y1 + h / 2
        areas = w * h


        min_area, max_area = torch.min(areas), torch.max(areas)
        size_depth = (areas - min_area) / (max_area - min_area + 1e-6)


        img_height = image_size[1]
        position_depth = 1 - (cy / img_height)


        x2 = x1 + w
        y2 = y1 + h
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        inter_top_left = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
        inter_bottom_right = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])

        inter_wh = (inter_bottom_right - inter_top_left).clamp(min=0)
        inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]


        torch.diagonal(inter_area).fill_(0)


        total_occlusion_area = torch.sum(inter_area, dim=1)
        occlusion_ratio = total_occlusion_area / (areas + 1e-6)
        occlusion_depth = 1 / (1 + occlusion_ratio)


        weight_size = 0.2
        weight_position = 0.4
        weight_occlusion = 0.4

        pseudo_depths = (weight_size * size_depth +
                         weight_position * position_depth +
                         weight_occlusion * occlusion_depth)

        return pseudo_depths


    def get_graph_from_seq_and_frames_with_buffer(self, seq_name, start_frame, end_frame, node_buffer_pool=None):
        """
        Main dataloading function. Returns a hierarchical graph belonging to the specified sequence range
        """
        graph_df, seq_info_dict = self.get_df_from_seq_and_frames(seq_name=seq_name, start_frame=start_frame,
                                                                  end_frame=end_frame)
        #
        # Ensure that there are at least 2 frames in the sampled graph
        assert len(graph_df['frame'].unique()) > 1, "There aren't enough frames in the sampled graph. Either 0 or 1"

        if self.mode == 'train' and self.config.augmentation:
            augmentor = GraphAugmentor(graph_df=graph_df, config=self.config)
            graph_df = augmentor.augment()

        # Load appearance data
        x_reid = self._load_precomputed_embeddings(det_df=graph_df, seq_info_dict=seq_info_dict,
                                                   embeddings_dir=self.config.reid_embeddings_dir)

        x_node = self._load_precomputed_embeddings(det_df=graph_df, seq_info_dict=seq_info_dict,
                                                   embeddings_dir=self.config.node_embeddings_dir)

        ori_graph_df=graph_df.copy()
        if node_buffer_pool is not None and len(node_buffer_pool) > 0:
            graph_df = self.add_buffer_pool_node_to_graph(graph_df, node_buffer_pool)
            buffer_reid= self._load_buffer_precomputed_embeddings(node_buffer_pool=node_buffer_pool, seq_info_dict=seq_info_dict,
                                                   embeddings_dir=self.config.reid_embeddings_dir)
            buffer_node=self._load_buffer_precomputed_embeddings(node_buffer_pool=node_buffer_pool, seq_info_dict=seq_info_dict,
                                                   embeddings_dir=self.config.node_embeddings_dir)
            x_reid=torch.concat([buffer_reid,x_reid],dim=0)
            x_node=torch.concat([buffer_node,x_node],dim=0)

            graph_df_with_buffer=graph_df.copy()

        # Copy node frames and ground truth ids from the dataframe
        x_frame = torch.tensor(graph_df[['detection_id', 'frame']].values)
        x_bbox = torch.tensor(graph_df[['detection_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height']].values)
        x_feet = torch.tensor(graph_df[['detection_id', 'feet_x', 'feet_y']].values)
        y_id = torch.tensor(graph_df[['detection_id', 'id']].values)


            ##########
        x_depth=self.calculate_pseudo_depth_for_graph_df(graph_df,(seq_info_dict['frame_width'],seq_info_dict['frame_height']))
        x_depth = x_depth[:, 1:]
            ##########

        # Assert that order of all the loaded values are the same
        assert (x_reid[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_node[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_frame[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_bbox[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_feet[:, 0].numpy() == y_id[:, 0].numpy()).all(), "Feature and id mismatch while loading"

        x_reid = x_reid[:, 1:]
        x_node = x_node[:, 1:]
        x_frame = x_frame[:, 1:]
        x_bbox = x_bbox[:, 1:]
        x_center = x_bbox[:, :2] + 0.5 * x_bbox[:, 2:]
        x_feet = x_feet[:, 1:]
        y_id = y_id[:, 1:]



        if self.config.l2_norm_reid:
            x_reid = F.normalize(x_reid, dim=-1, p=2)
            x_node = F.normalize(x_node, dim=-1, p=2)

        fps = torch.tensor(seq_info_dict['fps'])
        frames_total = torch.tensor(self.config.frames_per_graph)
        frames_per_level = torch.tensor(self.config.frames_per_level)
        start_frame = torch.tensor(start_frame)
        end_frame = torch.tensor(end_frame)


            # Create the object with float32 and int64 precision and send to the device
        hierarchical_graph = HierarchicalGraph(x_reid=x_reid.float(), x_node=x_node.float(), x_frame=x_frame.long(),
                                               x_bbox=x_bbox.float(), x_feet=x_feet.float(), x_center=x_center.float(),
                                               y_id=y_id.long(), fps=fps.long(), frames_total=frames_total.long(),
                                               frames_per_level=frames_per_level.long(),
                                               start_frame=start_frame.long(), end_frame=end_frame.long(),x_depth=x_depth.float())


        if node_buffer_pool is not None and len(node_buffer_pool) > 0:
            return hierarchical_graph,ori_graph_df,graph_df_with_buffer
        else:
            return hierarchical_graph,ori_graph_df,None

    def get_graph_from_seq_and_frames_with_buffer_in_track(self, seq_name, start_frame, end_frame, node_buffer_pool=None):
        """
        Main dataloading function. Returns a hierarchical graph belonging to the specified sequence range
        """
        graph_df, seq_info_dict = self.get_df_from_seq_and_frames(seq_name=seq_name, start_frame=start_frame,
                                                                  end_frame=end_frame)

        # Ensure that there are at least 2 frames in the sampled graph
        assert len(graph_df['frame'].unique()) > 1, "There aren't enough frames in the sampled graph. Either 0 or 1"

        if self.mode == 'train' and self.config.augmentation:
            augmentor = GraphAugmentor(graph_df=graph_df, config=self.config)
            graph_df = augmentor.augment()


        x_reid, x_node = self._get_embeddings_from_cache(graph_df, seq_name)


        ori_graph_df=graph_df.copy()
        if node_buffer_pool is not None and len(node_buffer_pool) > 0:
            graph_df = self.add_buffer_pool_node_to_graph_in_track(graph_df, node_buffer_pool)
            buffer_reid= self._load_buffer_precomputed_embeddings_in_track(node_buffer_pool=node_buffer_pool, seq_info_dict=seq_info_dict,
                                                   embeddings_dir=self.config.reid_embeddings_dir)
            buffer_node=self._load_buffer_precomputed_embeddings_in_track(node_buffer_pool=node_buffer_pool, seq_info_dict=seq_info_dict,
                                                   embeddings_dir=self.config.node_embeddings_dir)
            x_reid=torch.concat([buffer_reid,x_reid],dim=0)
            x_node=torch.concat([buffer_node,x_node],dim=0)

            graph_df_with_buffer=graph_df.copy()

        # Copy node frames and ground truth ids from the dataframe
        x_frame = torch.tensor(graph_df[['detection_id', 'frame']].values)
        x_bbox = torch.tensor(graph_df[['detection_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height']].values)
        x_feet = torch.tensor(graph_df[['detection_id', 'feet_x', 'feet_y']].values)
        y_id = torch.tensor(graph_df[['detection_id', 'id']].values)
        x_depth=self.calculate_pseudo_depth_for_graph_df(graph_df,(seq_info_dict['frame_width'],seq_info_dict['frame_height']))
        x_depth = x_depth[:, 1:]
        assert (x_reid[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_node[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_frame[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_bbox[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_feet[:, 0].numpy() == y_id[:, 0].numpy()).all(), "Feature and id mismatch while loading"

        x_reid = x_reid[:, 1:]
        x_node = x_node[:, 1:]
        x_frame = x_frame[:, 1:]
        x_bbox = x_bbox[:, 1:]
        x_center = x_bbox[:, :2] + 0.5 * x_bbox[:, 2:]
        x_feet = x_feet[:, 1:]
        y_id = y_id[:, 1:]

        if self.config.l2_norm_reid:
            x_reid = F.normalize(x_reid, dim=-1, p=2)
            x_node = F.normalize(x_node, dim=-1, p=2)


        fps = torch.tensor(seq_info_dict['fps'])
        frames_total = torch.tensor(self.config.frames_per_graph)
        frames_per_level = torch.tensor(self.config.frames_per_level)
        start_frame = torch.tensor(start_frame)
        end_frame = torch.tensor(end_frame)


            # Create the object with float32 and int64 precision and send to the device
        hierarchical_graph = HierarchicalGraph(x_reid=x_reid.float(), x_node=x_node.float(), x_frame=x_frame.long(),
                                               x_bbox=x_bbox.float(), x_feet=x_feet.float(), x_center=x_center.float(),
                                               y_id=y_id.long(), fps=fps.long(), frames_total=frames_total.long(),
                                               frames_per_level=frames_per_level.long(),
                                               start_frame=start_frame.long(), end_frame=end_frame.long(),x_depth=x_depth.float())


        if node_buffer_pool is not None and len(node_buffer_pool) > 0:
            return hierarchical_graph,ori_graph_df,graph_df_with_buffer
        else:
            return hierarchical_graph,ori_graph_df,None

    def __len__(self):
        return len(self.seq_and_frames)

    def __getitem__(self, ix):
        seq_name, start_frame, end_frame = self.seq_and_frames[ix]
        return self.get_graph_from_seq_and_frames(seq_name=seq_name, start_frame=start_frame, end_frame=end_frame)