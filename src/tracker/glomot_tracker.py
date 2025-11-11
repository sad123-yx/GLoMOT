from pickletools import optimize
import yaml
import torch.optim as optim
from torch_geometric.data import Batch
from src.models.mpntrack import MOTMPNet
from src.models.glomot_net import GLOMOT_Net
from src.models.motion.linear import LinearMotionModel
from src.utils.deterministic import seed_worker, seed_generator
from src.data.mot_datasets import MOTSceneDataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch
from src.utils.graph_utils import to_undirected_graph, to_lightweight_graph, to_positive_decision_graph
from src.utils.motion_utils import compute_giou_fwrd_bwrd_motion_sim
from src.tracker.projectors import GreedyProjector, ExactProjector
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np
import pandas as pd
from src.utils.lapsolver import solve_dense
import os.path as osp
from src.tracker.postprocessing import Postprocessor
import time
import statistics
import os
from TrackEval.scripts.run_mot_challenge import evaluate_mot17
import matplotlib.pyplot as plt
from torch import nn
import math
import pickle
from torch.utils.tensorboard.writer import SummaryWriter
from src.data.mot_datasets import MOTSceneDataset
from datetime import timedelta
from src.models.strong_mpntrack import SMOTMPNet

class GLoMOT_Tracker:
    def __init__(self, config, seqs, splits):
        self.config = config
        self.seqs = seqs
        self.train_split, self.val_split, self.test_split = splits

        self.model = self._get_model()

        gpu_num = len(self.config.gpus)
        if gpu_num > 1:
            self.model = torch.nn.DataParallel(self.model, self.config.gpus).to('cuda')
        else:
            self.model = self.model.to('cuda')

        if self.config.do_motion:
            self.motion_model = LinearMotionModel()
        
        else:
            self.motion_model = None

        # Training - Set up the dataset and optimizer
        if self.config.experiment_mode in ('train', 'train-cval'):
            self.loss_function = FocalLoss(logits=True, gamma=self.config.gamma)
            self.train_dataloader = self._get_train_dataloader()
            self.optimizer = self._get_optimizer()
            # Get validation dataset if exists
            if self.val_split:
                self.val_dataset = self._get_dataset(mode='val')

        # Testing - Set up the dataset
        elif self.config.experiment_mode == 'test':
            self.test_dataset = self._get_dataset(mode='test')

        # Iteration and epoch
        self.train_iteration = 0
        self.train_epoch = 0

        # Layers that are allowed to be trained
        if self.config.depth_pretrain_iteration == 0:
            self.active_train_depth = self.config.hicl_depth
        else:
            self.active_train_depth = min(1, self.config.hicl_depth)

        #if self.config.tensorboard:
        if self.config.experiment_mode == 'train':
            self.logger = SummaryWriter(osp.join(self.config.experiment_path, 'tf_logs'))

        self.node_buffer_pool=[]
        self.history_positions = {}
        self.use_buffer_motion = self.config.use_buffer_motion
        self.motion_decay_factor = self.config.motion_decay_factor

    def update_history(self, data):

        for _, row in data.iterrows():
            ped_id = row['ped_id']
            frame = row['frame']
            feet_x = row['feet_x']
            feet_y = row['feet_y']
            bbox = [row['bb_left'], row['bb_top'], row['bb_right'], row['bb_bot']]

            if ped_id not in self.history_positions:
                self.history_positions[ped_id] = []

            frame_exists = any(pos['frame'] == frame for pos in self.history_positions[ped_id])

            if not frame_exists:
                self.history_positions[ped_id].append({
                    'frame': frame,
                    'feet': (feet_x, feet_y),
                    'bbox': bbox
                })


                if len(self.history_positions[ped_id]) > 5:
                    self.history_positions[ped_id].pop(0)

    def _get_model(self):
        """
        Load the hierarchical model
        """
        model_yaml="E:/python/GLOMOT/configs/mpntrack_cfg.yaml"

        with open(model_yaml) as file:
            mpntrack_params = yaml.load(file, Loader=yaml.FullLoader)


        mpntrack_params['graph_model_params']['encoder_feats_dict']['node_in_dim'] = self.config.node_dim


        mpntrack_params['graph_model_params']['do_hicl_feats']=self.config.do_hicl_feats
        mpntrack_params['graph_model_params']['hicl_feats_encoder'].update(self.config.hicl_feats_args)

        if self.config.use_smpnet:
            submodel_type=SMOTMPNet
        else:
            submodel_type=MOTMPNet

        return GLOMOT_Net(submodel_type=submodel_type, submodel_params=mpntrack_params['graph_model_params'],
                       hicl_depth=self.config.hicl_depth, use_motion=self.config.mpn_use_motion,
                       use_reid_edge=self.config.mpn_use_reid_edge, use_pos_edge=self.config.mpn_use_pos_edge,
                       share_weights=self.config.share_weights, edge_level_embed=self.config.edge_level_embed,
                       node_level_embed=self.config.node_level_embed
                       ).to(self.config.device)

    def _get_dataset(self, mode):
        """
        Create dataset objects
        """
        return MOTSceneDataset(config=self.config, seqs=self.seqs[mode], mode=mode)

    def _get_train_dataloader(self):
        """
         Set up the dataset and the dataloader
        """

        train_dataset = self._get_dataset(mode='train')
        train_loader = DataLoader(train_dataset, batch_size=self.config.num_batch, num_workers=self.config.num_workers,
                                  shuffle=True,
                                  worker_init_fn=seed_worker, generator=seed_generator(), )
        return train_loader

    def _get_optimizer(self):
        """
         Set up the optimizer
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        return optimizer

    def _get_projector(self, graph):
        """
        Set up the projector that will round the edge predictions
        """
        if self.config.rounding_method == 'greedy':
            projector = GreedyProjector(full_graph=graph)

        elif self.config.rounding_method == 'exact':
            projector = ExactProjector(full_graph=graph, solver_backend=self.config.solver_backend)

        else:
            raise RuntimeError("Rounding type for projector not understood")
        return projector

    def _save_model(self):
        """
        Save the model
        """
        # Create models folder
        model_path = osp.join(self.config.experiment_path, "models")
        os.makedirs(model_path, exist_ok=True)

        # Create the file
        file_name = osp.join(model_path, f"hiclnet_epoch_{self.train_epoch}_iteration{self.train_iteration}.pth")
        torch.save(self.model.state_dict(), file_name)

        # Create a full checkpoint
        checkpoint_path = osp.join(self.config.experiment_path, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
        cp_file_name = osp.join(checkpoint_path, f"checkpoint_{self.train_epoch}.pth")
        torch.save({'epoch': self.train_epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, cp_file_name)

    def load_pretrained_model(self):
        """
        Load a pretrained model
        """
        # Initialize the model
        pretrained_model = self._get_model()

        ####
        gpu_num=len(self.config.gpus)
        if gpu_num > 1:
            pretrained_model= torch.nn.DataParallel(pretrained_model, self.config.gpus).to('cuda')




        # Load weights
        ckpt = torch.load(self.config.model_path)
        filtered_dict1 = {key: value for key, value in ckpt.items() if "layers.0.encoder.node_mlp" in key}
        filtered_dict2 = {key: value for key, value in ckpt.items() if "layers.0.encoder.edge_mlp" in key}
        filtered_dict3 = {key: value for key, value in ckpt.items() if 'layers.0.MPNet.edge_model' in key}
        filtered_dict4 = {key: value for key, value in ckpt.items() if 'layers.0.MPNet.edge_model' in key}
        filtered_dict5 = {key: value for key, value in ckpt.items() if 'layers.0.MPNet.node_model' in key}
        filtered_dict6 = {key: value for key, value in ckpt.items() if 'layers.0.classifier.edge_mlp' in key}
        filtered_dict7 = {key: value for key, value in ckpt.items() if 'edge_level_embed.weight' in key}
        filtered_dict=  {**filtered_dict1, **filtered_dict2, **filtered_dict3,**filtered_dict4,**filtered_dict5,**filtered_dict6,**filtered_dict7}


        if 'edge_level_embed.weight' in pretrained_model.state_dict() and 'edge_level_embed.weight' in filtered_dict:
            filtered_dict['edge_level_embed.weight'] = filtered_dict['edge_level_embed.weight'][:pretrained_model.state_dict()['edge_level_embed.weight'].shape[0], :]

        if self.config.use_smpnet:
            pretrained_model.load_state_dict(ckpt)
        else:
            pretrained_model.load_state_dict(filtered_dict)


        pretrained_model.train()

        return pretrained_model

    def _hicl_to_curr(self, hicl_graphs):
        """
        Method that creates a batch of current graphs from hierarchical graphs in three steps:
        1) Create the batch graphs with node features and all time valid edge connections
        2) **Optionally** compute motion features for all time valid edge connections
        3) Use those motion features, as well as reid to define KNN edges and define 
        edge features for each graph in the batch to obtain the final graphs
        """
        batch = Batch.from_data_list([hicl_graph.construct_curr_graph_nodes(self.config)
                                                        for hicl_graph in hicl_graphs])

        curr_depth = hicl_graphs[0].curr_depth
        if self.config.do_motion and curr_depth >0:
            motion_pred = self.predict_motion(batch, curr_depth = curr_depth)
            batch.pruning_score = compute_giou_fwrd_bwrd_motion_sim(batch, motion_pred)
            
            if 'estimate_vel' in motion_pred[0]:
                batch.fwrd_vel, batch.bwrd_vel = motion_pred[0]['estimate_vel'], motion_pred[1]['estimate_vel']
                            
        else:
            motion_pred = None
        
        # Now unbatch graphs, add their remaining features, and batch them again
        curr_graphs = Batch.to_data_list(batch)



        curr_graph_batch = Batch.from_data_list([hicl_graphs[0].add_edges_to_curr_graph(self.config, curr_graphs[0])])

        return curr_graph_batch, motion_pred

    def _postprocess_graph(self, graph, remove_negatives=True, decision_threshold=0.5):
        """
        Process the graph before feeding it to the projector
        """
        to_undirected_graph(graph, attrs_to_update=('edge_preds', 'edge_labels'))
        to_lightweight_graph(graph, attrs_to_del=('reid_emb_dists', 'x', 'edge_attr', 'edge_labels'))
        if remove_negatives:
            to_positive_decision_graph(graph, decision_threshold)

    def _project_graph(self, graph, decision_threshold=0.5):
        """
        Project model output with a solver
        """
        projector = self._get_projector(graph=graph)
        projector.project(decision_threshold)
        graph = graph.numpy()
        graph.constr_satisf_rate = projector.constr_satisf_rate
        return graph

    def _assign_labels(self, graph):
        """
        Clusters the nodes together based on edge predictions
        """
        # Only keep the non-zero edges
        nonzero_mask = graph.edge_preds == 1
        nonzero_edge_index = graph.edge_index.T[nonzero_mask].T
        nonzero_edges = graph.edge_preds[nonzero_mask].astype(int)
        graph_shape = (graph.num_nodes, graph.num_nodes)

        # Express the result as a CSR matrix so that it can be fed to 'connected_components')
        csr_graph = csr_matrix((nonzero_edges, (tuple(nonzero_edge_index))), shape=graph_shape)


        n_components, labels = connected_components(csgraph=csr_graph, directed=False, return_labels=True)

        return n_components, labels

    def _calculate_loss(self, outputs, edge_labels, edge_mask,curr_batch,frame_gap):
        """
        Calculate MPNTrack loss given edge predictions and edge_labels
        """

        # Compute Weighted BCE:
        loss = torch.as_tensor([.0], device=self.config.device)
        num_steps = len(outputs['classified_edges'])
        for step in range(num_steps):
            if self.config.no_fp_loss and torch.any(edge_mask).item():
                if self.config.use_appearance_loss or self.config.use_temporal_loss:
                    loss += self.loss_function(outputs['classified_edges'][step].view(-1)[edge_mask], edge_labels.view(-1)[edge_mask],curr_batch,frame_gap)
                else:
                    loss += self.loss_function(outputs['classified_edges'][step].view(-1)[edge_mask], edge_labels.view(-1)[edge_mask])
            else:
                if self.config.use_appearance_loss or self.config.use_temporal_loss:
                    loss += self.loss_function(outputs['classified_edges'][step].view(-1), edge_labels.view(-1),curr_batch,frame_gap)
                else:
                    loss += self.loss_function(outputs['classified_edges'][step].view(-1), edge_labels.view(-1))
        return loss


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
        boxes = torch.stack([x1, y1, x2, y2], dim=1)  # [N, 4]


        inter_top_left = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])  # [N, N, 2]
        inter_bottom_right = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])  # [N, N, 2]

        inter_wh = (inter_bottom_right - inter_top_left).clamp(min=0)  # [N, N, 2]
        inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # [N, N]


        torch.diagonal(inter_area).fill_(0)


        total_occlusion_area = torch.sum(inter_area, dim=1)  # [N]
        occlusion_ratio = total_occlusion_area / (areas + 1e-6)
        occlusion_depth = 1 / (1 + occlusion_ratio)


        weight_size = 0.2
        weight_position = 0.4
        weight_occlusion = 0.4

        pseudo_depths = (weight_size * size_depth +
                         weight_position * position_depth +
                         weight_occlusion * occlusion_depth)

        return pseudo_depths

    def _train_epoch(self):
        """
        Train a single epoch
        """
        logs = {"Loss": [], "Loss_per_Depth": [[] for j in range(self.config.hicl_depth)], "Time": []}

        for i, train_batch in enumerate(self.train_dataloader):
            t_start = time.time()

            # Iteration update
            self.train_iteration += 1
            self.graph_info=self.train_dataloader.sampler.data_source.seq_and_frames[i]
            self.optimizer.zero_grad()

            train_batch.to(self.config.device)  # Send batch to the device
            hicl_graphs = train_batch.to_data_list()  # Initialize the hierarchical graphs

            loss = torch.as_tensor([.0], device=self.config.device)  # Initialize the batch loss

            print(self.train_dataloader.sampler.data_source.seq_and_frames[i])

            if 'x_depth' not in hicl_graphs[0]:
                sequece_name = self.train_dataloader.sampler.data_source.seq_and_frames[i][0]
                img_width = self.image_size[sequece_name]['width']
                image_height = self.image_size[sequece_name]['height']
                bbox=hicl_graphs[0]['x_bbox']
                frame=hicl_graphs[0]['x_frame']

                x_depth=[]
                unique_frames = torch.unique(frame).cpu().numpy()
                for frame_value in unique_frames:

                    frame_mask = frame == frame_value
                    current_bbox = bbox[frame_mask.flatten()]
                    image_size = (img_width, image_height)
                    detections = [(x1, y1, w, h) for x1, y1, w, h in current_bbox]

                    pseudo_depths = self.calculate_pseudo_depth(image_size, detections)


                    x_depth.extend(pseudo_depths)

                x_depth = torch.tensor(x_depth, device=self.config.device).reshape_as(hicl_graphs[0]['x_frame']).float()
                hicl_graphs[0]['x_depth'] = x_depth

            _, loss, logs = self.hicl_forward(hicl_graphs = hicl_graphs, 
                                              logs = logs, 
                                              oracle = False, 
                                              mode = 'train', 
                                              max_depth = self.active_train_depth, 
                                              project_max_depth = self.active_train_depth - 1)

            # Update the weights
            loss.backward()
            self.optimizer.step()

            # Keep track of the logs
            t_end = time.time()
            logs["Loss"].append(loss.detach().item())
            logs["Time"].append(t_end-t_start)

            self._log_tb_train_metrics(logs)

            # Verbose
            if i % self.config.verbose_iteration == 0 and i != 0:
                #######
                avg_time_per_iter = (sum(logs["Time"][i - self.config.verbose_iteration:i]) / self.config.verbose_iteration)

                remaining_iterations = len(self.train_dataloader) - i

                remaining_time_seconds = remaining_iterations * avg_time_per_iter
                remaining_time = timedelta(seconds=int(remaining_time_seconds))
                hours = remaining_time.seconds // 3600
                minutes = (remaining_time.seconds % 3600) // 60
                remaining_time_formatted=f"remain {hours}hour{minutes}min"

                #print(f"Iteration {i} / {len(self.train_dataloader)} - Training Loss:", statistics.mean(logs["Loss"][i-self.config.verbose_iteration:i]), '- Time:', sum(logs["Time"][i-self.config.verbose_iteration:i]))  # Verbose
                print(f"Iteration {i} / {len(self.train_dataloader)} - Training Loss: {statistics.mean(logs['Loss'][i - self.config.verbose_iteration:i])} - Time: {sum(logs['Time'][i - self.config.verbose_iteration:i])}s - {remaining_time_formatted}")

            # Update active train depth if required
            if self.active_train_depth < self.config.hicl_depth:
                if self.train_iteration % self.config.depth_pretrain_iteration == 0:
                    self.active_train_depth = min(self.active_train_depth+1, self.config.hicl_depth)
                    print("*****")
                    print("Frozen layers are unlocked! Current active training depth is:", self.active_train_depth)
                    print("*****")

        self.train_epoch += 1

        return logs

    def predict_motion(self, batch, curr_depth):
        """Predict forward and backward future/past locations for each track with length >1"""

        motion_model = self.motion_model
        assert motion_model is not None

        fwrd_motion_pred = motion_model(x_motion=batch.x_fwrd_motion, 
                                        x_last_pos=batch.x_center_end[~batch.x_ignore_traj],
                                        pred_length=self.config.motion_pred_length[curr_depth - 1],
                                        linear_center_only=self.config.linear_center_only
                                        )
        
        bwrd_motion_pred = motion_model(x_motion=batch.x_bwrd_motion, 
                                        x_last_pos=batch.x_center_start[~batch.x_ignore_traj],
                                        pred_length=self.config.motion_pred_length[curr_depth - 1],
                                        linear_center_only=self.config.linear_center_only
                                        )            

        return (fwrd_motion_pred, bwrd_motion_pred)

    def hicl_forward(self, hicl_graphs, logs, oracle, mode, max_depth, project_max_depth):
        hicl_feats=None
        loss = torch.as_tensor([.0], device=self.config.device)  # Initialize the batch loss
        
        # For each depth
        for curr_depth in range(max_depth):
        
            # Put the graph into the correct format
            curr_batch, _  = self._hicl_to_curr(hicl_graphs=hicl_graphs)  # Create curr_graphs from hierarachical graphs
            batch_idx = curr_batch.batch

            if curr_depth == 0 or not self.config.do_hicl_feats:
                curr_batch.hicl_feats = None

            elif hicl_feats is not None:
                curr_batch.hicl_feats = hicl_feats

            # Forward pass if there is an edge
            if curr_batch.edge_index.numel():                
                if oracle:
                    # Oracle results
                    curr_batch.edge_preds = curr_batch.edge_labels
                    logs[curr_depth]["Loss"].append(0.)
                    # Calculate batch classification metrics and loss
                    logs[curr_depth] = self._calculate_true_false_metrics(edge_preds=curr_batch.edge_preds,
                                                                edge_labels=curr_batch.edge_labels, 
                                                                logs=logs[curr_depth])
                else:
                    # Graph based forward pass
                    outputs = self.model(curr_batch, curr_depth)  # Forward pass for this specific depth
                    
                    # Produce decisions
                    curr_batch.edge_preds = torch.sigmoid(outputs['classified_edges'][-1].view(-1).detach())

                    if mode == 'val':
                        # Calculate the batch loss
                        logs[curr_depth]["Loss"].append(self._calculate_loss(outputs=outputs, edge_labels=curr_batch.edge_labels, edge_mask=curr_batch.edge_mask).item())
                        # Calculate batch classification metrics and loss
                        logs[curr_depth] = self._calculate_true_false_metrics(edge_preds=curr_batch.edge_preds,
                                                                    edge_labels=curr_batch.edge_labels, logs=logs[curr_depth])
                    elif mode == 'train':
                        frame_gap=int(hicl_graphs[0].x_frame[-1]-hicl_graphs[0].x_frame[0])
                        # Calculate loss and prepare for a forward pass
                        loss_curr_depth = self._calculate_loss(outputs=outputs, edge_labels=curr_batch.edge_labels, edge_mask=curr_batch.edge_mask,curr_batch=curr_batch,frame_gap=frame_gap)
                        loss += loss_curr_depth
                        
                        logs["Loss_per_Depth"][curr_depth].append(loss_curr_depth.detach().item())  # log the curr loss


            graph_data_list = curr_batch.to_data_list()
            if mode != 'train':
                assert len(graph_data_list) == 1, "Track batch size is greater than 1"

            hicl_feats = []
            if curr_depth < project_max_depth:  # Last layer update is not necessary for training
                for ix_graph, graph in enumerate(graph_data_list):        
                    if graph.edge_index.numel():

                        # Process the graph before feeding it to the projector
                        self._postprocess_graph(graph)
                        # Project model output with a solver
                        graph = self._project_graph(graph)
                        
                        # Assign ped ids
                        n_components, labels = self._assign_labels(graph)
                    
                        node_mask = batch_idx == ix_graph
                        if self.config.do_hicl_feats and not oracle:
                            hicl_feats.append(self.model.layers[curr_depth].hicl_feats_encoder.pool_node_feats(outputs['node_feats'][node_mask], labels))

                        # Update the hierarchical graphs with new map_from_init and depth
                        hicl_graphs[ix_graph].update_maps_and_depth(labels)

                    else:
                        # Update the hierarchical graphs
                        hicl_graphs[ix_graph].update_maps_and_depth_wo_labels()
            
            if len(hicl_feats) > 0:
                hicl_feats = torch.cat(hicl_feats)
            
            else:
                hicl_feats =None

        return hicl_graphs, loss, logs

    def hicl_forward_with_buffer(self, hicl_graphs, logs, oracle, mode, max_depth, project_max_depth,buffer_node_pool):
        hicl_feats = None
        loss = torch.as_tensor([.0], device=self.config.device)  # Initialize the batch loss

        # For each depth
        for curr_depth in range(max_depth):

            # Put the graph into the correct format
            curr_batch, _ = self._hicl_to_curr(
                hicl_graphs=hicl_graphs)  # Create curr_graphs from hierarachical graphs
            batch_idx = curr_batch.batch

            if curr_depth == 0 or not self.config.do_hicl_feats:
                curr_batch.hicl_feats = None

            elif hicl_feats is not None:
                curr_batch.hicl_feats = hicl_feats

            # Forward pass if there is an edge
            if curr_batch.edge_index.numel():
                if oracle:
                    # Oracle results
                    curr_batch.edge_preds = curr_batch.edge_labels
                    logs[curr_depth]["Loss"].append(0.)
                    # Calculate batch classification metrics and loss
                    logs[curr_depth] = self._calculate_true_false_metrics(edge_preds=curr_batch.edge_preds,
                                                                          edge_labels=curr_batch.edge_labels,
                                                                          logs=logs[curr_depth])
                else:
                    # Graph based forward pass
                    outputs = self.model(curr_batch, curr_depth)  # Forward pass for this specific depth

                    # Produce decisions
                    curr_batch.edge_preds = torch.sigmoid(outputs['classified_edges'][-1].view(-1).detach())

                    if mode == 'val':
                        # Calculate the batch loss
                        logs[curr_depth]["Loss"].append(
                            self._calculate_loss(outputs=outputs, edge_labels=curr_batch.edge_labels,
                                                 edge_mask=curr_batch.edge_mask).item())
                        # Calculate batch classification metrics and loss
                        logs[curr_depth] = self._calculate_true_false_metrics(edge_preds=curr_batch.edge_preds,
                                                                              edge_labels=curr_batch.edge_labels,
                                                                              logs=logs[curr_depth])
                    elif mode == 'train':
                        # Calculate loss and prepare for a forward pass
                        loss_curr_depth = self._calculate_loss(outputs=outputs, edge_labels=curr_batch.edge_labels,
                                                               edge_mask=curr_batch.edge_mask)
                        loss += loss_curr_depth

                        logs["Loss_per_Depth"][curr_depth].append(loss_curr_depth.detach().item())  # log the curr loss

            graph_data_list = curr_batch.to_data_list()
            if mode != 'train':
                assert len(graph_data_list) == 1, "Track batch size is greater than 1"

            hicl_feats = []
            if curr_depth < project_max_depth:  # Last layer update is not necessary for training
                for ix_graph, graph in enumerate(graph_data_list):
                    if graph.edge_index.numel():

                        # Process the graph before feeding it to the projector
                        self._postprocess_graph(graph)
                        # Project model output with a solver
                        graph = self._project_graph(graph)  #

                        # Assign ped ids
                        n_components, labels = self._assign_labels(graph)

                        node_mask = batch_idx == ix_graph
                        if self.config.do_hicl_feats and not oracle:
                            hicl_feats.append(self.model.layers[curr_depth].hicl_feats_encoder.pool_node_feats(
                                outputs['node_feats'][node_mask], labels))

                        # Update the hierarchical graphs with new map_from_init and depth
                        hicl_graphs[ix_graph].update_maps_and_depth(labels)

                    else:
                        # Update the hierarchical graphs
                        hicl_graphs[ix_graph].update_maps_and_depth_wo_labels()

            if len(hicl_feats) > 0:
                hicl_feats = torch.cat(hicl_feats)

            else:
                hicl_feats = None

        return hicl_graphs, loss, logs

    def _log_tb_class_metrics(self, epoch_val_logs, epoc_val_logs_per_depth):
        # Struct is 'layer name :  metric_name :val
        if self.logger is not None:
            _METRICS_TO_LOG = ['Loss','Precision', 'Recall', 'F1', 'Accuracy']
            prefixs = [f'layer_{layer_idx + 1}' for layer_idx in range(len(epoc_val_logs_per_depth))] + ['overall']
            tb_logs = epoc_val_logs_per_depth + [epoch_val_logs]
            for prefix, logs_dict in zip(prefixs, tb_logs):
                for metric_name, metric_val in logs_dict.items():
                    if metric_name in _METRICS_TO_LOG:
                        tag = '/'.join(['val', 'secondary', prefix, metric_name])
                        self.logger.add_scalar(tag, metric_val, global_step=self.train_iteration)

    def _log_tb_train_metrics(self, logs):
        if self.logger is not None:
            loss_logs = logs['Loss_per_Depth'] + [logs['Loss']]
            prefixes = [f'layer_{layer_idx + 1}' for layer_idx in range(len(logs['Loss_per_Depth']))] + ['overall']

            for prefix, losses in zip(prefixes, loss_logs):
                if len(losses) > 0:
                    tag = '/'.join(['train', prefix, 'loss'])
                    self.logger.add_scalar(tag, losses[-1], global_step=self.train_iteration)

    def _log_tb_mot_metrics(self, mot_metrics):
        if self.logger is not None:
            path = list(mot_metrics['MotChallenge2DBox'].keys())[0]
            _METRICS_GROUPS = ['HOTA', 'CLEAR', 'Identity']
            _METRICS_TO_LOG = ['HOTA','AssA', 'DetA', 'MOTA', 'IDF1']

            metrics_ = mot_metrics['MotChallenge2DBox'][path]['COMBINED_SEQ']['pedestrian'] # We may need more classes for MOTCha
            for metrics_group_name in _METRICS_GROUPS:
                group_metrics = metrics_[metrics_group_name]
                for metric_name, metric_val in group_metrics.items():
                    if metric_name in _METRICS_TO_LOG:
                        if isinstance(metric_val, np.ndarray):
                            metric_val = np.mean(metric_val)
                        tag = '/'.join(['val', 'mot', metric_name])
                        self.logger.add_scalar(tag, metric_val, global_step=self.train_iteration)

    def train(self):
        """
        Perform a full training
        """
        self.val_split=False

        assert self.model.training, "Training error: Model is not in training mode"

        logs = {"Train_Loss": [], "Val_Loss": [],
                "Train_Loss_per_Depth": [[] for j in range(self.config.hicl_depth)],
                "Val_Loss_per_Depth": [[] for j in range(self.config.hicl_depth)]}  # Training logs

        # Training loop
        for epoch in range(1, self.config.num_epoch+1):
            t_start = time.time()
            print("###############")
            print("  Epoch ", epoch)
            print("###############")

            # Create epoch output dir
            epoch_path = osp.join(self.config.experiment_path, 'Epoch' + str(epoch))
            os.makedirs(epoch_path, exist_ok=True)

            if self.train_dataloader.dataset.seq_info_dicts:
                self.image_size = {}
                for key, seq_info in self.train_dataloader.dataset.seq_info_dicts.items():
                    self.image_size[key] = {}
                    frame_height = seq_info.get('frame_height')
                    frame_width = seq_info.get('frame_width')
                    if frame_height is not None:
                        self.image_size[key]['height'] = frame_height
                    if frame_width is not None:
                        self.image_size[key]['width'] = frame_width
            #######
            # Train for one epoch
            print(self.loss_function)

            epoch_train_logs = self._train_epoch()

            # Train loss logs
            logs["Train_Loss"].append(statistics.mean(epoch_train_logs["Loss"]))
            for j in range(self.config.hicl_depth):
                # If a layer is frozen, set the loss to 0
                if j < self.active_train_depth and len(epoch_train_logs["Loss_per_Depth"][j]) > 0:
                    logs["Train_Loss_per_Depth"][j].append(statistics.mean(epoch_train_logs["Loss_per_Depth"][j]))
                else:
                    logs["Train_Loss_per_Depth"][j].append(0.)

            # Validation steps
            if self.val_split and epoch >= self.config.start_eval:
                epoch_val_logs, epoc_val_logs_per_depth = self.track(dataset=self.val_dataset, output_path=epoch_path, mode='val', oracle=False)
                # Validation logs
                logs["Val_Loss"].append(epoch_val_logs["Loss"])
                for j in range(self.config.hicl_depth):
                    logs["Val_Loss_per_Depth"][j].append(epoc_val_logs_per_depth[j]["Loss"])

                # Tensorboard logging:
                self._log_tb_class_metrics(epoch_val_logs, epoc_val_logs_per_depth)

                # MOT metrics
                mot_metrics= evaluate_mot17(tracker_path=epoch_path, split=self.val_split, data_path=self.config.data_path,
                                            tracker_sub_folder=self.config.mot_sub_folder, output_sub_folder=self.config.mot_sub_folder)[0]
                self._log_tb_mot_metrics(mot_metrics) 

            # Plot losses
            self._plot_losses(logs)
            self._plot_losses_per_depth(logs)

            # Save model checkpoint
            if self.config.save_cp and (epoch + 1) % self.config.save_epoch_interval == 0:
                self._save_model()

            # Time information
            t_end = time.time()
            print(f"Epoch completed in {round((t_end - t_start) / 60, 2)} minutes")

    def track(self, dataset, output_path, mode='val', oracle=False):
        """
        Main tracking method. Given a dataset, track every sequence in it and create output files.
        """

        # Set the model in the test mode
        self.model.eval()
        assert not self.model.training, "Test error: Model is not in evaluation mode"

        # Disable gradients
        with torch.no_grad():

            # Separate dictionary to bookkeep the logs for each depth network
            logs_all = [{"Loss": [], "TP": [], "FP": [], "TN": [], "FN": [], "CSR": []} for i in range(self.config.hicl_depth)]

            # Loop over sequences
            for seq, seq_and_frames in dataset.sparse_frames_per_seq.items():
                print("Tracking", seq)
                # Loop over each datapoint - Equivalent to using a dataloader with batch 1
                seq_dfs = []
                for seq_name, start_frame, end_frame in seq_and_frames:
                    # Equivalent to train_batch with a single datapoint
                    data = dataset.get_graph_from_seq_and_frames(seq_name=seq_name, start_frame=start_frame, end_frame=end_frame)
                    data.to(self.config.device)

                    # Small trick to utilize torch-geometric built in functions
                    track_batch = Batch.from_data_list([data])
                    hicl_graphs = track_batch.to_data_list()
                    # For each depth
                    hicl_graphs, _, logs_all = self.hicl_forward(hicl_graphs = hicl_graphs, 
                                                                 logs = logs_all, 
                                                                 oracle = oracle, 
                                                                 mode = mode, 
                                                                 max_depth = self.config.hicl_depth, 
                                                                 project_max_depth = self.config.hicl_depth)

                    # Pedestrian ids
                    ped_labels = hicl_graphs[0].get_labels()

                    # Get graph df
                    graph_df, _ = dataset.get_df_from_seq_and_frames(seq_name=seq_name, start_frame=start_frame, end_frame=end_frame)
                    assert len(ped_labels) == graph_df.shape[0], "Ped Ids Label format is wrong"

                    # Each Connected Component is a Ped Id. Assign those values to our DataFrame:
                    graph_output_df = graph_df.copy()
                    graph_output_df['ped_id'] = ped_labels
                    graph_output_df = graph_output_df[self.config.VIDEO_COLUMNS + ['conf', 'detection_id']].copy()

                    # Append the new df
                    seq_dfs.append(graph_output_df)



                seq_merged_df = self._merge_subseq_dfs(seq_dfs)

                seq_output_df=seq_merged_df

                os.makedirs(osp.join(output_path, self.config.mot_sub_folder), exist_ok=True)
                tracking_file_path = osp.join(output_path, self.config.mot_sub_folder, seq_name + '.txt')
                self._save_results(df=seq_output_df, output_file_path=tracking_file_path)

            print("-----")
            print("Tracking completed!")

            # Print metrics
            logs_total = {}
            if mode != 'test':
                for i in range(self.config.hicl_depth):
                    # Calculate accuracy, precision, recall
                    logs = logs_all[i]
                    if logs["Loss"]:
                        print("Depth", i+1, "- Metrics:")
                        logs = self._postprocess_logs(logs=logs)

                # Total logs - Cumulative of every layer
                print("TOTAL - Metrics:")
                logs_total = {"Loss": [logs_all[i]["Loss"] for i in range(self.config.hicl_depth)],
                              "TP": [logs_all[i]["TP"] for i in range(self.config.hicl_depth)],
                              "FP": [logs_all[i]["FP"] for i in range(self.config.hicl_depth)],
                              "TN": [logs_all[i]["TN"] for i in range(self.config.hicl_depth)],
                              "FN": [logs_all[i]["FN"] for i in range(self.config.hicl_depth)],
                              "CSR": [logs_all[i]["CSR"] for i in range(self.config.hicl_depth)]}
                logs_total["Loss"] = [sum(logs_total["Loss"])]  # To be compatible with train loss (sum of hicl layers)
                logs_total = self._postprocess_logs(logs_total)

                print("-----")



        # Set the model back in training mode
        self.model.train()

        return logs_total, logs_all

    def merge_predictions(self,merged_info,previous_info,current_info,next_info):
        previous_bboxes = previous_info[['detection_id', 'ped_id', 'bb_left', 'bb_width']]
        current_bboxes = current_info[['detection_id', 'ped_id', 'bb_left', 'bb_width']]
        next_bboxes = next_info[['detection_id', 'ped_id', 'bb_left', 'bb_width']]
        next_ori_info=next_info

        current_next_ped_id_mapping = {}
        for _, current_row in current_bboxes.iterrows():

            matching_next_row = next_bboxes[next_bboxes['ped_id'] == current_row['ped_id']]
            if not matching_next_row.empty:
                current_next_ped_id_mapping[current_row['detection_id']] = matching_next_row['detection_id'].values[0]

        new_object_info = []
        if len(current_next_ped_id_mapping) < len(next_info):
            for det_id in next_info['detection_id']:
                if det_id not in current_next_ped_id_mapping.values():
                    new_object_info.append(det_id)

        miss_object_info = []
        if len(current_next_ped_id_mapping) < len(current_info):
            for det_id_miss in current_info['detection_id']:
                if det_id_miss not in current_next_ped_id_mapping.keys():
                    miss_object_info.append(det_id_miss)

        for det_id_update in previous_info['detection_id']:
            if det_id_update not in miss_object_info:
                matching_previous_bboxes = previous_bboxes[previous_bboxes['detection_id'] == det_id_update]
                current_info.loc[current_info['detection_id'] == det_id_update, 'ped_id'] = matching_previous_bboxes['ped_id'].values[0]
                next_pos=current_next_ped_id_mapping[det_id_update]
                next_info.loc[next_info['detection_id'] == next_pos, 'ped_id'] = matching_previous_bboxes['ped_id'].values[0]

        if len(current_info)>0:
            max_current_id=max(current_info['ped_id'])
        else:
            max_current_id=0
        if len(merged_info)>0:
            max_merged_id = max(merged_info['ped_id'])
        else:
            max_merged_id=0
        max_ped_id=max(max_current_id,max_merged_id)
        for det_id_new in new_object_info:
            next_info.loc[next_info['detection_id'] == det_id_new, 'ped_id'] = max_ped_id+1
            max_ped_id+=1

        for det_id_fix_miss in miss_object_info:
            current_info.loc[current_info['detection_id'] == det_id_fix_miss, 'ped_id'] = previous_info.loc[previous_info['detection_id'] == det_id_fix_miss, 'ped_id'].values[0]


        merged_info = pd.concat([merged_info,current_info,next_info]).drop_duplicates(
            subset=['frame', 'detection_id'], keep='last')


        return merged_info

    def save_dataframe(self,data,output_path):

        data.to_excel(output_path, index=False)


    def merge_first_frame_info(self,current_info,next_info):
        current_bboxes = current_info[['detection_id', 'ped_id', 'bb_left', 'bb_width']]
        next_bboxes = next_info[['detection_id', 'ped_id', 'bb_left', 'bb_width']]

        current_next_ped_id_mapping = {}
        for _, current_row in current_bboxes.iterrows():

            matching_next_row = next_bboxes[next_bboxes['ped_id'] == current_row['ped_id']]
            if not matching_next_row.empty:
                current_next_ped_id_mapping[current_row['detection_id']] = matching_next_row['detection_id'].values[0]

        miss_object_index = []
        miss_object_info = []
        if len(current_next_ped_id_mapping) < len(current_info):
            for det_id_miss in current_info['detection_id']:
                if det_id_miss not in current_next_ped_id_mapping.keys():
                    miss_object_info.append(det_id_miss)
                    miss_object_index.append(det_id_miss - min(current_info['detection_id']))  #index 从0开始


        merged_info = pd.concat([current_info, next_info]).drop_duplicates(
            subset=['frame', 'detection_id'], keep='last')

        unmatched_current_nodes = current_info[current_info['detection_id'].isin(miss_object_info)].copy()

        return merged_info, unmatched_current_nodes,miss_object_index

    def get_buffer_info(self,previous_info,current_info):

        detection_ids_in_B = set(previous_info['detection_id'].unique())

        new_rows = []
        node_buffer_detection_id=[]

        for index, row in current_info.iterrows():
            detection_id = row['detection_id']
            if (detection_id not in detection_ids_in_B) and row['id']!=-1:
                node_buffer_detection_id.append(detection_id)
                new_rows.append(row.to_dict())

        new_df = pd.DataFrame(new_rows)
        return new_df,node_buffer_detection_id


    def merge_predictions_with_buffer(self, merged_info, previous_info, current_info, next_info):
        previous_bboxes = previous_info[['detection_id', 'ped_id', 'bb_left', 'bb_width','id']]
        current_bboxes = current_info[['detection_id', 'ped_id', 'bb_left', 'bb_width','id']]
        next_bboxes = next_info[['detection_id', 'ped_id', 'bb_left', 'bb_width','id']]
        node_buffer_info,node_buffer_detection_id=self.get_buffer_info(previous_info,current_info)

        current_next_ped_id_mapping = {}
        ori_previous_info=previous_info.copy()
        for _, current_row in current_bboxes.iterrows():

            matching_next_row = next_bboxes[next_bboxes['ped_id'] == current_row['ped_id']]
            if not matching_next_row.empty:
                current_next_ped_id_mapping[current_row['detection_id']] = matching_next_row['detection_id'].values[0]
        new_object_info = []
        if len(current_next_ped_id_mapping) < len(next_info):
            for det_id in next_info['detection_id']:
                if det_id not in current_next_ped_id_mapping.values():
                    new_object_info.append(det_id)

        miss_object_info = []
        miss_object_index=[]
        if len(current_next_ped_id_mapping) < len(current_info):
            for det_id_miss in current_info['detection_id']:
                if det_id_miss not in current_next_ped_id_mapping.keys():
                    miss_object_info.append(det_id_miss)
                    if det_id_miss in node_buffer_detection_id:
                        miss_object_index.append(-1)
                    else:
                        frame_first_object_index= current_info['detection_id'].values[len(node_buffer_info)]
                        miss_object_index.append(det_id_miss-frame_first_object_index)

        if len(node_buffer_info)>0:
            process_buffer_info=node_buffer_info.copy()
            process_buffer_info['ped_id']=process_buffer_info['id']
            previous_info=pd.concat([process_buffer_info, previous_info], ignore_index=True)
            previous_bboxes = previous_info[['detection_id', 'ped_id', 'bb_left', 'bb_width', 'id']]

        for det_id_update in previous_info['detection_id']:
            if det_id_update not in miss_object_info:
                matching_previous_bboxes = previous_bboxes[previous_bboxes['detection_id'] == det_id_update]
                current_info.loc[current_info['detection_id'] == det_id_update, 'ped_id'] = \
                matching_previous_bboxes['ped_id'].values[0]
                next_pos = current_next_ped_id_mapping[det_id_update]
                next_info.loc[next_info['detection_id'] == next_pos, 'ped_id'] = \
                matching_previous_bboxes['ped_id'].values[0]

        if len(previous_info) > 0:
            max_previous_id = max(previous_info['ped_id'])
        else:
            max_previous_id = 0
        if len(merged_info) > 0:
            max_merged_id = max(merged_info['ped_id'])
        else:
            max_merged_id = 0
        max_ped_id = max(max_previous_id, max_merged_id)
        for det_id_new in new_object_info:
            next_info.loc[next_info['detection_id'] == det_id_new, 'ped_id'] = max_ped_id + 1
            max_ped_id += 1

        for det_id_fix_miss in miss_object_info:
            current_info.loc[current_info['detection_id'] == det_id_fix_miss, 'ped_id'] = \
            previous_info.loc[previous_info['detection_id'] == det_id_fix_miss, 'ped_id'].values[0]

        merged_info = pd.concat([merged_info, ori_previous_info, next_info]).drop_duplicates(
            subset=['frame', 'detection_id'], keep='last')
        merged_info.index=merged_info['detection_id'].values

        unmatched_current_nodes = current_info[current_info['detection_id'].isin(miss_object_info)].copy()
        unmatched_current_nodes['index_in_miss_frame']=miss_object_index


        delete_buffer_node_id=[]
        if len(node_buffer_info)>0:
            for id in node_buffer_info['detection_id']:
                if id not in miss_object_info:
                    delete_buffer_node_id.append(id)
        return merged_info,unmatched_current_nodes,miss_object_index,delete_buffer_node_id

    def merge_occlusion(self, merged_info, current_info, next_info,graph_df):
        current_bboxes = current_info[['detection_id', 'ped_id', 'bb_left', 'bb_width','id']]
        next_bboxes = next_info[['detection_id', 'ped_id', 'bb_left', 'bb_width','id']]

        current_no_buffer_info= graph_df[graph_df['frame']==current_info['frame'].values[0]]
        detection_ids_in_current = set(current_no_buffer_info['detection_id'])
        buffer_node_info= current_info[~current_info['detection_id'].isin(detection_ids_in_current)]

        current_next_ped_id_mapping = {}

        for _, current_row in current_bboxes.iterrows():

            matching_next_row = next_bboxes[next_bboxes['ped_id'] == current_row['ped_id']]
            if not matching_next_row.empty:
                current_next_ped_id_mapping[current_row['detection_id']] = matching_next_row['detection_id'].values[0]

        current_new_object_info = []
        for det_id in current_no_buffer_info['detection_id']:
            current_new_object_info.append(det_id)


        max_ped_id = max(merged_info['ped_id'])

        for det_id_new in current_new_object_info:
            current_info.loc[current_info['detection_id'] == det_id_new, 'ped_id']= max_ped_id + 1
            current_no_buffer_info.loc[current_no_buffer_info['detection_id'] == det_id_new, 'ped_id']= max_ped_id + 1
            max_ped_id += 1

        next_new_object_info = []
        if len(current_next_ped_id_mapping) < len(next_info):
            for det_id in next_info['detection_id']:
                if det_id not in current_next_ped_id_mapping.values():
                    next_new_object_info.append(det_id)

        for det_id_new in next_new_object_info:
            next_info.loc[next_info['detection_id'] == det_id_new, 'ped_id']= max_ped_id + 1
            max_ped_id += 1

        for cur_det_id in current_next_ped_id_mapping.keys():
            next_info.loc[next_info['detection_id'] == current_next_ped_id_mapping[cur_det_id], 'ped_id']\
                =current_info[current_info['detection_id']==cur_det_id]['ped_id'].values[0]

        miss_object_info = []
        miss_object_index=[]
        if len(current_next_ped_id_mapping) < len(current_info):
            for det_id_miss in current_info['detection_id']:
                if det_id_miss not in current_next_ped_id_mapping.keys():
                    miss_object_info.append(det_id_miss)
                    if det_id_miss in buffer_node_info['detection_id'].values: #buffer node 继续丢失
                        miss_object_index.append(-1)
                    else:
                        frame_first_object_index= current_info['detection_id'].values[len(buffer_node_info)]
                        miss_object_index.append(det_id_miss-frame_first_object_index)

        merged_info = pd.concat([merged_info,current_no_buffer_info,next_info]).drop_duplicates(
            subset=['frame', 'detection_id'], keep='last')
        merged_info.index=merged_info['detection_id'].values

        unmatched_current_nodes = current_info[current_info['detection_id'].isin(miss_object_info)].copy()
        unmatched_current_nodes['index_in_miss_frame']=miss_object_index


        delete_buffer_node_id=[]
        if len(buffer_node_info)>0:
            for id in buffer_node_info['detection_id']:
                if id not in miss_object_info:
                    delete_buffer_node_id.append(id)
        return merged_info,unmatched_current_nodes,miss_object_index,delete_buffer_node_id

    def update_node_buffer_pool(self, unmatched_current_nodes,miss_object_index,delete_buffer_node_id,buffer_len,past_frame,frame_width,frame_height):
        if unmatched_current_nodes is None or len(unmatched_current_nodes) == 0:
            self.node_buffer_pool=[]
            return

        node_num=0
        for idx, row in unmatched_current_nodes.iterrows():
            detection_id = row['detection_id']
            ped_id = row['ped_id']
            existing_index = None

            if miss_object_index[node_num]==-1:
                for i, buf_node in enumerate(self.node_buffer_pool):
                    if buf_node['detection_id'] == detection_id:
                        existing_index = i
                        break
                self.node_buffer_pool[existing_index]['unmatched_count'] = self.node_buffer_pool[existing_index]['unmatched_count']+past_frame+ 1   #可添加特征更新机制
            else:

                history = self.history_positions.get(ped_id, [])
                if len(history) >= 2:

                    last_frame_data = history[-1]
                    prev_frame_data = history[-2]
                    last_position = last_frame_data['feet']  # (feet_x, feet_y)
                    prev_position = prev_frame_data['feet']  # (feet_x, feet_y)
                    delta_t = last_frame_data['frame'] - prev_frame_data['frame']
                    if delta_t > 0:

                        vx = (last_position[0] - prev_position[0]) / delta_t
                        vy = (last_position[1] - prev_position[1]) / delta_t
                    else:
                        vx, vy = 0, 0
                else:

                    vx, vy = 0, 0
                #############

                buffer_node = {
                    'frame_path': row.get('frame_path', ''),
                    'frame': row['frame'],
                    'ped_id': row['ped_id'],
                    'bb_left': row['bb_left'],
                    'bb_top': row['bb_top'],
                    'ori_bb_left': row['bb_left'],
                    'ori_bb_top': row['bb_top'],
                    'bb_width': row['bb_width'],
                    'bb_height': row['bb_height'],
                    'bb_right': row.get('bb_right', 0.0),
                    'ori_bb_right': row.get('bb_right', 0.0),
                    'conf': row['conf'],
                    'detection_id': detection_id,
                    'unmatched_count': 1,
                    'missed_frame':row['frame'],
                    'index_in_missed_frame':miss_object_index[node_num],
                    'velocity': (vx, vy)
                }
                self.node_buffer_pool.append(buffer_node)
            node_num+=1


        indices_to_remove = []
        for i, buf_node in enumerate(self.node_buffer_pool):
            if buf_node['unmatched_count'] > buffer_len:
                indices_to_remove.append(i)

        for i in sorted(indices_to_remove, reverse=True):
            del self.node_buffer_pool[i]

        if len(delete_buffer_node_id)>0:
            for id in delete_buffer_node_id:
                self.node_buffer_pool=self.delete_node(self.node_buffer_pool,id)

    def delete_node(self,node_buffer_pool,id):
        new_list = []
        for node in node_buffer_pool:
            if node['detection_id']!=id:
                new_list.append(node)
        return new_list



    def track_adjacent_frame(self, dataset, output_path, mode='val', oracle=False,node_buffer=None):
        """
        Main tracking method. Given a dataset, track every sequence in it and create output files.
        """

        # Set the model in the test mode
        self.model.eval()
        assert not self.model.training, "Test error: Model is not in evaluation mode"

        # Disable gradients
        with torch.no_grad():

            # Separate dictionary to bookkeep the logs for each depth network
            logs_all = [{"Loss": [], "TP": [], "FP": [], "TN": [], "FN": [], "CSR": []} for i in range(self.config.hicl_depth)]


            # Loop over sequences
            for seq, seq_and_frames in dataset.sparse_frames_per_seq.items():
                print("Tracking", seq)
                merge_seqs = pd.DataFrame(columns=['frame_path', 'frame', 'ped_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'bb_right',
                             'conf', 'detection_id'])
                # Loop over each datapoint - Equivalent to using a dataloader with batch 1
                seq_dfs = []
                node_buffer_pool=[]
                for seq_name, start_frame, end_frame in seq_and_frames:
                    # Equivalent to train_batch with a single datapoint
                    data = dataset.get_graph_from_seq_and_frames(seq_name=seq_name, start_frame=start_frame, end_frame=end_frame)
                    data.to(self.config.device)

                    # Small trick to utilize torch-geometric built in functions
                    track_batch = Batch.from_data_list([data])
                    hicl_graphs = track_batch.to_data_list()

                    hicl_graphs, _, logs_all = self.hicl_forward(hicl_graphs = hicl_graphs,
                                                                 logs = logs_all,
                                                                 oracle = oracle,
                                                                 mode = mode,
                                                                 max_depth = self.config.hicl_depth,
                                                                 project_max_depth = self.config.hicl_depth)

                    # Pedestrian ids
                    ped_labels = hicl_graphs[0].get_labels()

                    # Get graph df
                    graph_df, _ = dataset.get_df_from_seq_and_frames(seq_name=seq_name, start_frame=start_frame, end_frame=end_frame)
                    assert len(ped_labels) == graph_df.shape[0], "Ped Ids Label format is wrong"

                    # Each Connected Component is a Ped Id. Assign those values to our DataFrame:
                    graph_output_df = graph_df.copy()
                    graph_output_df['ped_id'] = ped_labels
                    graph_output_df = graph_output_df[self.config.VIDEO_COLUMNS + ['conf', 'detection_id']].copy()

                    if len(merge_seqs)>0:
                        current_frame=graph_output_df['frame'][0]
                        next_frame=graph_output_df['frame'][len(graph_output_df)-1]

                        merged_info=merge_seqs[merge_seqs['frame']<current_frame]
                        previous_info=merge_seqs[merge_seqs['frame'] == current_frame]
                        current_info =graph_output_df[graph_output_df['frame'] == current_frame]
                        next_info=graph_output_df[graph_output_df['frame'] == next_frame]
                        #####


                        merge_seqs =self.merge_predictions(merged_info,previous_info,current_info,next_info)
                    else:
                        merge_seqs = graph_output_df.copy()


                    # Append the new df
                    seq_dfs.append(graph_output_df)



                seq_output_df=merge_seqs
                official_seq_merged_df = self._merge_subseq_dfs(seq_dfs)

                os.makedirs(osp.join(output_path, self.config.mot_sub_folder), exist_ok=True)
                tracking_file_path = osp.join(output_path, self.config.mot_sub_folder, seq_name + '.txt')
                self._save_results(df=seq_output_df, output_file_path=tracking_file_path)

            print("-----")
            print("Tracking completed!")

            # Print metrics
            logs_total = {}
            if mode != 'test':
                for i in range(self.config.hicl_depth):
                    # Calculate accuracy, precision, recall
                    logs = logs_all[i]
                    if logs["Loss"]:
                        print("Depth", i+1, "- Metrics:")
                        logs = self._postprocess_logs(logs=logs)

                # Total logs - Cumulative of every layer
                print("TOTAL - Metrics:")
                logs_total = {"Loss": [logs_all[i]["Loss"] for i in range(self.config.hicl_depth)],
                              "TP": [logs_all[i]["TP"] for i in range(self.config.hicl_depth)],
                              "FP": [logs_all[i]["FP"] for i in range(self.config.hicl_depth)],
                              "TN": [logs_all[i]["TN"] for i in range(self.config.hicl_depth)],
                              "FN": [logs_all[i]["FN"] for i in range(self.config.hicl_depth)],
                              "CSR": [logs_all[i]["CSR"] for i in range(self.config.hicl_depth)]}
                logs_total["Loss"] = [sum(logs_total["Loss"])]  # To be compatible with train loss (sum of hicl layers)
                logs_total = self._postprocess_logs(logs_total)

                print("-----")



        # Set the model back in training mode
        self.model.train()

        return logs_total, logs_all

    def update_node_buffer_pool_with_motion(self, start_frame,end_frame,node_buffer_pool,motion_decay_factor):
        if self.use_buffer_motion and node_buffer_pool is not None and len(node_buffer_pool) > 0:
            updated_buffer_pool = []
            for buffer_node in node_buffer_pool:

                updated_node = buffer_node.copy()

                delta_t = start_frame - buffer_node['missed_frame']
                if delta_t > 0:

                    vx, vy = buffer_node['velocity']

                    decay_rate = motion_decay_factor
                    if delta_t <= 10:

                        delta_x = vx * decay_rate * (1 - decay_rate ** delta_t) / (1 - decay_rate)
                        delta_y = vy * decay_rate * (1 - decay_rate ** delta_t) / (1 - decay_rate)
                    else:

                        delta_x = vx * decay_rate * (1 - decay_rate ** 5) / (1 - decay_rate)
                        delta_y = vy * decay_rate * (1 - decay_rate ** 5) / (1 - decay_rate)

                    updated_node['bb_left'] = updated_node['ori_bb_left'] + delta_x
                    updated_node['bb_top'] = updated_node['ori_bb_top'] + delta_y
                    updated_node['bb_right'] = updated_node['bb_left'] + buffer_node['bb_width']

                updated_buffer_pool.append(updated_node)

            node_buffer_pool = updated_buffer_pool
            return node_buffer_pool
        ################################################

    def track_adjacent_frame_with_buffer(self, dataset, output_path, mode='val', oracle=False,node_buffer=None,buffer_len=None):
        """
        Main tracking method. Given a dataset, track every sequence in it and create output files.
        """

        # Set the model in the test mode
        self.model.eval()
        assert not self.model.training, "Test error: Model is not in evaluation mode"

        # Disable gradients
        with torch.no_grad():

            # Separate dictionary to bookkeep the logs for each depth network
            logs_all = [{"Loss": [], "TP": [], "FP": [], "TN": [], "FN": [], "CSR": []} for i in range(self.config.hicl_depth)]
            seq_max_id = {}
            # Loop over sequences
            for seq, seq_and_frames in dataset.sparse_frames_per_seq.items():
                print("Tracking", seq)
                merge_seqs = pd.DataFrame(columns=['frame_path', 'frame', 'ped_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'bb_right',
                             'conf', 'detection_id'])
                # Loop over each datapoint - Equivalent to using a dataloader with batch 1
                seq_dfs = []
                self.node_buffer_pool = []
                seq_start_time=time.time()
                for seq_name, start_frame, end_frame in seq_and_frames:
                    # Equivalent to train_batch with a single datapoint
                    start_time=time.time()

                    #################  update node_buffer_pool with motion
                    if self.use_buffer_motion and self.node_buffer_pool is not None and len(self.node_buffer_pool) > 0:
                        self.node_buffer_pool = self.update_node_buffer_pool_with_motion(start_frame,end_frame,self.node_buffer_pool,self.motion_decay_factor)
                    #################

                    data,graph_df,graph_df_with_buffer = dataset.get_graph_from_seq_and_frames_with_buffer(seq_name=seq_name, start_frame=start_frame, end_frame=end_frame,node_buffer_pool=self.node_buffer_pool)



                    data.to(self.config.device)

                    # Small trick to utilize torch-geometric built in functions
                    track_batch = Batch.from_data_list([data])
                    hicl_graphs = track_batch.to_data_list()

                    hicl_graphs, _, logs_all = self.hicl_forward(hicl_graphs = hicl_graphs,
                                                                 logs = logs_all,
                                                                 oracle = oracle,
                                                                 mode = mode,
                                                                 max_depth = self.config.hicl_depth,
                                                                 project_max_depth = self.config.hicl_depth)

                    # Pedestrian ids
                    ped_labels = hicl_graphs[0].get_labels()

                    # Get graph df
                    if graph_df_with_buffer is not None:
                        assert len(ped_labels) == graph_df_with_buffer.shape[0], "Ped Ids Label format is wrong"
                        graph_df_with_buffer['ped_id'] = ped_labels
                        graph_output_df=graph_df_with_buffer

                    else:
                        assert len(ped_labels) == graph_df.shape[0], "Ped Ids Label format is wrong"
                        graph_output_df = graph_df.copy()
                        graph_output_df['ped_id'] = ped_labels

                    current_frame = graph_output_df['frame'].iloc[0]
                    next_frame = graph_output_df['frame'].iloc[-1]
                    current_info = graph_output_df[graph_output_df['frame'] == current_frame]
                    next_info = graph_output_df[graph_output_df['frame'] == next_frame]
                    past_frame = 0
                    if current_frame==1:
                        merge_seqs, unmatched_current_nodes,miss_object_index = self.merge_first_frame_info(current_info,next_info)
                        delete_buffer_node_id=[]
                    else:
                        merged_info = merge_seqs[merge_seqs['frame'] < current_frame]
                        previous_frame=max(merge_seqs['frame'].unique())
                        if previous_frame==current_frame:
                            previous_info= merge_seqs[merge_seqs['frame'] == current_frame]
                            merge_seqs, unmatched_current_nodes, miss_object_index, delete_buffer_node_id = self.merge_predictions_with_buffer(
                                merged_info,
                                previous_info,
                                current_info,
                                next_info)
                        else:
                            past_frame = current_frame-previous_frame
                            merge_seqs, unmatched_current_nodes, miss_object_index, delete_buffer_node_id = self.merge_occlusion(
                                merged_info,
                                current_info,
                                next_info,graph_df)


                    if self.use_buffer_motion:
                        self.update_history(merge_seqs)

                    frame_width=dataset.seq_info_dicts[seq_name]['frame_width']
                    frame_height=dataset.seq_info_dicts[seq_name]['frame_height']

                    self.update_node_buffer_pool(unmatched_current_nodes,miss_object_index,delete_buffer_node_id,buffer_len,past_frame,
                                                 frame_width,frame_height)

                    end_time=time.time()
                    cost_time=(end_time-start_time)*1000

                    #######
                    remaining_time = (len(seq_and_frames)-end_frame)*cost_time/1000 #
                    remaining_time = timedelta(seconds=int(remaining_time))
                    hours = remaining_time.seconds // 3600
                    minutes = (remaining_time.seconds % 3600) // 60
                    #######

                    print("Tracking Seq:{} Frame: {} in {} ms, remaining time is {} hours {} minutes".format(seq_name,end_frame,cost_time, hours, minutes))

                seq_end_time=time.time()
                seq_tracking_time=seq_end_time-seq_start_time

                print("The Seq: {} Tracking in {}s ".format(seq_name,seq_tracking_time))

                seq_output_df=merge_seqs

                os.makedirs(osp.join(output_path, self.config.mot_sub_folder), exist_ok=True)
                tracking_file_path = osp.join(output_path, self.config.mot_sub_folder, seq_name + '.txt')
                max_id=self._save_results(df=seq_output_df, output_file_path=tracking_file_path)
                seq_max_id[seq_name]=max_id
            print("-----")
            print("Tracking completed!")

            ###############
            max_id_file_path=osp.join(output_path, 'max_id.txt')
            with open(max_id_file_path, 'w') as file:
                for key, value in seq_max_id.items():
                    file.write(f"{key}: {value}\n")
            ###############
            # Print metrics
            logs_total = {}
            if mode != 'test':
                for i in range(self.config.hicl_depth):
                    # Calculate accuracy, precision, recall
                    logs = logs_all[i]
                    if logs["Loss"]:
                        print("Depth", i+1, "- Metrics:")
                        logs = self._postprocess_logs(logs=logs)

                # Total logs - Cumulative of every layer
                print("TOTAL - Metrics:")
                logs_total = {"Loss": [logs_all[i]["Loss"] for i in range(self.config.hicl_depth)],
                              "TP": [logs_all[i]["TP"] for i in range(self.config.hicl_depth)],
                              "FP": [logs_all[i]["FP"] for i in range(self.config.hicl_depth)],
                              "TN": [logs_all[i]["TN"] for i in range(self.config.hicl_depth)],
                              "FN": [logs_all[i]["FN"] for i in range(self.config.hicl_depth)],
                              "CSR": [logs_all[i]["CSR"] for i in range(self.config.hicl_depth)]}
                logs_total["Loss"] = [sum(logs_total["Loss"])]  # To be compatible with train loss (sum of hicl layers)
                logs_total = self._postprocess_logs(logs_total)

                print("-----")



        # Set the model back in training mode
        self.model.train()

        return logs_total, logs_all

    def _merge_subseq_dfs_adjacent_frame(self, subseq_dfs):
        """
        Experimental merge dfs copied from MPNTrack. Might not be optimized. Check its behavior.

            Algorithm:
            Create a df consisting of ids from the left, ids from the right and how many times they match with
            each other based on the detection id.

            Create a cost matrix that has NaN everywhere except the matched ids. The cost at these locations are
            -(num_matched). E.g: if id 1 and 100 matched 10 times -> cost_matrix[1, 100] = -10

            Solve the cost matrix and find the assignments

            Merge both dataframes and replace the larger of the matched ids with the smaller one

        """
        seq_df = subseq_dfs[0]
        for subseq_df in subseq_dfs[1:]:
            # Make sure that ped_ids in subseq_df are new and all greater than the ones in seq_df:
            subseq_df['ped_id'] += seq_df['ped_id'].max() + 1

            intersect_frames = np.intersect1d(seq_df.frame, subseq_df.frame)  # Common frames between 2 dfs

            # Detections in common frames within seq_df
            left_df = seq_df[['detection_id', 'ped_id']][seq_df.frame.isin(intersect_frames)]
            left_ids_pos = left_df[['ped_id']].drop_duplicates();
            left_ids_pos['ped_id_pos'] = np.arange(left_ids_pos.shape[0])
            left_df = left_df.merge(left_ids_pos, on='ped_id').set_index('detection_id')

            # Detections in common frames within subseq_df
            right_df = subseq_df[['detection_id', 'ped_id']][subseq_df.frame.isin(intersect_frames)]
            right_ids_pos = right_df[['ped_id']].drop_duplicates();
            right_ids_pos['ped_id_pos'] = np.arange(right_ids_pos.shape[0])
            right_df = right_df.merge(right_ids_pos, on='ped_id').set_index('detection_id')

            # Count how many times each left_id corresponds to right_id (based on detection_id)
            common_boxes = \
                left_df[['ped_id_pos']].join(right_df['ped_id_pos'], lsuffix='_left', rsuffix='_right').dropna(
                    thresh=2).reset_index().groupby(['ped_id_pos_left', 'ped_id_pos_right'])['detection_id'].count()
            common_boxes = common_boxes.reset_index().astype(int)

            # Create a cost matrix with negative count (more match, less cost). Everywhere else is NaN
            cost_mat = np.full((common_boxes['ped_id_pos_left'].max() + 1, common_boxes['ped_id_pos_right'].max() + 1),
                               fill_value=np.nan)
            cost_mat[common_boxes['ped_id_pos_left'].values, common_boxes['ped_id_pos_right'].values] = - common_boxes[
                'detection_id'].values

            # Find the min cost solution
            matched_left_ids_pos, matched_right_ids_pos = solve_dense(cost_mat)

            # Map of the matched ids
            matched_ids = pd.DataFrame(data=np.stack((left_ids_pos['ped_id'].values[matched_left_ids_pos],
                                                      right_ids_pos['ped_id'].values[matched_right_ids_pos])).T,
                                       columns=['left_ped_id', 'right_ped_id'])

            # Assign the ids matched to subseq_df
            subseq_df = pd.merge(subseq_df, matched_ids, how='outer', left_on='ped_id', right_on='right_ped_id')
            # subseq_df['left_ped_id'].fillna(np.inf, inplace=True)
            subseq_df['left_ped_id'] = subseq_df['left_ped_id'].fillna(np.inf)
            subseq_df['ped_id'] = np.minimum(subseq_df['left_ped_id'], subseq_df['ped_id'])

            # Update seq_df
            seq_df = pd.concat([seq_df, subseq_df[subseq_df['frame'] > seq_df['frame'].max()]])
        return seq_df.reset_index(drop=True)

    def _merge_subseq_dfs(self, subseq_dfs):
        """
        Experimental merge dfs copied from MPNTrack. Might not be optimized. Check its behavior.

            Algorithm:
            Create a df consisting of ids from the left, ids from the right and how many times they match with
            each other based on the detection id.

            Create a cost matrix that has NaN everywhere except the matched ids. The cost at these locations are
            -(num_matched). E.g: if id 1 and 100 matched 10 times -> cost_matrix[1, 100] = -10

            Solve the cost matrix and find the assignments

            Merge both dataframes and replace the larger of the matched ids with the smaller one

        """
        seq_df = subseq_dfs[0]
        for subseq_df in subseq_dfs[1:]:

            # Make sure that ped_ids in subseq_df are new and all greater than the ones in seq_df:
            subseq_df['ped_id'] += seq_df['ped_id'].max() + 1

            intersect_frames = np.intersect1d(seq_df.frame, subseq_df.frame)  # Common frames between 2 dfs

            # Detections in common frames within seq_df
            left_df = seq_df[['detection_id', 'ped_id']][seq_df.frame.isin(intersect_frames)]
            left_ids_pos = left_df[['ped_id']].drop_duplicates();
            left_ids_pos['ped_id_pos'] = np.arange(left_ids_pos.shape[0])
            left_df = left_df.merge(left_ids_pos, on='ped_id').set_index('detection_id')

            # Detections in common frames within subseq_df
            right_df = subseq_df[['detection_id', 'ped_id']][subseq_df.frame.isin(intersect_frames)]
            right_ids_pos = right_df[['ped_id']].drop_duplicates();
            right_ids_pos['ped_id_pos'] = np.arange(right_ids_pos.shape[0])
            right_df = right_df.merge(right_ids_pos, on='ped_id').set_index('detection_id')

            # Count how many times each left_id corresponds to right_id (based on detection_id)
            common_boxes = \
                left_df[['ped_id_pos']].join(right_df['ped_id_pos'], lsuffix='_left', rsuffix='_right').dropna(
                    thresh=2).reset_index().groupby(['ped_id_pos_left', 'ped_id_pos_right'])['detection_id'].count()
            common_boxes = common_boxes.reset_index().astype(int)

            # Create a cost matrix with negative count (more match, less cost). Everywhere else is NaN
            cost_mat = np.full((common_boxes['ped_id_pos_left'].max() + 1, common_boxes['ped_id_pos_right'].max() + 1),
                               fill_value=np.nan)
            cost_mat[common_boxes['ped_id_pos_left'].values, common_boxes['ped_id_pos_right'].values] = - common_boxes[
                'detection_id'].values

            # Find the min cost solution
            matched_left_ids_pos, matched_right_ids_pos = solve_dense(cost_mat)

            # Map of the matched ids
            matched_ids = pd.DataFrame(data=np.stack((left_ids_pos['ped_id'].values[matched_left_ids_pos],
                                                      right_ids_pos['ped_id'].values[matched_right_ids_pos])).T,
                                       columns=['left_ped_id', 'right_ped_id'])

            # Assign the ids matched to subseq_df
            subseq_df = pd.merge(subseq_df, matched_ids, how='outer', left_on='ped_id', right_on='right_ped_id')

            subseq_df['left_ped_id'] = subseq_df['left_ped_id'].fillna(np.inf)
            subseq_df['ped_id'] = np.minimum(subseq_df['left_ped_id'], subseq_df['ped_id'])

            # Update seq_df
            seq_df = pd.concat([seq_df, subseq_df[subseq_df['frame'] > seq_df['frame'].max()]])
        return seq_df.reset_index(drop=True)

    def _calculate_true_false_metrics(self, edge_preds, edge_labels, logs):
        """
        Calculate TP, FP, TN, FN
        """

        # edge_preds needs to be already after a sigmoid
        preds = (edge_preds.view(-1) > 0.5).float()

        # Metrics
        TP = ((edge_labels == 1) & (preds == 1)).sum().float()
        FP = ((edge_labels == 0) & (preds == 1)).sum().float()
        TN = ((edge_labels == 0) & (preds == 0)).sum().float()
        FN = ((edge_labels == 1) & (preds == 0)).sum().float()

        # Update the logs
        logs["TP"].append(TP.item())
        logs["FP"].append(FP.item())
        logs["TN"].append(TN.item())
        logs["FN"].append(FN.item())

        return logs

    def _postprocess_logs(self, logs):
        """
        Calculate accuracy, precision, recall
        """
        logs["Loss"] = statistics.mean(logs["Loss"])
        logs["TP"] = sum(logs["TP"])
        logs["FP"] = sum(logs["FP"])
        logs["TN"] = sum(logs["TN"])
        logs["FN"] = sum(logs["FN"])

        logs["Accuracy"] = (logs["TP"] + logs["TN"]) / (logs["TP"] + logs["FP"] + logs["TN"] + logs["FN"])
        logs["Recall"] = logs["TP"] / (logs["TP"] + logs["FN"]) if logs["TP"] + logs["FN"] > 0 else 0
        logs["Precision"] = logs["TP"] / (logs["TP"] + logs["FP"]) if logs["TP"] + logs["FP"] > 0 else 0
        logs["F1"] = 2*logs["TP"] / (2*logs["TP"] + logs['FP']+ logs['FN']) if (logs["TP"] + logs["FP"] +logs['FN']) > 0 else 0

        if logs["CSR"]:
            logs["CSR"] = statistics.mean(logs["CSR"])
        else:
            logs["CSR"] = math.nan

        # Verbose
        print("     Loss: ", logs["Loss"])
        print("     Accuracy: ", logs["Accuracy"])
        print("     Recall: ", logs["Recall"])
        print("     Precision: ", logs["Precision"])
        print("     Constraint Satisfaction Rate: ", logs["CSR"])
        print("     TP+FP+TN+FN: ", int((logs["TP"] + logs["FP"] + logs["TN"] + logs["FN"])))

        return logs

    def _plot_losses(self, logs):
        """
        Plot training and validation losses
        """
        plt.figure(figsize=(12, 9))
        plt.plot(logs["Train_Loss"], label='Training Loss')

        if logs["Val_Loss"]:
            plt.plot(logs["Val_Loss"], label='Validation Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(len(logs["Train_Loss"])), np.arange(1, len(logs["Train_Loss"]) + 1))
        plt.savefig(osp.join(self.config.experiment_path, 'loss_plots-' + self.train_split + '.png'), bbox_inches='tight')
        plt.close()

    def _plot_losses_per_depth(self, logs):
        """
        Plot training and validation losses per depth
        """
        plt.figure(figsize=(12, 9))
        for j in range(self.config.hicl_depth):
            plt.plot(logs["Train_Loss_per_Depth"][j], label=f'Training Loss - Depth {j}')
            if logs["Val_Loss"]:
                plt.plot(logs["Val_Loss_per_Depth"][j], label=f'Validation Loss - Depth {j}')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(len(logs["Train_Loss"])), np.arange(1, len(logs["Train_Loss"]) + 1))
        plt.savefig(osp.join(self.config.experiment_path, 'loss_plots_per_depth-' + self.train_split + '.png'), bbox_inches='tight')
        plt.close()

    def _plot_metrics(self, logs):
        """
        Plot validation metrics
        """
        plt.figure(figsize=(12, 9))
        plt.plot(logs["Val_HOTA"], label='Validation HOTA')
        plt.plot(logs["Val_MOTA"], label='Validation MOTA')
        plt.plot(logs["Val_IDF1"], label='Validation IDF1')

        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(len(logs["Train_Loss"])), np.arange(1, len(logs["Val_HOTA"]) + 1))

        plt.savefig(osp.join(self.config.experiment_path, 'val_metrics-' + self.train_split + '-' + self.val_split + '.png'), bbox_inches='tight')

    def _save_results(self, df, output_file_path):
        """
        Save the tracking output df
        """
        df['conf'] = 1
        df['x'] = -1
        df['y'] = -1
        df['z'] = -1
        df['ped_id'] = df['ped_id'].astype('int64')

        # Coordinates are 1 based - revert
        df['bb_left'] += 1
        df['bb_top'] += 1

        final_out = df[self.config.TRACKING_OUT_COLS].sort_values(by=['frame', 'ped_id'])
        final_out.to_csv(output_file_path, header=False, index=False)
        return df['ped_id'].max()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


