import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_min, scatter_max, scatter_sum

#
def masked2original(tensor, num_nodes, nodes_mask, fill_value=0):
    assert nodes_mask.shape[0] == num_nodes
    assert nodes_mask.sum() == tensor.shape[0]
    orig_shape = [num_nodes] + list(tensor.shape[1:]) #全节点矩阵 包括忽略的节点
    full_tensor = torch.full(orig_shape, device = tensor.device, dtype=tensor.dtype, fill_value=fill_value)
    full_tensor[nodes_mask] = tensor
    return full_tensor

def ccwh2tlbr(boxes):
    """
    Change box parametrization from (center_x, center_y, width, height) to (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    boxes: Tensor with shape (N, 4)
    """
    assert boxes.dim() == 2 and boxes.shape[-1] == 4
    tl = boxes[:, :2] - 0.5*boxes[:, 2:]
    br = boxes[:, :2]  + 0.5*boxes[:, 2:]
    
    return torch.cat((tl, br), dim = 1)

def compute_iou(output, target):
    """
    GIoU computation, adapted from https://pytorch.org/vision/stable/_modules/torchvision/ops/boxes.html#generalized_box_iou
    output and target are tensors with shape (4, N), representing bounding boxes parametrized by their
    top-left and bottom-right corner.
    """
    assert output.dim() == 2 and output.shape[0] == 4
    assert target.dim() == 2 and target.shape[0] == 4

    x1, y1, x2, y2 = output
    x1g, y1g, x2g, y2g = target 

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(output.device)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
    miouk = iouk - ((area_c - unionk) / area_c)

    return iouk, miouk
def compute_giou_fwrd_bwrd_motion_sim(curr_batch, m_pred, edge_index=None):
    """For each edge in the graph, given two trajectories (nodes), 
    find the middle time-point between them, collect the predicted backward/forward position for 
    those trajectories, and compute their Generalized IoU"""

    has_motion_feats = ~curr_batch.x_ignore_traj
    fwrd_pred_, bwrd_pred_ = m_pred
    fwrd_pred_, bwrd_pred_ = fwrd_pred_['pred_pos'].detach(), bwrd_pred_['pred_pos'].detach()
    fwrd_pred = masked2original(fwrd_pred_, curr_batch.num_nodes, has_motion_feats, np.nan)  #构建全节点矩阵 包括忽略节点
    bwrd_pred = masked2original(bwrd_pred_, curr_batch.num_nodes, has_motion_feats, np.nan)

    if edge_index is None:
        edge_index = curr_batch.edge_index
    #row, col = raw_batch.edge_index
    row, col = edge_index
    assert (row < col).all()

    m_pred_horizon=fwrd_pred.shape[1]  #预测的帧数
    t_diff = (curr_batch.x_frame_start[col] - curr_batch.x_frame_end[row]).reshape(-1)  #计算帧间隔
    replace = t_diff == 1   #标记时间差为 1 的边
    t_diff_ = torch.minimum(t_diff.float() / 2, torch.as_tensor(m_pred_horizon))      #时间差归一化 最大不超过 m_pred_horizon  除以2是取中间节点更好评估轨迹时空相似性
    t_diff_fwrd = torch.ceil(t_diff_).long() - 1
    t_diff_bwrd = torch.maximum(torch.floor(t_diff_).long(), torch.as_tensor(1)) -1

    assert (t_diff_bwrd >=0).all()
    assert (t_diff_fwrd >=0).all()

    # Forward/bwrd pred
    #对于每个边，获取对应时间步 t_diff_fwrd 和 t_diff_bwrd 的 xywh
    fwrd_start_box = fwrd_pred[row].gather(1, t_diff_fwrd[:,None, None].expand(-1, -1, 4))[:, 0]
    bwrd_end_box = bwrd_pred[col].gather(1, t_diff_bwrd[:,None, None].expand(-1, -1, 4))[:, 0]

    _, gious = compute_iou(ccwh2tlbr(fwrd_start_box).T, ccwh2tlbr(bwrd_end_box).T)  # xywh->xyxy后进行计算IoU
    ##### 起始节点在前向预测和后向预测的中间点的 box 计算 GIoU
    # When the time distance is 1, we instead avg pred + real location for fwr/bwrd  对时间差为 1的边进行前后向GIoU计算
    _, gious_rep_fwrd = compute_iou(ccwh2tlbr(fwrd_start_box[replace]).T, ccwh2tlbr(curr_batch.x_center_start[col][replace]).T)
    _, gious_rep_bwrd = compute_iou(ccwh2tlbr(bwrd_end_box[replace]).T, ccwh2tlbr(curr_batch.x_center_end[row][replace]).T)
    gious[replace] = (gious_rep_bwrd + gious_rep_fwrd) / 2
    #return torch.nan_to_num(gious, -1).cuda()
    return torch.clamp(torch.nan_to_num(gious, -2), min=-2, max=1).cuda()  #约束在[-2,1]

#计算运动特征xywh
def torch_get_motion_feats(map_from_init, x_frame_, x_center, x_bbox, max_length, interpolate):
    x_frame = x_frame_.view(-1)
    box_coords = torch.cat((x_center, x_bbox[:, 2:]), dim=1)
    # box_coords: xywh    x_bbox:tlwh

    # Strategy: map frames from their value to positions within the trajectory
    # Then directly index an array with shape (num_ids, max_trajectory_length) using those
    num_ids = map_from_init.max() + 1

    # Forward x_motion
    device = box_coords.device
    x_fwrd_bwrd_motion = []
    keep_traj = torch.ones(num_ids, device=device, dtype=torch.bool)
    for mode in ['forward', 'backward']:
        scatter_fn = scatter_max if mode == 'forward' else scatter_min  #前向轨迹找到每个节点的最后帧,后向轨迹找到每个节点的最早帧
        last_frame, _ = scatter_fn(x_frame, index=map_from_init, dim=0, dim_size = num_ids)

        # Frame index WITHIN trajectory with positions from 0 until max_length - 1
        if mode == 'forward':
            norm_frame = x_frame - last_frame[map_from_init] + max_length - 1  #归一化的时间帧索引，将帧号映射到[0, max_length-1]范围内
        
        else:
            norm_frame = -x_frame + last_frame[map_from_init] + max_length - 1
            # e.g. 30, 31, 32, 33, 38 and max_length=4 -->  3, 2, 1, 0

        mask = (norm_frame >= 0) #确保索引在合法范围内

        # Assign Values  包括当前帧信息
        x_motion_ = torch.full((num_ids, max_length, 4), fill_value = np.nan, device=device)
        x_motion_[map_from_init[mask], norm_frame[mask]] = box_coords[mask]  #按照归一化索引，将节点的box_coords分配到对应的轨迹位置
        
        # Ignore trajectories that do not have more than one past box
        count_per_id = scatter_sum(mask.int(), index=map_from_init, dim=0, dim_size = num_ids)
        keep_traj = keep_traj & (count_per_id>1)

        if interpolate: # tricky to vectorize
            # Determine which values need to be interpolated (dont extrapolate starts of trajectories)
            has_nans = torch.isnan(x_motion_)
            cumsum = has_nans[..., 0].cumsum(dim=1)
            track_idx = torch.arange(max_length, device=device).view(-1, max_length).expand(cumsum.shape[0], -1)

            needs_interp = has_nans[..., 0] & ((cumsum - 1) < track_idx)

            # Main strategy: flatten coords columns to make the interpolation 1d
            # then call np.interp on the result and unflatten
            flatten_ = lambda x: x.permute(0, 2, 1).flatten()    #插值只能处理一维数据，需要将三维张量展平为一维
            unflatten_ = lambda x: x.reshape(x_motion_.shape[0], 4, max_length).permute(0, 2, 1)  #插值完成后，再将数据重新恢复为原始形状

            flat_x_motion = flatten_(x_motion_) #[num_nodes * max_length * 4]
            interp_idxs = torch.where(flatten_(needs_interp[..., None].expand(-1, -1, 4)))[0]
            given_idxs = torch.where(flatten_(~has_nans))[0]
            given_vals = flat_x_motion[given_idxs]
    ##      ######## np.interp()线性插值函数    interp_idxs：需要插值的位置    given_idxs：已知值的位置    given_vals：已知值  interp_vals：插值后的值，与 interp_idxs 对应
            interp_vals = np.interp(interp_idxs.cpu().numpy(), given_idxs.cpu().numpy(), given_vals.cpu().numpy())
            flat_x_motion[interp_idxs] = torch.as_tensor(interp_vals, device=device, dtype=flat_x_motion.dtype)   #将插值结果写回到展平的运动轨迹张量中
            x_motion_ = unflatten_(flat_x_motion)   #还原为三维张量


        x_fwrd_bwrd_motion.append(x_motion_)

        #feat_names = ['x_fwrd_rel_vel', 'x_bwrd_rel_vel', 'x_fwrd_motion', 'x_fwrd_length', 'x_bwrd_motion', 'x_bwrd_length', 'x_ignore_traj', 'x_fwrd_missing', 'x_bwrd_missing']

    return dict(x_fwrd_motion=x_fwrd_bwrd_motion[0][keep_traj],
                x_bwrd_motion=x_fwrd_bwrd_motion[1][keep_traj],
                x_ignore_traj = ~keep_traj
                )