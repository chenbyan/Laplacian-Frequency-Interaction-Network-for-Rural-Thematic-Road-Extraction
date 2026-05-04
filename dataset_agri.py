import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
import graph_utils
import rtree
import scipy
import pickle
import os
import glob

def read_rgb_img(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"无法读取图片: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def agriculture_data_partition():

    root_dir = './dataset_agri/new_agri'

    def get_basenames(split):
        search_path = os.path.join(root_dir, split, 'rgb', '*_sat.png')
        files = glob.glob(search_path)
        return [os.path.basename(f).replace('_sat.png', '') for f in files]

    train_list = get_basenames('train')
    val_list = get_basenames('val')  
    test_list = get_basenames('test')

    print(f"✅ 农机数据集加载: Train={len(train_list)}, Val={len(val_list)},Test={len(test_list)}")
    return train_list, val_list, test_list

def get_patch_info_one_img(image_index, image_size, sample_margin, patch_size, patches_per_edge):
    patch_info = []
    if patches_per_edge <= 0:
        return []
        
    sample_min = sample_margin
    sample_max = image_size - (patch_size + sample_margin)
    
    if sample_max <= sample_min:
        patch_info.append((image_index, (0, 0), (image_size, image_size)))
        return patch_info

    eval_samples = np.linspace(start=sample_min, stop=sample_max, num=patches_per_edge)
    eval_samples = [round(x) for x in eval_samples]
    
    for x in eval_samples:
        for y in eval_samples:
            patch_info.append(
                (image_index, (x, y), (x + patch_size, y + patch_size))
            )
    return patch_info


class GraphLabelGenerator():
    def __init__(self, config, full_graph, coord_transform):
        self.config = config
        
        if isinstance(full_graph, dict) and 'node_coords' in full_graph:
            # 说明是我们新生成的格式
            coords = full_graph['node_coords']
            edges = full_graph['edges']
            

            adj_dict = {}
            for i, (x, y) in enumerate(coords):
                adj_dict[(x, y)] = []
            
            for u, v in edges:
                u_pos = tuple(coords[u])
                v_pos = tuple(coords[v])
                adj_dict[u_pos].append(v_pos)
                adj_dict[v_pos].append(u_pos) # 无向图
            
            full_graph = adj_dict
        
        self.full_graph_origin = graph_utils.igraph_from_adj_dict(full_graph, coord_transform)
        self.crossover_points = graph_utils.find_crossover_points(self.full_graph_origin)
        self.subdivide_resolution = 4
        self.full_graph_subdivide = graph_utils.subdivide_graph(self.full_graph_origin, self.subdivide_resolution)
        self.subdivide_points = np.array(self.full_graph_subdivide.vs['point'])
        self.graph_rtee = rtree.index.Index()
        for i, v in enumerate(self.subdivide_points):
            x, y = v
            self.graph_rtee.insert(i, (x, y, x, y))
        
        # 防止空图报错
        if len(self.subdivide_points) > 0:
            self.graph_kdtree = scipy.spatial.KDTree(self.subdivide_points)
        else:
            self.graph_kdtree = None

        crossover_exclude_radius = 4
        exclude_indices = set()
        if self.graph_kdtree:
            for p in self.crossover_points:
                nearby_indices = self.graph_kdtree.query_ball_point(p, crossover_exclude_radius)
                exclude_indices.update(nearby_indices)
        self.exclude_indices = exclude_indices

        itsc_indices = set()
        point_num = len(self.full_graph_subdivide.vs)
        for i in range(point_num):
            if self.full_graph_subdivide.degree(i) != 2:
                itsc_indices.add(i)
        self.nms_score_override = np.zeros((point_num, ), dtype=np.float32)
        self.nms_score_override[np.array(list(itsc_indices))] = 2.0 

        interesting_indices = set()
        interesting_radius = 32
        if self.graph_kdtree:
            for i in itsc_indices:
                p = self.subdivide_points[i]
                nearby_indices = self.graph_kdtree.query_ball_point(p, interesting_radius)
                interesting_indices.update(nearby_indices)
            for p in self.crossover_points:
                nearby_indices = self.graph_kdtree.query_ball_point(np.array(p), interesting_radius)
                interesting_indices.update(nearby_indices)
        self.sample_weights = np.full((point_num, ), 0.1, dtype=np.float32)
        self.sample_weights[list(interesting_indices)] = 0.9
    
    def sample_patch(self, patch, rot_index = 0):
        (x0, y0), (x1, y1) = patch
        query_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        patch_indices_all = set(self.graph_rtee.intersection(query_box))
        patch_indices = patch_indices_all - self.exclude_indices

        patch_indices = np.array(list(patch_indices))
        if len(patch_indices) == 0:
            sample_num = self.config.TOPO_SAMPLE_NUM
            max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES
            fake_points = np.array([[0.0, 0.0]], dtype=np.float32)
            fake_sample = ([[0, 0]] * max_nbr_queries, [False] * max_nbr_queries, [False] * max_nbr_queries)
            return fake_points, [fake_sample] * sample_num

        patch_points = self.subdivide_points[patch_indices, :]
        
        nms_scores = np.random.uniform(low=0.9, high=1.0, size=patch_indices.shape[0])
        nms_score_override = self.nms_score_override[patch_indices]
        nms_scores = np.maximum(nms_scores, nms_score_override)
        nms_radius = self.config.ROAD_NMS_RADIUS
        
        nmsed_points, kept_indices = graph_utils.nms_points(patch_points, nms_scores, radius=nms_radius, return_indices=True)
        nmsed_indices = patch_indices[kept_indices]
        nmsed_point_num = nmsed_points.shape[0]

        sample_num = self.config.TOPO_SAMPLE_NUM 
        sample_weights = self.sample_weights[nmsed_indices]
        
        # Fix: avoid sum=0 error
        if np.sum(sample_weights) == 0:
            p_weights = None
        else:
            p_weights = sample_weights / np.sum(sample_weights)

        sample_indices_in_nmsed = np.random.choice(
            np.arange(start=0, stop=nmsed_points.shape[0], dtype=np.int32),
            size=sample_num, replace=True, p=p_weights)
        sample_indices = nmsed_indices[sample_indices_in_nmsed]
        
        radius = self.config.NEIGHBOR_RADIUS
        max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES
        nmsed_kdtree = scipy.spatial.KDTree(nmsed_points)
        sampled_points = self.subdivide_points[sample_indices, :]
        knn_d, knn_idx = nmsed_kdtree.query(sampled_points, k=max_nbr_queries + 1, distance_upper_bound=radius)

        samples = []

        for i in range(sample_num):
            source_node = sample_indices[i]
            # Fix: handle cases where knn returns index >= nmsed_point_num (not found)
            valid_nbr_mask = knn_idx[i, :] < nmsed_point_num
            valid_nbr_indices = knn_idx[i, valid_nbr_mask]
            
            # remove self
            if len(valid_nbr_indices) > 0 and valid_nbr_indices[0] == i: # usually index 0 is self
                 valid_nbr_indices = valid_nbr_indices[1:]
            # simple filter, ensure we don't include self
            valid_nbr_indices = [idx for idx in valid_nbr_indices if nmsed_indices[idx] != source_node]

            target_nodes = [nmsed_indices[ni] for ni in valid_nbr_indices]  

            reached_nodes = graph_utils.bfs_with_conditions(self.full_graph_subdivide, source_node, set(target_nodes), radius // self.subdivide_resolution)
            shall_connect = [t in reached_nodes for t in target_nodes]

            pairs = []
            valid = []
            source_nmsed_idx = sample_indices_in_nmsed[i]
            for target_nmsed_idx in valid_nbr_indices:
                pairs.append((source_nmsed_idx, target_nmsed_idx))
                valid.append(True)

            for i in range(len(pairs), max_nbr_queries):
                pairs.append((source_nmsed_idx, source_nmsed_idx))
                shall_connect.append(False)
                valid.append(False)

            samples.append((pairs, shall_connect, valid))

        nmsed_points -= np.array([x0, y0])[np.newaxis, :]
        nmsed_points = np.concatenate([nmsed_points, np.ones((nmsed_point_num, 1), dtype=nmsed_points.dtype)], axis=1)
        trans = np.array([
            [1, 0, -0.5 * self.config.PATCH_SIZE],
            [0, 1, -0.5 * self.config.PATCH_SIZE],
            [0, 0, 1],
        ], dtype=np.float32)
        rot = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        nmsed_points = nmsed_points @ trans.T @ np.linalg.matrix_power(rot.T, rot_index) @ np.linalg.inv(trans.T)
        nmsed_points = nmsed_points[:, :2]
            
        noise_scale = 1.0 
        nmsed_points += np.random.normal(0.0, noise_scale, size=nmsed_points.shape)

        return nmsed_points, samples

def graph_collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if key == 'graph_points':
            tensors = [item[key] for item in batch]
            max_point_num = max([x.shape[0] for x in tensors])
            padded = []
            for x in tensors:
                pad_num = max_point_num - x.shape[0]
                padded_x = torch.concat([x, torch.zeros(pad_num, 2)], dim=0)
                padded.append(padded_x)
            collated[key] = torch.stack(padded, dim=0)
        else:
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
    return collated

class SatMapDataset(Dataset):
    def __init__(self, config, is_train, dev_run=False):
        self.config = config

        if self.config.DATASET == 'agriculture':
            self.IMAGE_SIZE = 256
            self.SAMPLE_MARGIN = 0 
            rgb_pattern = './dataset_agri/new_agri/rgb/{}_sat.png'
            keypoint_mask_pattern = './dataset_agri/new_agri/keypoint/{}_keypoint.png'
            road_mask_pattern = './dataset_agri/new_agri/mask/{}_mask.png'
            gt_graph_pattern = './dataset_agri/new_agri/graph/{}_refine_gt_graph.p'
            
            train, val, test = agriculture_data_partition()
    
            coord_transform = lambda v : v 

        self.is_train = is_train

        train_split = [os.path.join('train', t) for t in train]
        val_split   = [os.path.join('val', v) for v in val]  
        test_split  =[os.path.join('test', e) for e in test] 

        tile_indices = train_split if self.is_train else val_split
        self.tile_indices = tile_indices


        
        self.rgbs, self.keypoint_masks, self.road_masks = [], [], []
        self.graph_label_generators = []

        if dev_run:
            tile_indices = tile_indices[:4]

        print(f"Dataset 初始化: 正在加载 {len(tile_indices)} 张图片到内存...")
        
        for tile_idx in tile_indices:
            if isinstance(tile_idx, str) and '/' in tile_idx:
                split, name = tile_idx.split('/', 1)
            else:
                split, name = '', tile_idx

            candidates = []
            if split:
                candidates.append({
                    'rgb': os.path.join('./dataset_agri/new_agri', split, 'rgb', f'{name}_sat.png'),
                    'road_mask': os.path.join('./dataset_agri/new_agri', split, 'mask', f'{name}_mask.png'),
                    'keypoint': os.path.join('./dataset_agri/new_agri', split, 'keypoint', f'{name}_keypoint.png'),
                    'gt_graph': os.path.join('./dataset_agri/new_agri', split, 'graph', f'{name}_refine_gt_graph.p')
                })

            chosen = None
            for cand in candidates:
                if os.path.exists(cand['gt_graph']):
                    chosen = cand
                    break

            if chosen is None:
                print(f'Warning: 缺少 gt_graph 文件，跳过 tile {tile_idx}. Checked candidates:')
                for cand in candidates:
                    print('  -', cand['gt_graph'])
                continue

            try:
                with open(chosen['gt_graph'], 'rb') as f:
                    gt_graph_adj = pickle.load(f)
            except Exception as e:
                print(f'Error loading gt_graph for tile {tile_idx}: {e}. Skipping.')
                continue

            if gt_graph_adj is None or (isinstance(gt_graph_adj, dict) and len(gt_graph_adj) == 0):
                print(f'===== skipped empty tile {tile_idx} =====')
                continue

            rgb_path = chosen['rgb']
            road_mask_path = chosen['road_mask']
            keypoint_mask_path = chosen['keypoint']

            if not os.path.exists(rgb_path):
                print(f'Warning: rgb not found: {rgb_path}. Skipping tile {tile_idx}')
                continue
            if not os.path.exists(road_mask_path):
                print(f'Warning: road mask not found: {road_mask_path}. Skipping tile {tile_idx}')
                continue
            if not os.path.exists(keypoint_mask_path):
                print(f'Warning: keypoint not found: {keypoint_mask_path}. Skipping tile {tile_idx}')
                continue

            self.rgbs.append(read_rgb_img(rgb_path))
            self.road_masks.append(cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE))
            self.keypoint_masks.append(cv2.imread(keypoint_mask_path, cv2.IMREAD_GRAYSCALE))

            graph_label_generator = GraphLabelGenerator(config, gt_graph_adj, coord_transform)
            self.graph_label_generators.append(graph_label_generator)


        if not self.is_train:
            if self.IMAGE_SIZE == self.config.PATCH_SIZE:
                 self.eval_patches = []
                 for i in range(len(tile_indices)):
                     self.eval_patches.append((i, (0,0), (512,512)))
            else:
                eval_patches_per_edge = math.ceil((self.IMAGE_SIZE - 2 * self.SAMPLE_MARGIN) / self.config.PATCH_SIZE)
                self.eval_patches = []
                for i in range(len(tile_indices)):
                    self.eval_patches += get_patch_info_one_img(
                        i, self.IMAGE_SIZE, self.SAMPLE_MARGIN, self.config.PATCH_SIZE, eval_patches_per_edge
                    )

    def __len__(self):
        if self.is_train:
            return len(self.rgbs) 
        else:
            return len(self.eval_patches)

    def __getitem__(self, idx):
        if self.is_train:
            img_idx = np.random.randint(low=0, high=len(self.rgbs))
            
            if self.IMAGE_SIZE == self.config.PATCH_SIZE:
                begin_x, begin_y = 0, 0
            else:
                begin_x = np.random.randint(low=self.sample_min, high=self.sample_max+1)
                begin_y = np.random.randint(low=self.sample_min, high=self.sample_max+1)
            
            end_x, end_y = begin_x + self.config.PATCH_SIZE, begin_y + self.config.PATCH_SIZE
        else:
            img_idx, (begin_x, begin_y), (end_x, end_y) = self.eval_patches[idx]
        
        rgb_patch = self.rgbs[img_idx][begin_y:end_y, begin_x:end_x, :]
        keypoint_mask_patch = self.keypoint_masks[img_idx][begin_y:end_y, begin_x:end_x]
        road_mask_patch = self.road_masks[img_idx][begin_y:end_y, begin_x:end_x]

        rot_index = 0
        if self.is_train:
            rot_index = 0
            # CCW
            rgb_patch = np.rot90(rgb_patch, rot_index, [0,1]).copy()
            keypoint_mask_patch = np.rot90(keypoint_mask_patch, rot_index, [0, 1]).copy()
            road_mask_patch = np.rot90(road_mask_patch, rot_index, [0, 1]).copy()
        
        patch = ((begin_x, begin_y), (end_x, end_y))
        graph_points, topo_samples = self.graph_label_generators[img_idx].sample_patch(patch, rot_index)
        
        pairs, connected, valid = zip(*topo_samples)
        
        return {
            'rgb': torch.tensor(rgb_patch, dtype=torch.float32),
            'keypoint_mask': torch.tensor(keypoint_mask_patch, dtype=torch.float32) / 255.0,
            'road_mask': torch.tensor(road_mask_patch, dtype=torch.float32) / 255.0,
            
            'graph_points': torch.tensor(graph_points, dtype=torch.float32),
            'pairs': torch.tensor(pairs, dtype=torch.int32),
            'connected': torch.tensor(connected, dtype=torch.bool),
            'valid': torch.tensor(valid, dtype=torch.bool),
        }

if __name__ == '__main__':

    pass