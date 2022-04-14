import sys; sys.path.append('.')
import pickle
import torch
import numpy as np
from src.trajnet.reader import Reader
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class CFF6(Dataset):
    def __init__(self, path_x, path_y, windows_size):
        '''
        Input: 
            path: <str> path of single pkl file (for a single scene);
        '''
        super(CFF6, self).__init__()
        with open(path_x, 'rb') as f:  
            self.traj_list_x = pickle.load(f)
        with open(path_y, 'rb') as f:  
            self.traj_list_y = pickle.load(f)
        self.traj_keys = list(self.traj_list_x.keys())
        self.len_traj = len(self.traj_list_x[self.traj_keys[0]])
        self.windows_size = windows_size
        self.num_node = []
        self.node_list = []
        self.adj_list = []
        self.edg_list = []
        self.pos_list = []
        for traj_idx in range(self.len_traj):
            num_node = 0
            node_list = []
            pos_list = []
            for key in self.traj_keys:
                x = self.traj_list_x[key][traj_idx] 
                y = self.traj_list_y[key][traj_idx] 
                if x != 0 and y != 0: 
                    num_node += 1
                    node_list.append(key)
                    pos_list.append([x, y])
                    
            self.num_node.append(num_node) 
            self.node_list.append(node_list)
            self.adj_list.append(torch.ones(num_node, num_node))
            self.edg_list.append(self.create_edge(node_list))
            self.pos_list.append(torch.tensor(pos_list))
        # self.pos_list = torch.tensor(self.pos_list)

    def __len__(self):
        return self.len_traj - self.windows_size + 1

    def __getitem__(self, index):
        '''
        Return: 
            adj: <torch.tensor> [windows_size, #node, #node] adjanency matrix;
            edg: <torch.tensor> [windows_size, 2, #edge] edge list;
            pos: <torch.tensor> [windows_size, #node_dy, h_dim] node position list, node_dy means the number of node at that frame;
        '''
        return self.adj_list[index:index+self.windows_size], self.edg_list[index:index+self.windows_size], self.pos_list[index:index+self.windows_size]

    def create_edge(self, node_list):
        edg_list = []
        for i in node_list:
            for j in node_list:
                edg_list.append([i, j])
        return torch.tensor(edg_list)

    def create_pos(self):
        pos_list = []
        for traj_idx in range(self.len_traj):
            pos_list_traj = []
            for node_idx in range(self.num_node):
                pos = [self.traj_list_x[self.traj_keys[node_idx]][traj_idx], self.traj_list_y[self.traj_keys[node_idx]][traj_idx]]
                if pos[0] == 0 or pos[1] == 0: continue
                pos_list_traj.append(pos)
            pos_list.append(pos_list_traj)
        pos_list = torch.tensor(pos_list)

        # Normalize
        max_x = torch.max(pos_list[:, :, 0])
        min_x = torch.min(pos_list[:, :, 0])
        max_y = torch.max(pos_list[:, :, 1])
        min_y = torch.min(pos_list[:, :, 1])
        pos_list[:, :, 0] = (pos_list[:, :, 0] - min_x)/(max_x - min_x)
        pos_list[:, :, 1] = (pos_list[:, :, 1] - min_y)/(max_y - min_y)
        return pos_list
