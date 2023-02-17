import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from os import listdir
from os.path import join, isfile
import pickle
import torch_geometric

from torch_geometric.data import Data, InMemoryDataset, DataLoader

class HexDataset(InMemoryDataset):
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(HexDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['processed_data.dataset']
    
    def download(self):
        # Download to `self.raw_dir`.
        pass
    
    def process(self):

        data_list = []
        
        files = listdir('./data_new/hex-logs')
#         files = listdir('./local_data/hex-logs') #for testing
        count = 0
        print(len(files))
        for i in range(len(files)):
            f = files[i]
            filepath = join('./data_new/hex-logs',f)
#             filepath = join('./local_data/hex-logs',f) #for testing
            if isfile(filepath):
            
                if count == 1000:
                    break
                count += 1
            
                print("File: " + str(i))
                
                try:
                    fileData = pickle.load(open(filepath, 'rb'))
                except EOFError as e:
                    print(filepath)
                        
                traj = fileData['traj']
                nHis = fileData['nHis']
                acc = fileData['acc']
                
                nAgents = len(traj)      # 20 agents
                nSteps = len(traj[0])    # 250 steps (25 seconds)

                for j in range(nSteps-1):

                    d1 = [] # current positions
                    d2 = [] # plus one step
                    e1 = [] # edge from
                    e2 = [] # edge to
                    
                    d11 = []
                    d12 = []
                    d21 = []
                    d22 = []
                    
                    att = []

                    for k in range(nAgents):
                        d2 = [] # plus one step
                        focal=traj[k]
                        focal_y=acc[k]
                        graph_init=[[0,0]]
                        graph_init_y=[[focal_y[j][0],focal_y[j][1]]]
                        e=[]
                        e1 = [] # edge from
                        e2 = []
                        for l in range(nAgents):
                            neighbor=traj[l]
#                             neighbor_y=acc[l]
                            if k==l:
                                continue
                            e1.extend([0, len(graph_init)])
                            e2.extend([len(graph_init), 0])
                            graph_init.append([neighbor[j][0]-focal[j][0], neighbor[j][1]-focal[j][1]])
#                             graph_init_y.append([neighbor_y[j][0]-focal_y[j][0], neighbor_y[j][1]-focal_y[j][1]])
                                
#                         v1 = traj[k][j][0]
#                         v2 = traj[k][j][1]
#                         d1.append([v1, v2])
#                         d11.append(v1)
#                         d12.append(v2)
                        
#                         v3 = traj[k][j+1][0]
#                         v4 = traj[k][j+1][1]
                        d2.append(acc[k][j][:2]) # get only the x and y accel
                        
                        x = torch.tensor(graph_init, dtype=torch.float)
                        e = torch.tensor(e, dtype=torch.long)
                        e = torch.tensor([e1, e2], dtype=torch.long)
                        a = torch.tensor(att, dtype=torch.float)
#                         y = torch.tensor(graph_init_y, dtype=torch.float)
                        y = torch.tensor(d2, dtype=torch.float)
#                         print(graph_init_y)
#                         print("y",y)
                        data_list.append(Data(x=x, edge_index=e, edge_attr=None, y=y))



        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
#103->

#                         e1.extend([k for _ in range(len(nHis[k][j]))])
#                         for val in nHis[k][j]:
#                             try:
#                                 e2.append(val[0])
#                                 att.append(val[1])
#                             except TypeError as e:
#                                 e2.append(val-1)
#                                 att.append([1])

#                     e = []
#                     local_x=[]
#                     local_y=[]
                    #converting global to local(wrt to each other)
#                     print(len(d1[0]))
#                     for i in range(len(d1)):
#                         temp_x=d1[i][0]
#                         temp_y=d1[i][1]
# #                         print(temp_x)
#                         curr_local_x=[[x-temp_x,y-temp_y] for x,y in d1]
#                         local_x.extend(curr_local_x)
#                         curr_local_y=[[x-temp_x,y-temp_y] for x,y in d2]
#                         local_y.extend(curr_local_y)

#                     for k in range(len(e1)):
#                         e.append([e1[k], e2[k]])
#                     e=np.repeat(e, len(d1)-1)
#                     print(e)
#                     x = torch.tensor(d1, dtype=torch.float)
               