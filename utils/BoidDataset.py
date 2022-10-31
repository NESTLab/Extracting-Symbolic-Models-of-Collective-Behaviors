import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from os import listdir
from os.path import join, isfile
import pickle
import torch_geometric
import math

from torch_geometric.data import Data, InMemoryDataset, DataLoader

class BoidDataset(InMemoryDataset):
    
    def __init__(self, root, getXData=True, transform=None, pre_transform=None):
        super(BoidDataset, self).__init__(root, transform, pre_transform)        
        if getXData:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['processed_data_x.dataset', 'processed_data_y.dataset']
    
    def download(self):
        # Download to `self.raw_dir`.
        pass
    
    def process(self):
        nAgents = 50
        
        x_data_list = []
        y_data_list = []
        
        files = listdir('./data/boid-logs/')
        count = 0
        minX = 10000
        minY = 10000
        maxX = -10000
        maxY = -10000
        minXD = 100000
        minYD = 100000
        maxXD = -100000
        maxYD = -100000
        minD = 1000000
        maxD = -1000000
        minRX = 1000000
        maxRX = -1000000
        minRY = 1000000
        maxRY = -1000000
        for i in range(int(len(files))):
            f = files[i]
            filepath = join('./data/boid-logs/',f)
            if isfile(filepath):
            
                if count == 1000:
                    break
                count += 1
            
                print("File: " + str(i))
                
                dataList = []
                
                with open(filepath, 'r') as f:
                    for line in f.readlines():
                        d = line.strip().split(",")
                        step = int(d[0])
                        rid = int(d[1])
                        xVal = float(d[2])
                        yVal = float(d[3])
                        xaVal = float(d[4])
                        yaVal = float(d[5])
                        xRes = float(d[6])
                        yRes = float(d[7])
                        
                        agentData = [xVal, yVal, xaVal, yaVal, xRes, yRes]
                                              
                        if step+1 > len(dataList):
                            dataList.append([])
                        dataList[step].append(agentData)
                    
                    for i in range(len(dataList)-1):
                        
                        d1 = []
                        d2 = []
                        for j in range(len(dataList[i+1])):
                            d1.append([1.])
                            d2.append([dataList[i][j][4], dataList[i][j][5]])
                            
                        for j in range(len(dataList[i])):
                            
                            gnn = [[1.]]
                            e1 = []
                            e2 = []
                            att = []
                            xRes = [[dataList[i][j][4]/4.0 + 0.5]]
                            yRes = [[dataList[i][j][5]/8.0 + 0.5]]
                            
                            count = 1
                            
                            for k in range(len(dataList[i])):
                                if j!=k:
                                    
                                    visRange=100
                                    MAXSPEED=3
                                    desiredSep=35
                                    
                                    x1 = dataList[i][j][0]
                                    x2 = dataList[i][k][0]
                                    y1 = dataList[i][j][1]
                                    y2 = dataList[i][k][1]
                                    dx1 = dataList[i][j][2]
                                    dx2 = dataList[i][k][2]
                                    dy1 = dataList[i][j][3]
                                    dy2 = dataList[i][k][3]
                                    s1 = math.sqrt(dx1*dx1 + dy1*dy1)
                                    
                                    lFX = (x2-x1)*(dx1/s1) - (y2-y1)*(dy1/s1)
                                    lFY = (x2-x1)*(dy1/s1) + (y2-y1)*(dx1/s1)
                                    lFXD = dx2-dx1
                                    lFYD = dy2-dy1
                                    d = math.sqrt(pow(lFX, 2) + pow(lFY,2))
                                    s = math.sqrt(pow(lFXD, 2) + pow(lFYD,2))
                                    invd = 1.0 / d
                                    invs = 0
                                    if s != 0:
                                        invs = 1.0 / s
                                    lFnX = lFX * invd
                                    lFnY = lFY * invd
                                    lFnXD = lFXD * invs
                                    lFnYD = lFYD * invs
                                    
                                    dS = desiredSep
                                    
                                    # NORMALIZE
                                    #lFX /= visRange
                                    #lFY /= visRange
                                    #d /= visRange
                                    #dS /= visRange
                                    #lFXD /= (2*MAXSPEED)
                                    #lFYD /= (2*MAXSPEED)
                                    #
                                    #if speed == 0:
                                    #    invSpeed = 0
                                    #    normlFXD = 0
                                    #    normlFYD = 0
                                    #else:
                                    #    invSpeed = 1.0 / speed
                                    #    normlFXD = lFXD / speed
                                    #    normlFYD = lFYD / speed
                                    
                                    if math.sqrt(pow(x2-x1, 2) + pow(y2-y1,2)) <= visRange:
                                        
                                        a = (lFX + 1) / 2
                                        b = (lFY + 1) / 2
                                        c = (lFXD + 1) / 2
                                        f = (lFYD + 1) / 2
                                        
                                        minX = min(minX, a)
                                        minY = min(minY, b)
                                        maxX = max(maxX, a)
                                        maxY = max(maxY, b)
                                        minXD = min(minXD, c)
                                        minYD = min(minYD, f)
                                        maxXD = max(maxXD, c)
                                        maxYD = max(maxYD, f)
                                        minD = min(minD, d)
                                        maxD = max(maxD, d)
                                        
                                        gnn.append([1.])
                                        xRes.append([0.])
                                        yRes.append([0.])
                                        e1.append(0)
                                        e2.append(count)
                                        count += 1
                                        
                                        #att.append([a, b, c, f, d])                                    
                                        att.append([lFX, lFY, d, invd, lFnX, lFnY, lFXD, lFYD, s, invs, lFnXD, lFnYD])
                                        
                            x = torch.tensor(gnn, dtype=torch.float)
                            e = torch.tensor([e2, e1], dtype=torch.long)
                            a = torch.tensor(att, dtype=torch.float)
                            y = torch.tensor(xRes, dtype=torch.float)
                            y2 = torch.tensor(yRes, dtype=torch.float)
                            
                            if 0 < xRes[0][0] < 1:                                                        
                                x_data_list.append(Data(x=x, edge_index=e, edge_attr=a, y=y))
                                minRX = min(xRes[0][0], minRX)
                                maxRX = max(xRes[0][0], maxRX)                            
                            if 0 < yRes[0][0] < 1:  
                                y_data_list.append(Data(x=x, edge_index=e, edge_attr=a, y=y2))
                                minRY = min(yRes[0][0], minRY)
                                maxRY = max(yRes[0][0], maxRY)
        
        print("X: [" + str(minX) + " .. " + str(maxX) + "]")
        print("Y: [" + str(minY) + " .. " + str(maxY) + "]")
        print("XD: [" + str(minXD) + " .. " + str(maxXD) + "]")
        print("YD: [" + str(minYD) + " .. " + str(maxYD) + "]")
        print("d: [" + str(minD) + " .. " + str(maxD) + "]")
        print("RX: [" + str(minRX) + " .. " + str(maxRX) + "]")
        print("RY: [" + str(minRY) + " .. " + str(maxRY) + "]")
        xdata, xslices = self.collate(x_data_list)
        ydata, yslices = self.collate(y_data_list)
        torch.save((xdata, xslices), self.processed_paths[0])
        torch.save((ydata, yslices), self.processed_paths[1])