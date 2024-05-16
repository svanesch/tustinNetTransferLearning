import torch
import numpy as np
import pandas as pd

from utils.customDataset import CustomDataset

def LoadData(files, seq_len, step_size, batch_size, truncation=False):
    
    u_subsequences = []
    y_subsequences = []
    
    theta_scaled = []
    alpha_scaled = []

    for file in files:
        data = pd.read_csv(file, header=None, names=['voltage', 'theta', 'alpha', 'timestamp', 'thetad', 'alphad'])

        dfU = data['voltage']
        dfY = data[['theta', 'alpha', 'thetad', 'alphad']]

        uTensor = torch.tensor(dfU.values).to(torch.float32) 
        uTensor = torch.reshape(uTensor, (len(uTensor), 1))

        yTensor = torch.tensor(dfY.values).to(torch.float32)
        yTensor = torch.reshape(yTensor, (len(yTensor), 4))

        theta_mean = torch.mean(torch.abs(yTensor[:, 0]))
        thetadot_mean = torch.mean(torch.abs(yTensor[:, 2]))
        theta_scale = thetadot_mean / theta_mean
        theta_scaled.append(theta_scale)

        alpha_mean = torch.mean(torch.abs(yTensor[:, 1]))
        alphadot_mean = torch.mean(torch.abs(yTensor[:, 3]))
        alpha_scale = alphadot_mean / alpha_mean
        alpha_scaled.append(alpha_scale)

        if truncation:
            for i in range(0, round((len(data) - seq_len)/4), step_size):
                u_subsequences.append(uTensor[i:i+seq_len])
                y_subsequences.append(yTensor[i:i+seq_len])
        else:
            for i in range(0, len(data) - seq_len, step_size):
                u_subsequences.append(uTensor[i:i+seq_len])
                y_subsequences.append(yTensor[i:i+seq_len])
            

    u_subsequences = torch.stack(u_subsequences, dim=0)             # shape (batch_size, seq_len, 1)
    y_subsequences = torch.stack(y_subsequences, dim=0)             # shape (batch_size, seq_len, 4)

    data = CustomDataset(U=u_subsequences, Y=y_subsequences)

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    return data, loader, torch.mean(torch.stack(theta_scaled, dim=0)), torch.mean(torch.stack(alpha_scaled), dim=0)