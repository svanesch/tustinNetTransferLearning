import torch

def Loss(prediction, target):
    pos_diff = torch.cos(prediction[..., 0:2] - target[..., 0:2])           # shape: (batch_size, 2)

    loss = torch.mean(2*(1 - pos_diff))

    return loss


def LossVel(prediction, target):
    pos_diff = torch.cos(prediction[..., 0:2] - target[..., 0:2])           # shape: (batch_size, 2)

    loss = torch.mean(2*(1 - pos_diff) + (prediction[..., 2:4] - target[..., 2:4])**2) 

    return loss