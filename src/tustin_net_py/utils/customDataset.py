from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, U, Y):
        self.U = U
        self.Y = Y

    def __len__(self):
        return len(self.U)
    
    def __getitem__(self, idx):
        return self.U[idx], self.Y[idx]