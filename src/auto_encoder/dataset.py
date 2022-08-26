import torch
from torch.utils.data import Dataset


class AutoEncoderDataset(Dataset):
    def __init__(self, all_data, frames_count=3):
        self.x = []
        self.frames_count = frames_count

        for i in range(len(all_data)):
            self.x.append(torch.tensor(all_data[i]))

    def __len__(self):
        return len(self.x) - (self.frames_count - 1)

    def __getitem__(self, index):
        return (torch.cat(self.x[index: index + self.frames_count]),
                torch.cat(self.x[index: index + self.frames_count]))
