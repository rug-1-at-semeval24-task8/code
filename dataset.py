from torch.utils.data import Dataset, DataLoader


class TensorTextDataset(Dataset):
    def __init__(self, tensor, text, labels):
        self.tensor = tensor
        self.text = text
        self.labels = labels

    def __len__(self):
        """
        This is simply the number of labels in the dataseta.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Generate one sample of data
        """
        label = self.labels[idx]
        text = self.text[idx]
        tensor = self.tensor[idx]
        sample = {"Text": text, "Label": label, "Tensor" : tensor}
        return sample