from torch.utils.data import Dataset, DataLoader


class TensorTextDataset(Dataset):
    def __init__(self, tensor, labels, doc_feats):
        self.tensor = tensor
        self.labels = labels
        self.doc_feats = doc_feats

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
        doc_feats = self.doc_feats[idx]
        tensor = self.tensor[idx]
        sample = {"DocFeats": doc_feats, "Label": label, "Tensor" : tensor}
        return sample