import torch
from torch.nn import LSTM, Linear, LogSoftmax, Module, NLLLoss


class BiLSTM(Module):
    def __init__(self, feature_len, doc_feature_len, task, local_device, device):
        super(BiLSTM, self).__init__()
        self.lstm_layer = LSTM(
            input_size=feature_len, hidden_size=64, batch_first=True, bidirectional=True
        )
        self.linear_layer = Linear(
            self.lstm_layer.hidden_size * 2 + doc_feature_len, 2 if task == "A" else 6
        )
        self.softmax_layer = LogSoftmax(1)
        self.loss_fn = NLLLoss()
        self.local_device = local_device
        self.device = device

    def forward(self, x, doc_feats):
        _, (hidden_state, _) = self.lstm_layer(x)
        transposed = torch.transpose(hidden_state, 0, 1)
        reshaped = torch.reshape(transposed, (transposed.shape[0], -1))

        # Concatenate document features with reshaped hidden state
        if not torch.is_tensor(doc_feats):
            doc_feats = torch.from_numpy(doc_feats)
        if doc_feats.get_device() == -1:  # -1 means CPU
            doc_feats = doc_feats.to(self.device, dtype=torch.float32)
        reshaped = torch.cat((reshaped, doc_feats), 1)

        scores = self.linear_layer(reshaped)
        logprobabilities = self.softmax_layer(scores)
        return logprobabilities

    def compute_loss(self, pred, true):
        output = self.loss_fn(pred, true)
        return output

    def postprocessing(self, Y, argmax=True):
        if argmax:
            decisions = Y.argmax(1).to(self.local_device).numpy()
        else:
            decisions = Y.to(self.local_device).numpy()
        return decisions
