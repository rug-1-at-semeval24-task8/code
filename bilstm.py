import torch
from torch.nn import LSTM, Linear, LogSoftmax, Module, NLLLoss, Dropout
import torch.nn.functional as F


class Attention(Module):
    def __init__(self, hidden_size, dropout):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.attn_weights = Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        attn_energies = self.attn_weights(lstm_output).squeeze(2)
        attn_weights = F.softmax(attn_energies, dim=1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return attn_output


class BiLSTM(Module):
    def __init__(
        self,
        feature_len,
        doc_feature_len,
        task,
        local_device,
        device,
        hidden_size,
        num_lstm_layers,
        dropout_prop,
        attention_enabled,
    ):
        super(BiLSTM, self).__init__()
        self.lstm_layer = LSTM(
            input_size=feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
            num_layers=num_lstm_layers,
        )

        if attention_enabled:
            self.linear_layer = Linear(
                hidden_size * 2 + doc_feature_len, 2 if task == "A" else 6
            )
        else:
            self.linear_layer = Linear(
                hidden_size * 2 * num_lstm_layers + doc_feature_len,
                2 if task == "A" else 6,
            )
        self.dropout = Dropout(dropout_prop)
        self.attention = Attention(hidden_size, self.dropout)
        self.softmax_layer = LogSoftmax(1)
        self.loss_fn = NLLLoss()

        self.local_device = local_device
        self.device = device
        self.attention_enabled = attention_enabled

    def forward(self, x, doc_feats):
        if not self.attention_enabled:
            _, (hidden_state, _) = self.lstm_layer(x)
            hidden_state = self.dropout(hidden_state)
            transposed = torch.transpose(hidden_state, 0, 1)
            reshaped = torch.reshape(transposed, (transposed.shape[0], -1))
        else:
            lstm_output, _ = self.lstm_layer(x)
            reshaped = self.attention(lstm_output)

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
