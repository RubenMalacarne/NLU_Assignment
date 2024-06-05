
import torch.nn as nn
import torch

from transformers import BertTokenizer, BertModel

device="cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(device)

class BertCstm(nn.Module):

    def __init__(
        self,
        hid_size,
        out_slot,
        out_int,
        emb_size,
        vocab_len,
        n_layer=1,
        pad_index=0,
        bidirectional=False,
        dropout=None,):
        super(BertCstm, self).__init__()

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(
            emb_size, hid_size, n_layer, bidirectional=bidirectional
        )

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.slot_out = nn.Linear(self.bert.config.hidden_size, out_slot)

        self.intent_out = nn.Linear(self.bert.config.hidden_size, out_int)

        # Dropout layer How do we apply it?
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, utterance, utterance_mask):

        bert_out = self.bert(utterance, attention_mask=utterance_mask)
        sequence_output = bert_out[0]
        pooled_output = bert_out[1]

        if self.dropout:
            sequence_output = self.dropout(sequence_output)
            pooled_output = self.dropout(pooled_output)

        # Compute slot logits
        slots = self.slot_out(sequence_output)
        # Compute intent logits
        intent = self.intent_out(pooled_output)

        # Slot size: seq_len, batch size, calsses
        slots = slots.permute(0, 2, 1)  # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len

        # remove first and last token from slots
        slots = slots[:, :, 1:-1]

        return slots, intent


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

