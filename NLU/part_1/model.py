import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, out_dropout=0.1):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        #bidirectional:
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)
        #self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=False, batch_first=True)
        #??come ridimensionare la matrice sum hidden state
        self.slot_out = nn.Linear(hid_size*2, out_slot)
        # Dropout layer How/Where do we apply it? --> what's happen here???
        self.dropout = nn.Dropout(out_dropout)

        self.intent_out = nn.Linear(hid_size, out_int)


    def forward(self, utterance, seq_lengths):

        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        print("Dimensione dell'embedding:", utt_emb.size())
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost

        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)
        print("Dimensione dell'output LSTM prima del padding:", packed_output.data.size())

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        print("Dimensione dell'output LSTM dopo il padding:", utt_encoded.size())
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]
        print("Dimensione dell'ultimo hidden state:", last_hidden.size())

        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]

        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        print("Dimensione dell'output dei slot:", slots.size())
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        print("Dimensione dell'output dell'intent:", intent.size())

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
