from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

device="cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(device)

class BertCstm(nn.Module):

    def __init__(self,bert, hid_size, out_slot, out_int, dout=0.3):
        super(BertCstm, self).__init__()
        
        self.bert = bert
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.slt_hid = nn.Linear(hid_size,hid_size)
        self.slt_hid2 = nn.Linear(hid_size,hid_size)
        self.intent_out = nn.Linear(hid_size, out_int)
        self.dropout = nn.Dropout(dout)
        self.d2 = nn.Dropout(dout+0.1)
        self.bn = nn.BatchNorm1d(hid_size)
        self.bn2 = nn.BatchNorm1d(hid_size)
        self.m = nn.Mish(inplace=True)
        
    def forward(self, utterance,attm):

        bx = self.bert(utterance,attention_mask=attm)

        bx = self.dropout(bx.last_hidden_state)                
            
        slots = self.slt_hid(bx).permute(0,2,1)
        slots = self.bn(slots).permute(0,2,1)
        slots = self.m(slots)
        slots = self.d2(slots)
        slots = self.slt_hid2(slots).permute(0,2,1)
        slots = self.bn2(slots).permute(0,2,1)
        slots = self.m(slots)
        slots = self.d2(slots)
        
        slots = self.slot_out(slots).permute(0,2,1)#bx.pooler_output)
        # Compute intent logits
        intent = self.intent_out(bx[:,0,:])
        
        
        return slots, intent