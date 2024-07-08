import lightning as L
import torch.nn as nn
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class S2LM(L.LightningModule):
    def __init__(self, batch):
        super(S2LM, self).__init__()

        self.batch = batch
        drop_prob = 0.2
        output_size = 68
        hidden_size = 256
        self.MOUTH_LANDMARKS =[49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 48]


        self.dropout = nn.Dropout(p=drop_prob)
        
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-100h")
        self.lstm_f = nn.LSTM(input_size=768, hidden_size=hidden_size, num_layers=8, batch_first=True, bidirectional=True, dropout=drop_prob)
        self.lstm_m = nn.LSTM(input_size=768, hidden_size=hidden_size, num_layers=8, batch_first=True, bidirectional=True, dropout=drop_prob)

        self.conv_f = nn.Conv1d(in_channels= 99, out_channels=60, kernel_size=1)

        self.output_mapper = nn.Linear(hidden_size*2, output_size)
        self.output_mapper_m = nn.Linear(hidden_size*2, len(self.MOUTH_LANDMARKS))

        self.loss = nn.MSELoss()
        self.init_lr = 3e-1


    def forward(self, x):
        x = x.view(4,-1)

        x = self.audio_encoder(x).last_hidden_state
        x = self.conv_f(x)

        # x = self.sigmoid(x)

        v, _ = self.lstm_m(x)
        x, _ = self.lstm_f(x)

        v = self.output_mapper_m(v)
        x = self.output_mapper(x)

        v = v.view(4,2,30,-1)
        x = x.view(4,2,30,-1)

        y = x.clone()
        y[:,:,:,self.MOUTH_LANDMARKS] = v
        return y
    
    def training_step(self, batch, batch_idx):
        x = batch['audio']
        y = batch['target']
        y_ = self.forward(x)
        loss = self.loss(y_.float(), y.float())
        # m_loss = self.loss(y_[:,:,:,self.MOUTH_LANDMARKS].float(), y[:,:,:,self.MOUTH_LANDMARKS].float())
        # loss = tt_loss*0.3 + m_loss*0.7
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['audio']
        y = batch['target']

        y_ = self.forward(x)
        loss = self.loss(y_.float(), y.float())
        # m_loss = self.loss(y_[:,:,:,[self.MOUTH_LANDMARKS]].float(), y[:,:,:,[self.MOUTH_LANDMARKS]].float())
        # loss = tt_loss*0.3 + m_loss*0.7
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss_epoch'  
        }