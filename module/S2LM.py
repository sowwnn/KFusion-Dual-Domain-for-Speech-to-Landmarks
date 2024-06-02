import lightning as L
import torch.nn as nn
from torchmetrics.regression import CosineSimilarity
import torch

class S2LM(L.LightningModule):
    def __init__(self, batch):
        super(S2LM, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=3200, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.linear = nn.Linear(in_features=3200, out_features=956)
        decoder_layer = nn.TransformerDecoderLayer(d_model=956, nhead=2)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.conv = nn.Conv1d(in_channels=1, out_channels=60, kernel_size=1)

        self.loss = nn.MSELoss()
        self.batch = batch


    def forward(self, x, v):

        v = v.view(4,1, -1)
        v = self.conv(v.float())


        x = x.view(self.batch, 60, -1)
        x = self.transformer_encoder(x)
        x = self.linear(x)
        x = torch.softmax(x, dim=-1)
        x = self.transformer_decoder(x, v)
        y = x.view(self.batch, 2, 60, -1)
        return y
    
    def training_step(self, batch, batch_idx):
        x = batch['audio']
        y = batch['target']
        v = batch['ilm']
        y_ = self.forward(x, v)
        loss = self.loss(y_.float(), y.float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['audio']
        y = batch['target']
        v = batch['ilm']

        y_ = self.forward(x,v )
        loss = self.loss(y_.float(), y.float())
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    