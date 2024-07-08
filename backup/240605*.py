import torch
import torch.nn as nn
import torchaudio
import pytorch_lightning as L

class Transformer(L.LightningModule):
    def __init__(self, in_c, out_c, nhead, num_lay=2, sub_chanel=49,):
        super(Transformer, self).__init__()

        torch.autograd.set_detect_anomaly(True)

        en_lay = nn.TransformerEncoderLayer(d_model=in_c, nhead=nhead, dropout=0.2)
        self.transformer_encoder = nn.Sequential(
            nn.TransformerEncoder(en_lay, num_layers=num_lay),
            nn.Linear(in_features=in_c, out_features=out_c*2)
        )
        de_lay = nn.TransformerDecoderLayer(d_model=out_c*2, nhead=nhead, dropout=0.2)
        self.transformer_decoder = nn.TransformerDecoder(de_lay, num_layers=num_lay)
        self.conv = nn.Conv1d(1, sub_chanel, 1,1)

    def forward(self, x, v):
        x = self.transformer_encoder(x)
        v = self.conv(v)
        x = self.transformer_decoder(v, x)
        return x


class S2LM(L.LightningModule):
    def __init__(self, batch, init_lr, num_of_landmarks=478):
        super(S2LM, self).__init__()

        torch.autograd.set_detect_anomaly(True)

        self.MOUTH_LANDMARKS = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 48]
        self.init_lr = init_lr
        self.batch = batch
        self.loss = nn.MSELoss()
        self.LD = LandmarkDistance()
        self.LVD = LandmarkVelocityDifference()
        self.w2v = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        self.w2v.eval()
        for param in self.w2v.parameters():
            param.requires_grad = False

        self.trans_ct = Transformer(in_c=768, out_c=num_of_landmarks, nhead=1, num_lay=1)
        self.trans_gb = Transformer(in_c=768, out_c=num_of_landmarks, nhead=2, num_lay=1)
        self.trans_gbm = Transformer(in_c=768, out_c=len(self.MOUTH_LANDMARKS), nhead=2, num_lay=1)

        self.lstm = nn.LSTM(input_size=768, hidden_size=68, num_layers=1, batch_first=True, bidirectional=True, dropout=0.3)
        self.mouth = nn.LSTM(input_size=768, hidden_size=len(self.MOUTH_LANDMARKS), num_layers=1, batch_first=True, bidirectional=True, dropout=0.3)

        self.R1 = nn.Sequential(
            nn.Conv2d(147, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 30, 3, 1, 1)
        )
        self.R2 = nn.Sequential(nn.Conv2d(147, 30, 1, 1))
                                
        self.R3 = nn.Sequential(
            nn.Conv2d(98, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 30, 3, 1, 1)
        )
        self.KAN = KAN([68, 512, 68])

    def forward(self, x, v, w):

        x = self.w2v(x)[0]
        vm = v[:,:,self.MOUTH_LANDMARKS].view(self.batch, 1, -1)
        v = v.view(self.batch, 1, -1)
        gbm = self.trans_gbm(x, vm)
        gb = self.trans_gb(x, v)

        with torch.no_grad():
            w = nn.functional.softmax(w.float(), dim=0)
            w = w[:, None, None]
            x = torch.mul(x, w)

        f = self.lstm(x)[0]
        m = self.mouth(x)[0]
        x = self.trans_ct(x, v)

        x = torch.cat([f, gb, x], dim=1)
        m = torch.cat([m, gbm], dim=1)

        x = x.view(self.batch, -1, 2, 68)
        m = m.view(self.batch, -1, 2, 20)


        m = self.R3(m)
        t = self.R2(x)
        x = self.R1(x)
        x *= t
        x[:, :, :, self.MOUTH_LANDMARKS] = m
        y = self.KAN(x)
        
        return y

    def training_step(self, batch, batch_idx):
        x = batch['audio']
        y = batch['target']
        v = batch['ilm']
        w = batch['label']
        y_ = self.forward(x, v, w)
        tt_loss = self.loss(y_.float(), y.float())
        m_loss = self.loss(y_[:, :, :, self.MOUTH_LANDMARKS].float(), y[:, :, :, self.MOUTH_LANDMARKS].float())
        loss = tt_loss * 0.2 + m_loss * 0.8
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['audio']
        y = batch['target']
        v = batch['ilm']
        w = batch['label']

        y_ = self.forward(x, v, w)
        loss = self.loss(y_.float(), y.float())
        ld = self.LD(y_, y)
        lvd = self.LVD(y_, y)
        ldm = self.LD(y_[:,:, :,self.MOUTH_LANDMARKS], y[:,:, :, self.MOUTH_LANDMARKS])
        lvdm = self.LVD(y_[:,:, :,self.MOUTH_LANDMARKS], y[:,:, :, self.MOUTH_LANDMARKS])
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('ld', ld, on_epoch=True, prog_bar=True, logger=True)
        self.log('lvd', lvd, on_epoch=True, prog_bar=True, logger=True)
        self.log('ldm', ldm, on_epoch=True, prog_bar=True, logger=True)
        self.log('lvdm', lvdm, on_epoch=True, prog_bar=True, logger=True)


        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss_epoch'
        }
