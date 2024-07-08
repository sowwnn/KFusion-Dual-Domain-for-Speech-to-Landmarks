
class Transformer(L.LightningModule):
    def __init__(self, in_c, out_c):
        super(Transformer, self).__init__()

        torch.autograd.set_detect_anomaly(True)

        en_lay = nn.TransformerEncoderLayer(d_model=in_c, nhead=1)
        self.transformer_encoder = nn.Sequential(nn.TransformerEncoder(en_lay, num_layers=2), nn.Linear(in_features=in_c, out_features=out_c*2))
        de_lay = nn.TransformerDecoderLayer(d_model=out_c*2, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(de_lay, num_layers=2)

    def forward(self, x, v):
        x = self.transformer_encoder(x)
        x = self.transformer_decoder(v,x)
        return x


class S2LM(L.LightningModule):
    def __init__(self, batch, init_lr, num_of_landmarks=478):
        super(S2LM, self).__init__()

        torch.autograd.set_detect_anomaly(True)

        # self.MOUTH_LANDMARKS =[61, 76, 62, 185, 184, 183, 78, 77, 146, 191, 95, 96, 40, 74, 42, 80, 88, 89, 90, 91, 39, 73, 41, 81, 178, 179, 
        #                         180, 181, 37, 72, 38, 82, 87, 86, 85, 84, 0, 11, 12, 13, 14, 15, 16, 17, 267, 302, 268, 312, 317, 316, 315, 314,
        #                             269, 303, 271, 311, 402, 403, 404, 405, 270, 304, 272, 310, 318, 319, 320, 321, 409, 408, 407, 415, 324, 325,
        #                             307, 375, 308, 292, 306, 291]

        self.MOUTH_LANDMARKS =[49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 48]

        self.init_lr = init_lr
        self.loss = nn.MSELoss()
        self.batch = batch

        self.w2v = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()

        self.trans_ct = Transformer(in_c=87, out_c=num_of_landmarks)
        self.trans_gb = Transformer(in_c=1068, out_c=num_of_landmarks)

        self.lstm = nn.LSTM(input_size=87, hidden_size=68, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.mouth = nn.LSTM(input_size=87, hidden_size=len(self.MOUTH_LANDMARKS), num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)

        self.R1 = nn.Sequential(nn.Conv1d(6,32,3,1,1), nn.Conv1d(32,32,1,1), nn.Conv1d(32,2,3,1,1))
        self.R2 = nn.Conv1d(6,2,1,1)

        self.KAN = KAN([68,128,68])


    def forward(self, x, v, w):

        v = v.view(self.batch,1, -1)
        gb = x.view(self.batch,1, -1)
        gb = self.trans_gb(gb, v)
        gb = gb.view(self.batch, 2, -1)

        x = x.view(self.batch,-1)
        x = self.w2v(x)[0]
        x = x.view(self.batch, 1,-1)
        with torch.no_grad():
            w = nn.functional.softmax(w.float(), dim=0)
            w = w[:, None, None]
            x = torch.mul(x, w)
            # x = torch.add(x, w)

        m = self.lstm(x)[0]
        m = m.view(self.batch, 2, -1)

        m_ = self.mouth(x)[0]
        m_ = m_.view(self.batch, 2, -1)

        x = self.trans_ct(x,v)
        x = x.view(self.batch, 2, -1)
        x = torch.cat([m, gb, x],dim=1)
        t = self.R2(x)
        x = self.R1(x)

        x = x.view(self.batch, 2, -1)
        x *= t
        # x += t
        x[:,:,self.MOUTH_LANDMARKS] = x[:,:,self.MOUTH_LANDMARKS] * m_
        y = self.KAN(x)
        return y
    
    def training_step(self, batch, batch_idx):
        x = batch['audio']
        y = batch['target']
        v = batch['ilm']
        w = batch['label']
        y_ = self.forward(x, v, w)
        tt_loss = self.loss(y_.float(), y.float())
        m_loss = self.loss(y[:,:,self.MOUTH_LANDMARKS].float(), y_[:,:,self.MOUTH_LANDMARKS].float())
        loss = tt_loss*0.4 + m_loss*0.6
        # self.log('weighted', self.normalize_lr(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['audio']
        y = batch['target']
        v = batch['ilm']
        w = batch['label']

        y_ = self.forward(x,v,w)
        loss = self.loss(y_.float(), y.float())
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=50, eta_min=0.00001)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': scheduler,
            'monitor': 'val_loss_epoch'  
        }

    # def normalize_lr(self):
    #     current_lr = self.lr_schedulers().get_last_lr()[-1]
    #     eta_min = 0.0001
    #     initial_lr = self.init_lr

    #     return (current_lr - eta_min) / (initial_lr - eta_min)

# model = S2LM(batch=4, init_lr=1e-3, num_of_landmarks=68)
# item = next(iter(test_dataloader))
# x_ = item['audio']
# y = item['target']
# v = item['ilm']
# w = item['label']

# y_ = model(x_, v, w)