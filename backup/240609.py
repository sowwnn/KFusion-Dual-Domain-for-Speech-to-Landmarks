class S2LM(L.LightningModule):
    def __init__(self, batch, init_lr, duration):
        super(S2LM, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=3200, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.linear = nn.Linear(in_features=3200, out_features=956)
        decoder_layer = nn.TransformerDecoderLayer(d_model=956, nhead=2)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.frame = 30*duration
        self.conv = nn.Conv1d(in_channels=1, out_channels=self.frame, kernel_size=1)

        self.loss = nn.MSELoss()
        self.batch = batch
        self.init_lr = init_lr

        self.MOUTH_LANDMARKS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]



    def forward(self, x, v, w=None):

        v = v.view(4,1, -1)
        v = self.conv(v.float())
        x = x.view(self.batch, self.frame, -1)
        x = self.transformer_encoder(x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        if w is not None:
            weighted = self.normalize_lr()
            w = w.view(4,self.frame, -1)
            x = x*1-weighted + w*weighted
            
        x = self.transformer_decoder(x, v)
        y = x.view(self.batch, 2, self.frame, -1)
        return y
    
    def training_step(self, batch, batch_idx):
        x = batch['audio']
        y = batch['target']
        v = batch['ilm']
        y_ = self.forward(x, v, y)
        tt_loss = self.loss(y_.float()*256, y.float()*256)
        m_loss = self.loss(y[:,:,:,[self.MOUTH_LANDMARKS]].float()*256, y_[:,:,:,[self.MOUTH_LANDMARKS]].float()*256)
        loss = tt_loss*0.4 + m_loss*0.6
        self.log('weighted', self.normalize_lr(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'  
        }

    def normalize_lr(self):
        current_lr = self.lr_schedulers().get_last_lr()[-1]
        eta_min = 0.0001
        initial_lr = self.init_lr

        return (current_lr - eta_min) / (initial_lr - eta_min)
    


    
datas = MEAD("dataset/M003.json", duration=1)
dataloader = DataLoader(datas, batch_size=4, shuffle=True,  num_workers=4)
model = S2LM(batch=4, init_lr=3e-4, duration=1)
