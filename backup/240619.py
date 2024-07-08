
class S2LM(L.LightningModule):
    def __init__(self, batch, init_lr, duration, num_of_landmarks=478):
        super(S2LM, self).__init__()

        torch.autograd.set_detect_anomaly(True)


        # self.MOUTH_LANDMARKS =[61, 76, 62, 185, 184, 183, 78, 77, 146, 191, 95, 96, 40, 74, 42, 80, 88, 89, 90, 91, 39, 73, 41, 81, 178, 179, 
        #                         180, 181, 37, 72, 38, 82, 87, 86, 85, 84, 0, 11, 12, 13, 14, 15, 16, 17, 267, 302, 268, 312, 317, 316, 315, 314,
        #                             269, 303, 271, 311, 402, 403, 404, 405, 270, 304, 272, 310, 318, 319, 320, 321, 409, 408, 407, 415, 324, 325,
        #                             307, 375, 308, 292, 306, 291]

        self.MOUTH_LANDMARKS =[49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 48]

        self.init_lr = init_lr
        self.duration = 1
        self.batch = 4
        self.frame = 30*duration
        self.loss = nn.MSELoss()
        self.batch = batch


        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-100h").train()
        self.conv = nn.Conv1d(in_channels=1, out_channels=self.frame, kernel_size=1)
        self.conv_m = nn.Conv1d(in_channels=1, out_channels=self.frame, kernel_size=1)
        self.conv_f = nn.Sequential(nn.Conv1d(99, 512, 3, 1, 1), nn.Conv1d(512, self.frame, 1, 1))

        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=2)
        self.transformer_encoder = nn.Sequential(nn.TransformerEncoder(encoder_layer, num_layers=4), nn.Linear(in_features=768, out_features=num_of_landmarks*2))
        decoder_layer = nn.TransformerDecoderLayer(d_model=num_of_landmarks*2, nhead=2)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=8)

        self.transformer_encoder_m = nn.Sequential(nn.TransformerEncoder(encoder_layer, num_layers=4), nn.Linear(in_features=768, out_features=len(self.MOUTH_LANDMARKS)*2))
        decoder_layer_m = nn.TransformerDecoderLayer(d_model=len(self.MOUTH_LANDMARKS)*2, nhead=2)
        self.transformer_decoder_m = nn.TransformerDecoder(decoder_layer_m, num_layers=8)


        self.linear_out = nn.Conv2d(30, 256, 3, 1, 1)
        self.linear_m = nn.Conv2d(30, 256, 3, 1, 1)
        self.linear_e =  nn.Conv2d(256,30,1,1)
        self.linear_end = nn.Sequential(nn.Linear(68,512), nn.Linear(512,68))


    def forward(self, x, v, w):

        v_m = v[:,:,self.MOUTH_LANDMARKS]
        v_m = v_m.view(self.batch,1, -1)
        v_m = self.conv_m(v_m)
        v = v.view(self.batch,1, -1)
        v = self.conv(v)

        x = x.view(self.batch,-1)
        x = self.audio_encoder(x).last_hidden_state
        x = self.conv_f(x)

        with torch.no_grad():
            w = w[:, None, None]
            x = torch.mul(x, w)
        k = self.transformer_encoder_m(x)
        x = self.transformer_encoder(x)
        k = self.transformer_decoder_m(k, v_m)
        k = k.view(self.batch, self.frame, 2, -1)
        k = self.linear_m(k)


        x = self.transformer_decoder(x, v)
        x = x.view(self.batch, self.frame, 2, -1)
        x = self.linear_out(x)

        x[:,:,:,self.MOUTH_LANDMARKS] = k
        x = self.linear_e(x)
        x = x.view(self.batch, 2, self.frame,-1)
        y = self.linear_end(x)
        return y
    
    def training_step(self, batch, batch_idx):
        x = batch['audio']
        y = batch['target']
        v = batch['ilm']
        w = batch['label']
        y_ = self.forward(x, v, w)
        loss = self.loss(y_.float(), y.float())
        # m_loss = self.loss(y[:,:,:,[self.MOUTH_LANDMARKS]].float(), y_[:,:,:,[self.MOUTH_LANDMARKS]].float())
        # loss = tt_loss*0.4 + m_loss*0.6
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
