class S2LM(L.LightningModule):
    def __init__(self, batch, init_lr, duration, num_of_landmarks=478):
        super(S2LM, self).__init__()


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


        self.conv = nn.Conv1d(in_channels=1, out_channels=self.frame, kernel_size=1)
        self.conv_m = nn.Conv1d(in_channels=1, out_channels=self.frame, kernel_size=1)
        # self.linear_in = nn.Linear(in_features=29, out_features=30)


        encoder_layer = nn.TransformerEncoderLayer(d_model=3200, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.linear = nn.Linear(in_features=3200, out_features=num_of_landmarks*2)
        decoder_layer = nn.TransformerDecoderLayer(d_model=num_of_landmarks*2, nhead=2)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.linear_m = nn.Linear(in_features=num_of_landmarks*2, out_features=len(self.MOUTH_LANDMARKS)*2)
        decoder_layer_m = nn.TransformerDecoderLayer(d_model=len(self.MOUTH_LANDMARKS)*2, nhead=2)
        self.transformer_decoder_m = nn.TransformerDecoder(decoder_layer_m, num_layers=6)

        self.linear_end = nn.Sequential(nn.Linear(in_features=num_of_landmarks*2, out_features=1024), nn.ReLU(), nn.Linear(in_features=1024, out_features=num_of_landmarks*2))
        self.linear_end_m = nn.Sequential(nn.Linear(in_features=len(self.MOUTH_LANDMARKS)*2, out_features=256), nn.ReLU(), nn.Linear(in_features=256, out_features=len(self.MOUTH_LANDMARKS)*2))
        self.linear_out = nn.Sequential(nn.Linear(in_features=num_of_landmarks, out_features=1024), nn.ReLU(), nn.Linear(in_features=1024, out_features=num_of_landmarks))
    def forward(self, x, v, w=None):

        v_m = v[:,:,self.MOUTH_LANDMARKS]
        v = v.view(self.batch,1, -1)
        v = self.conv(v.float())

        # x = self.linear_in(x)
        x = x.view(self.batch, self.frame, -1)
        x = self.transformer_encoder(x)
        x = self.linear(x)
        x = torch.softmax(x,1)

        v_m = v_m.view(self.batch,1, -1)
        v_m = self.conv_m(v_m.float())

        x_m = self.linear_m(x)
        x_m = torch.softmax(x_m,1)
        x_m = self.transformer_decoder_m(x_m, v_m)
        x_m = self.linear_end_m(x_m)
        x_m = x_m.view(self.batch, 2, self.frame, -1)

        
        x = self.transformer_decoder(x, v)
        x = self.linear_end(x)
        x = x.view(self.batch, 2, self.frame, -1)
        x_t = x.clone()
        x_t[:,:,:,self.MOUTH_LANDMARKS] = x_m
        y = self.linear_out(x_t)

        return y
    
    def training_step(self, batch, batch_idx):
        x = batch['audio']
        y = batch['target']
        v = batch['ilm']
        y_ = self.forward(x, v, y)
        tt_loss = self.loss(y_.float(), y.float())
        m_loss = self.loss(y[:,:,:,[self.MOUTH_LANDMARKS]].float(), y_[:,:,:,[self.MOUTH_LANDMARKS]].float())
        loss = tt_loss*0.3 + m_loss*0.7
        # self.log('weighted', self.normalize_lr(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss_epoch'  
        }

    # def normalize_lr(self):
    #     current_lr = self.lr_schedulers().get_last_lr()[-1]
    #     eta_min = 0.0001
    #     initial_lr = self.init_lr

    #     return (current_lr - eta_min) / (initial_lr - eta_min)
    


    
datas = MEAD("dataset/fa_datalist.json", duration=1)
dataloader = DataLoader(datas, batch_size=4, shuffle=True,  num_workers=8)
model = S2LM(batch=4, init_lr=1e-3, duration=1, num_of_landmarks=68)
