def unet_v2(self):
        ''' more conv                                                                                                                      
        '''
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x / 255.0) (inputs)

        c1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
        c1 = Dropout(DROPOUT_RATE) (c1)
        c1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
        c2 = Dropout(DROPOUT_RATE) (c2)
        c2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
        p2 = MaxPooling2D((2, 2), padding='valid') (c2)

        c3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
        c3 = Dropout(DROPOUT_RATE) (c3)
        c3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
        c4 = Dropout(DROPOUT_RATE) (c4)
        c4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
        c5 = Dropout(DROPOUT_RATE) (c5)
        c5 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

        u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
        c6 = Dropout(DROPOUT_RATE) (c6)
        c6 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
        c6 = Dropout(DROPOUT_RATE) (c6)
        c6 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

	u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='valid') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
        c7 = Dropout(DROPOUT_RATE) (c7)
        c7 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

	u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
        c8 = Dropout(DROPOUT_RATE) (c8)
        c8 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

        u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
        c9 = Dropout(DROPOUT_RATE) (c9)
	c9 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)


def reconstruction_loss_v2(y_true, y_pred):                                                                                               
    recon_true_f = K.flatten(y_true)                                                                                                      
    recon_pred_f = K.flatten(y_pred)                                                                                                      
    recon_err = losses.mean_squared_error(recon_true_f/255, recon_pred_f)                                                                 
    return recon_err


def dice_coef_loss_v2(y_true, y_pred):                                                                                                    
    smooth = 1
    intersection = K.sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)


self.model.compile(loss={'reconstruction': reconstruction_loss_v2,                                                               
                          'segmentation': dice_coef_loss_v2},                                                                          
                   loss_weights={'reconstruction': 1.0, 'segmentation': 10.0},                                                                        
              optimizer='adam',                                                                                                          
              metrics={'reconstruction': reconstruction_loss_v2, 'segmentation': dice_coef_loss_v2}) 
