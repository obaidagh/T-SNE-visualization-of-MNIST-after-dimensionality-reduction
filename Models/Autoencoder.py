import tensorflow as tf
from tensorflow import keras
from keras import layers,Input,Model,activations,Sequential,callbacks
from keras.utils.vis_utils import plot_model
import os,numpy as np 



def Compile_model(input_shape=(28, 28,1),code_size=9):
    input_img = Input(shape=input_shape, name="encoder_input")
    x      = layers.Conv2D(filters=1,kernel_size=3, padding='same', activation="relu",kernel_initializer='he_normal')(input_img)
    x      = layers.Flatten()(x)
    x      = layers.Dense(150, activation="elu",kernel_initializer='he_normal')(x)
    x      = layers.Dense(30, activation="elu",kernel_initializer='he_normal')(x)
    code   = layers.Dense(code_size, activation="elu",kernel_initializer='he_normal')(x)

    De_in=Input(shape=(code_size,) , name="decoder_input")
    z      = layers.Dense(30, activation="elu",kernel_initializer='he_normal')(De_in)
    z      = layers.Dense(150, activation="elu",kernel_initializer='he_normal')(z)
    z      = layers.Dense(input_shape[0]*input_shape[1], activation="elu",kernel_initializer='he_normal')(z)
    z      = layers.Reshape((28, 28, 1))(z)

    de_out = layers.Conv2DTranspose(filters=1,kernel_size=1,activation='relu',padding='same',kernel_initializer='he_normal')(z)

    Encoder=Model(input_img, code)
    Decoder=Model(De_in, de_out)

    code_decoded=Decoder(code)

    Autoencoder=Model(input_img, code_decoded)



    learning_rate=0.003
    optimizer   = keras.optimizers.Adam(learning_rate=learning_rate)
    Autoencoder.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
    return Encoder,Decoder,Autoencoder


def Train_model(Model,Train_X,Train_Y,Val_X,Val_Y):
    
    #callbacks
    reduce_LR = callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2, verbose=1, factor=0.5, min_lr=0.000000001)
    early_stopping = callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=30,restore_best_weights=True,verbose=0)
    
    os.chdir('./saved')
    weights_file = 'Tsne_mnist_mlp.h5'
    
    #if model is alerady trained load weights and training history
    if os.path.exists(weights_file):
        
        Model.load_weights(weights_file)
        my_history=np.load('my_history.npy',allow_pickle='TRUE').item()
        print('Loaded weights!')
        
    #if not train the model
    else:
        
        history_keras =  Model.fit(Train_X,Train_Y, epochs=200, validation_data=(Val_X,Val_Y), callbacks=[early_stopping,reduce_LR],verbose=0)
        Model.save_weights(weights_file)
        np.save('my_history.npy',history_keras.history)
        my_history=history_keras.history
        
    return Model,my_history