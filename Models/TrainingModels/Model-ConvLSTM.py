from keras import Sequential
from tensorflow.keras.callbacks import History,ModelCheckpoint
from Dataset.DataInput import databuilder as db,databuilder
import os;
import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D,BatchNormalization,Conv3D
class ModelConvLSTM():

    ignoreself=1
    def create_Model(ignoreself):
        model=tf.keras.models.Sequential()
        model.add(
            ConvLSTM2D(filters=64,kernel_size=(7,7),input_shape=(None,32,32,1),padding='same',return_sequences=True,
                       activation='tanh',recurrent_activation='hard_sigmoid',kernel_initializer='glorot_uniform',
                       unit_forget_bias=True,dropout=0.3,recurrent_dropout=0.3))
        model.add(BatchNormalization())

        model.add(ConvLSTM2D(filters=32,kernel_size=(7,7),padding='same',return_sequences=True,activation='tanh',
                             recurrent_activation='hard_sigmoid',kernel_initializer='glorot_uniform',
                             unit_forget_bias=True,dropout=0.4,recurrent_dropout=0.3))
        model.add(BatchNormalization())

        model.add(ConvLSTM2D(filters=32,kernel_size=(7,7),padding='same',return_sequences=True,activation='tanh',
                             recurrent_activation='hard_sigmoid',kernel_initializer='glorot_uniform',
                             unit_forget_bias=True,dropout=0.4,recurrent_dropout=0.3))
        model.add(BatchNormalization())

        model.add(ConvLSTM2D(filters=32,kernel_size=(7,7),padding='same',return_sequences=True,activation='tanh',
                             recurrent_activation='hard_sigmoid',kernel_initializer='glorot_uniform',
                             unit_forget_bias=True,dropout=0.4,recurrent_dropout=0.3))
        model.add(BatchNormalization())

        model.add(Conv3D(filters=1,kernel_size=(1,1,1),activation='sigmoid',padding='same',data_format='channels_last'))

        ### !!! try go_backwards=True !!! ###

        print(model.summary())

        return model


    def train(input_data):
        OUTPUT_SHAPE=(10,32,32,1)
        NUM_POINTS=2000
        BATCH_SIZE=16
        PRE_FETCH=2
        EPOCHS=10
        ignoreself=1
        model=ModelConvLSTM.create_Model(ignoreself)
        model.compile(loss='mse',optimizer='adam')
        history=History()
        #filepath=str(os.getcwd())+"SavedModels/HPC2-{epoch:02d}.h5"

        #cp=ModelCheckpoint(filepath,verbose=1,save_best_only=False,mode='max',period=10)

        trainData=db.generatorParametersTrain(input_data,OUTPUT_SHAPE,NUM_POINTS,BATCH_SIZE,PRE_FETCH)
        valData=db.generatorParametersValidation(input_data,OUTPUT_SHAPE,NUM_POINTS,BATCH_SIZE,PRE_FETCH)
        history=model.fit(trainData,epochs=EPOCHS,verbose=1, validation_data=valData)# ,callbacks=[cp])

        return history
        # if is_graph:
        #     fig,ax1=plt.subplots(1,1)
        #     ax1.plot(history.history["val_loss"])
        #     ax1.plot(history.history["loss"])

def main():

    count=input("Number of files to read in (default= all (95)): ")
    path="/Users/agupta/Documents/DM LAB/patch_code/Dataset/Data/br002_file/"
    data=databuilder.readfile(path,int(count))
    new_data=databuilder.getIntrestingFrames(data)
    patched_data=databuilder.createPatchyData(new_data,(32,32),(3,3))
    input_data=databuilder.apply_min_max_scaling(patched_data)
    ModelConvLSTM.train(input_data)

if __name__ == "__main__":
    main()