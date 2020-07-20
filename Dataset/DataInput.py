import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append('/Users/agupta/opt/anaconda3/lib/python3.7/site-packages')
from pyhdf.SD  import *
import numpy as np
import math
from Utility.utils import utilityfunctions as uf
import tensorflow as tf
from sklearn.preprocessing import minmax_scale

class databuilder():
    def readfile(path,count=1):
        sequence=[]
        for i in range(0,count):
            thispath=path+"br002_"+str(i)+".hdf"
            #print("Reading Files :",thispath)
            image_sequence = SD(thispath, SDC.READ)
            sds_obj = image_sequence.select('Data-Set-2')
            dim3 = sds_obj.get()
            frame=[]
            for i in range(0,141):
                frame.append(dim3[:,:,i])
            frame=np.array(frame)
            sequence.append(frame)

        sequence=np.array(sequence)
        data=sequence.reshape(count,141,128,110,1)
        print("Data is read, Shape of the data is:")
        uf.fn_print(data.shape)
        return data

    def getIntrestingFrames(data):
        short_data = np.zeros((data.shape[0],11,128,110,1))
        # for i in range(0,1):
        #         frames.append(np.concatenate((data[i,0:10,:,:,:],data[i,-1,:,:,:]),axis=0))
        # frames=np.array(frames)
        # print(frames.shape)
        for file in range(data.shape[0]):
            short_data[file,0:10]=data[file,0:10,:,:,:]
            short_data[file,10]=data[file,140,:,:,:]
        s=np.array(short_data)
        print("Intresting frames have been selected the shape of the selected data is :")
        uf.fn_print(s.shape)
        return s

    def createPatchyData(data,window,stride):
        image_size=(data.shape[2],data.shape[3])
        #window=(32,32)
        #stride=(3,3)
        new_data=[]
        frame_num=-1
        for file in range(data.shape[0]):
            h_stop=0
            for h in range(0,image_size[0],stride[0]):
                if h_stop:
                    break
                if h + window[0] >= image_size[0]:
                    h=image_size[0] - window[0]
                    h_stop=1
                v_stop=0
                for v in range(0,image_size[1],stride[1]):
                    if v_stop:
                        break
                    if v + window[1] >= image_size[1]:
                        v=image_size[1] - window[1]
                        v_stop=1
                    frame_num+=1
                    new_data.append(data[file,:,h:h + window[0],v:v + window[1],:])
        print("Patched have been created from the data and the final shape is  :")
        new_data=np.array(new_data)
        uf.fn_print(new_data.shape)
        return new_data

    def dataGeneratorTrain(patched_data):
        tensor=tf.convert_to_tensor(patched_data)
        for i in range(len(tensor)):
            x=tensor[i,0:10,:,:,:]
            y=tensor[i,1:11,:,:,:]
            yield x,y

    def apply_min_max_scaling(data):
        shape=data.shape
        data=minmax_scale(data.ravel(),feature_range=(0,255)).reshape(shape)
        print("Intresting frames have been selected the shape of the selected data is :")
        uf.fn_print(data.shape)

        return data


    def generatorParametersTrain(OUTPUT_SHAPE=(10,32,32,1), NUM_POINTS=2000,BATCH_SIZE=16,PRE_FETCH=2):
        ds=tf.data.Dataset.from_generator(databuilder.dataGeneratorTrain,output_types=(tf.float32,tf.float32),
                                          output_shapes=(OUTPUT_SHAPE,OUTPUT_SHAPE))
        ds=ds.shuffle(1000,reshuffle_each_iteration=True)
        ds=ds.batch(BATCH_SIZE).prefetch(PRE_FETCH)
        return ds

def main():

    count=input("Number of files to read in (default= all (95)): ")
    path=os.getcwd()+"/Data/br002_file/"
    data=databuilder.readfile(path,int(count))
    new_data=databuilder.getIntrestingFrames(data)
    patched_data=databuilder.createPatchyData(new_data,(32,32),(3,3))
    input_data=databuilder.apply_min_max_scaling(patched_data)

if __name__ == "__main__":
    main()



