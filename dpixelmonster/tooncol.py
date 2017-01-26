import numpy as np
import os
import h5py
import glob
import scipy.misc
import json
"""
Written by Larry Du for Insight 2017 Remote Program

"""


class ImagesToHdf5:
    def __init__(self,image_dir,output_file,img_dims=[96,96,4],drop_channel=None,num_classes=1,fext='.png'):
        self.image_dir = os.path.abspath(image_dir)
        self.output_file=output_file

        self.drop_channel = drop_channel #0=r,1=g, 2=b, 3=alpha

        
        if self.drop_channel != None:
            #Change image dims removing one channel
            self.img_dims = img_dims[:-1]+[img_dims[-1]-1]
        else:
            self.img_dims=img_dims

        
        self.num_classes=1
        
        self.fext = fext
        self.image_files = glob.glob(self.image_dir+os.sep+'*'+self.fext)
        self.num_examples = len(self.image_files)

        
    def create_hdf5(self):
        with h5py.File(self.output_file,'w') as of:
            of.attrs['num_classes'] = self.num_classes

            num_rows= len(self.image_files)
            num_cols = np.prod(self.img_dims) + 1
            dset = of.create_dataset('toons',
                                     (num_rows,num_cols),
                                     chunks=(64,num_cols),
                                     compression=None,
                                     maxshape = (1000000,num_cols))
            
            
            for i,img in enumerate(self.image_files):
                img_arr = scipy.misc.imread(img) #[96,96,4] RGBA
                if self.drop_channel != None:
                    img_arr = np.delete(img_arr,self.drop_channel,axis=2)

                
                flat_arr = np.reshape(img_arr,num_cols-1)
                flat_arr = np.concatenate((flat_arr,[0]),axis=0) #Last position reserved for label
                dset[i,:] = flat_arr
            print "Finished writing file",self.output_file


       
                
class JsonToonParams:
    def __init__(self,json_file):
        self.filename = json_file
        self.path = os.path.dirname(os.path.abspath(self.filename))
        print "Parsing json file", self.filename
        self.data_key ="toons"
        with open(self.filename,"r") as jf:
            data = json.load(jf)
            self.training_dir = os.path.abspath(data["images_dir"])
            self.training_file = data['training_file']
            self.save_dir = self.path+os.sep+data['save_dir']
            self.num_epochs = data["num_epochs"]
            self.batch_size = data["batch_size"]
            self.learning_rate=np.float32(data["learning_rate"])
            self.beta1 = np.float32(data["beta1"])
            
            with h5py.File(self.training_file,'r') as rf:
                dset = rf[self.data_key]
                self.num_examples = dset.shape[0]
                self.record_width = dset.shape[1]
                #self.num_classes = dset.attrs["num_classes"]
                

class ToonReaderInput:
    def __init__(self,toon_params,fext='.png'):
        self.params = toon_params
        self.images_dir = self.params.images_dir
        self.fext=fext
        self.image_files = []
        for im_name in glob.glob(images_dir+os.sep+'*'+fext):
            self.image_files.append(im_name)
            
        self.image_queue = tf.train.string_input_producer(self.image_files)

        
        
        #self.input_range = input_range
        #self.min_max= self.input_range[1]-self.input_range[0]


                
            
class ToonInput:
    def __init__(self,toon_params):

        #Note batches drawn from this class will be scaled [-1,1.]
        self.params = toon_params
        self.filename = self.params.training_file
        self.hdf5_file=self.filename
        self.data_key="toons"
        #self.input_range = input_range
        #self.min_max= self.input_range[1]-self.input_range[0]

        
        
        with h5py.File(self.filename,'r') as rf:
            dataset = rf[self.data_key]
            self.num_examples = dataset.shape[0]
            self.record_width = dataset.shape[1]


        self.perm_indices = np.random.permutation(range(self.num_examples))
        self.data_key="toons"
        self.epoch_tracker = EpochTracker(self.num_examples)
        self.open()
        
    def open(self):
        self.fhandle = h5py.File(self.filename,'r')
        self.data = self.fhandle[self.data_key]
        
    def close(self):
        self.reader.close()
        
    def next_batch(self,batch_size):
        return self.pull_batch(batch_size)
        
    def pull(self,batch_size):
        return self.pull_batch(batch_size)
    
    def pull_batch(self,batch_size):
        #TODO:batch_start and batch_end
        do_reset = self.epoch_tracker.increment(batch_size)
        if do_reset:
            #Reset when next pull goes over the limit for current batch size
            self.perm_indices = np.random.permutation(range(self.num_examples))
            
        batch_start = self.epoch_tracker.cur_index
        batch_end = batch_start+batch_size

        batch_indices = self.perm_indices[batch_start:batch_end]
        #Note, must draw data via shuffled indices

        data = []
        labels =[]
        for bi in batch_indices:
            #Rescale pixel values to range [-1,1]
            norm_img = [(self.data[bi,:-1]/127.5) - 1]
            data.append(norm_img)
            labels.append([self.data[bi,-1]])
        
        all_data = np.concatenate(data,axis=0)
        all_labels = np.concatenate(labels,axis=0)

        return all_data,all_labels
            
class EpochTracker:
    def __init__(self,num_examples):
        #Reminder: this exists as a seperate class to ToonCollection
        #because the epoch tracking index need to be tracked separately during training
        # and evaluation
        self.num_examples = num_examples
        self.num_epochs = 0 #The number of epochs that have been passed
        self.cur_index = 0 #The index position on current epoch

    def increment(self,increment_size):
        #Returns true if end of current epoch
        new_index = self.cur_index + increment_size
        #Reset epoch counter if end of current epoch has been reached.
        if ( new_index+increment_size >= self.num_examples):
            
            self.num_epochs += 1
            self.cur_index = 0
            #Reshuffle indices
            return True
        else:
            self.cur_index = new_index
            return False
