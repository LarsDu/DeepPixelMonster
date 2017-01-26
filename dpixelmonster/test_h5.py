import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except:
    print "Error using Tclkit"
import matplotlib.pyplot as plt
#import scipy.misc as misc
import h5py

def main():
    #img_fname = "pokemon_1298_96x96.h5"
    img_fname = "pokemon_bw_flp.h5"
    with h5py.File(img_fname,'r') as rf:
        dataset = rf["toons"]
        num_examples = dataset.shape[0]

        print "Num ex",num_examples

        test_img = dataset[125,:-1]
        print "MAx val",np.max(test_img)
        print "Min val",np.min(test_img)
        print "Med val",np.median(test_img)
        print "Mean val",np.mean(test_img)
        
        
        #show_img(dataset[169,:-1],(96,96,3))
        #show_img(dataset[3,:-1],(96,96,3))

        show_img(test_img,(96,96,3))
        show_img((test_img/255.),(96,96,3))

def show_img(np_arr,shape):
    reshaped_img = np.reshape(np_arr,shape)
    
    plt.imshow(reshaped_img)
    plt.show()
    
if __name__=="__main__":
    main()
