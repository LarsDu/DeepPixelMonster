import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except:
    print "Error using Tclkit"
import matplotlib.pyplot as plt
import scipy.misc as misc


def main():
    my_img = '/media/apartment_ssd/DeepPixelMonster/PKMN_TRAINING_SET/fully_curated/RGBA/1.png'
    im_arr = misc.imread(my_img)
    print np.max(im_arr )


    
if __name__=="__main__":
    main()
