import tooncol
import gflags
import sys
import os


FLAGS = gflags.FLAGS

gflags.DEFINE_string('image_dir','',"""Directory with .png images""")
gflags.DEFINE_string('output','',"""Output HDF5 file with extension '.h5'""")
gflags.DEFINE_string('img_dims','',"""e.g. '64,64,3''""")



def main(argv):
    #image_dir = "/media/apartment_ssd/Pokemonify/PKMN.NETSR4/PokBmon/BW"
    #output_file = "pokemon_bw.h5"

    #Parse gflags
    try:
        py_file = FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        sys.exit(1)

    img_dims = [int(d) for d in FLAGS.img_dims.split(',')] #e.g. 96,96,4 Must include alpha channel
    converter = tooncol.ImagesToHdf5(FLAGS.image_dir,FLAGS.output,img_dims=img_dims,drop_channel=3)
    converter.create_hdf5()

if __name__=='__main__':
    main(sys.argv)
