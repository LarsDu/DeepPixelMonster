import tensorflow as tf
import duinfnn
import tooncol

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('json','',"""JSON file with training params""")
flags.DEFINE_string('poke_data','pokedex.sqlite',"""pokemon.sqlite datebase file with morphology column in 'pokemon' table""")
flags.DEFINE_integer('image_resize',64,"""H and W of image. Will resize if needed using scipy.misc""")
flags.DEFINE_string('feature_key','exclusive_morph',"""Column in sqlite table from which to generate labels. Can be "morphology" or "exclusive_morph""")

flags.DEFINE_string('identity_key','identifier',"""Select id for number, identifier for name""")

flags.DEFINE_string('make_background','white',"""Change alpha channel to "white" or "noise". """)


def main(_):
    params = tooncol.JsonToonParams(FLAGS.json)

    """
    PokemonInput creates our image processing pipeline (image and label batches)
    from an image directory and a sqlite database containing morphology information

    """

    
    with tf.Session() as sess:

        
        input_shape = [params.batch_size,FLAGS.image_resize,FLAGS.image_resize,3]

        toon_collection = tooncol.PokemonInput(params.image_dir,
                                               FLAGS.poke_data,
                                               params.num_epochs,
                                               FLAGS.image_resize,
                                               identity_key = FLAGS.identity_key,
                                               feature_key=FLAGS.feature_key,
                                               make_background = FLAGS.make_background)
        print "Number of training examples", toon_collection.num_examples
#        print "Number of channels",toon_collection.num_channels
        
        dcgan = duinfnn.DCGAN(sess,
                              params,
                              input_shape,
                              toon_collection,
                              #label_dim=toon_collection.num_classes,
                              label_dim=None,
                              z_dim=128)

        dcgan.make_ops(dcgan.discriminator_shekkizh,dcgan.generator_shekkizh)
        dcgan.train()
        
if __name__ == '__main__':
    tf.app.run()
