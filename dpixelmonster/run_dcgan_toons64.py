import tensorflow as tf
import duinfnn
import tooncol

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('json','',"""JSON file with training params""")

def main(_):
    params = tooncol.JsonToonParams(FLAGS.json)
    toon_collection = tooncol.ToonInput(params)
    with tf.Session() as sess:
        input_shape = [params.batch_size,64,64,3]
        dcgan = duinfnn.DCGAN(sess,params,input_shape,seed_size=128)
        dcgan.make_ops(dcgan.discriminatorD,dcgan.generatorD)
        dcgan.train(toon_collection)
        
if __name__ == '__main__':
    tf.app.run()

