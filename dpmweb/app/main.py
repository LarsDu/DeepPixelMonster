from flask import render_template
from flask import request
from flask import Flask
#from app import app
import os
import os.path
import sys
import tensorflow as tf
import numpy as np

#Go two directories up and append to path to access dpixelmonster package
#The following statement is equiv to sys.path.append("../")
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir,os.path.pardir)))

import dpixelmonster.duinfnn as duinfnn
import dpixelmonster.tooncol as tooncol


import json
import glob
import random

from cadmus import *




##############################################################################
#                                                                            #
#  Start up Tensorflow and keep active to handle image generation requests   #
#                                                                            #
##############################################################################

#flags = tf.app.flags
#FLAGS = tf.app.flags.FLAGS

#flags.DEFINE_string('json','/media/apartment_ssd/DeepPixelMonster/dpmweb/app/static/model_checkpoints/pkmn14.json',"""JSON file with training params""")
#flags.DEFINE_string('poke_data','/media/apartment_ssd/DeepPixelMonster/dpixelmonster/pokedex.sqlite',"""pokemon.sqlite datebase file with morphology column in 'pokemon' table""")
#flags.DEFINE_integer('image_resize',64,"""H and W of image. Will resize if needed using scipy.misc""")
#flags.DEFINE_string('feature_key','exclusive_morph',"""Column in sqlite table from which to generate labels. Can be "morphology" or "exclusive_morph""")

#flags.DEFINE_string('identity_key','identifier',"""Select id for number, identifier for name""")

#flags.DEFINE_string('make_background','white',"""Change alpha channel to "white" or "noise". """)

#flags.DEFINE_string('flask_checkpoint_dir','/media/apartment_ssd/DeepPixelMonster/dpmweb/app/static/model_checkpoints',""" Flask directory to load model checkpoint """)
#flags.DEFINE_string('flask_image_dir','/media/apartment_ssd/DeepPixelMonster/dpmweb/app/static/img/live_monsters',""" Flask directory where images are stored """)

class flags:
    def __init__(self):
        self.json = '/media/apartment_ssd/DeepPixelMonster/dpmweb/app/static/model_checkpoints/pkmn14.json'
        self.poke_data = '/media/apartment_ssd/DeepPixelMonster/dpixelmonster/pokedex.sqlite'
        self.image_resize=64
        self.feature_key='exclusive_morph'
        self.identity_key='identifier'
        self.make_background ='white'
        self.flask_checkpoint_dir ='/media/apartment_ssd/DeepPixelMonster/json/pkmn14/checkpoints'
        self.flask_image_dir='/media/apartment_ssd/DeepPixelMonster/dpmweb/app/static/img/live_monsters'
    

FLAGS = flags()
params = tooncol.JsonToonParams(FLAGS.json)

"""
PokemonInput creates our image processing pipeline (image and label batches)
from an image directory and a sqlite database containing morphology information

"""


sess=tf.InteractiveSession()
#sess=tf.Session()

input_shape = [params.batch_size,FLAGS.image_resize,FLAGS.image_resize,3]

#toon_collection = tooncol.PokemonInput(params.image_dir,
#                                   FLAGS.poke_data,
#                                   params.num_epochs,
#                                   FLAGS.image_resize,
#                                   identity_key = FLAGS.identity_key,
#                                   feature_key=FLAGS.feature_key,
#                                   make_background = FLAGS.make_background)
#print "Number of training examples", toon_collection.num_examples
#        print "Number of channels",toon_collection.num_channels

dcgan = duinfnn.DCGAN(sess,
                      params,
                      input_shape,
                      None,
                      #label_dim=toon_collection.num_classes,
                      label_dim=None,
                      z_dim=128)

dcgan.make_ops(dcgan.discriminator_shekkizh,dcgan.generator_shekkizh)
dcgan.flask_demo(flask_checkpoint_dir=FLAGS.flask_checkpoint_dir)

caller = DcganCaller(dcgan,FLAGS.flask_image_dir)
    
z_dim = dcgan.z_dim
#############################################################################



#webapp
app = Flask(__name__)

"""Dict with uuid as keys and z_keys(numpy arrays) as values"""
uuid_dict={}

@app.route('/')
@app.route('/dpm_index')
def dpm_main():
    question_mark = '/static/img/question_mark.png'
    return render_template('dpm_index.html',
                           caption = "Create pixel artwork using deep generative neural networks!",
                           title='DeepPixelMonster',
                           button1_link='gen_more',
                           button1_message="GENERATE",
                           main_table='',
                           img_list=map(json.dumps,[question_mark]))


#img_list = map(json.dumps,[question_mark])  )



@app.route('/gen_more')
def gen_more():
    #Create 
    monster_list = caller.create_random(int(6*6))

    for monster in monster_list:
        uuid_dict[monster.uuid] = monster.z_key
        gen_table = MonTable(monster_list,6,6)

    return render_template('dpm_index.html',
                        caption=Markup("Click on a monster to make more like it or press \
                        <br> GENERATE MORE to create more monsters"),
                        title='DeepPixelMonster',
                        main_table=gen_table.write_table(),
                        button1_link='gen_more',
                        button1_message="GENERATE_MORE",
                        img_list={})


@app.route('/select_fav/<image_uuid>')
def select_fav(image_uuid=None):
    #Generate some monsters
    ##TODO
    print "Image uuid is",image_uuid
    z_key = uuid_dict[image_uuid]
    num_rows=8
    num_cols=8
    latent_list = []
    for row in num_rows:
        rand_z_index = int(np.random.randint(0,z_dim))
        mr_list = caller.create_latent_series(z_key,rand_z_index,step_size=0.05,
                                                  num_steps=num_cols)
        latent_list.append(mr_list)
    selected_table = MonTable(latent_list,num_rows,num_cols)

    return render_template('dpm_index.html',
                           title='DeepPixelMonster - Select Your Favorite Monster',
                           main_table = selected_table.write_table(),
                           button1_link='gen_more',
                           button1_message="GENERATE MORE")




'''
@app.route('/generate_monsters',methods=['GET','POST'])
def generate_monsters():
    return render_template('dpm_index.html',
                           title='DeepPixelMonster',
                           )
'''

app.run()

#if __name__ == '__main__':
#    tf.app.run()
