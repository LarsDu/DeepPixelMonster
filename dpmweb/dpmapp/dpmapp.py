#from app import app
from flask import render_template
from flask import request
from flask import Flask
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

dpmapp = Flask(__name__)


##############################################################################
#                                                                            #
#  Start up Tensorflow and keep active to handle image generation requests   #
#                                                                            #
##############################################################################

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('json','../../json/pkmn15.json',"""JSON file with training params""")
flags.DEFINE_string('poke_data','../../dpixelmonster/pokedex.sqlite',"""pokemon.sqlite datebase file with morphology column in 'pokemon' table""")
flags.DEFINE_integer('image_resize',64,"""H and W of image. Will resize if needed using scipy.misc""")
flags.DEFINE_string('feature_key','exclusive_morph',"""Column in sqlite table from which to generate labels. Can be "morphology" or "exclusive_morph""")

flags.DEFINE_string('identity_key','identifier',"""Select id for number, identifier for name""")

flags.DEFINE_string('make_background','white',"""Change alpha channel to "white" or "noise". """)

flags.DEFINE_string('flask_checkpoint_dir','../../json/pkmn15/checkpoints',""" Flask directory to load model checkpoint """)
flags.DEFINE_string('flask_image_dir','static/img/live_monsters',""" Flask directory where images are stored """)
   


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
#app = Flask(__name__)

"""Dict with uuid as keys and z_keys(numpy arrays) as values"""
#Note, this is sort of global and may cause issues when multiple people access site
#Fix this in the future



#uuid_dict={}
#Create Folder Nanny to delete image files after 10 minutes (600s)
monster_nanny = FolderNanny(FLAGS.flask_image_dir,time_limit = 600)

'''
def clear_uuid_images(uuid_dict,image_dir =FLAGS.flask_image_dir):
    """Remove previously generated images"""
    for uuid in uuid_dict.keys():
        os.remove(image_dir+os.sep+str(uuid)+'.png')
'''



@dpmapp.route('/')
@dpmapp.route('/dpm_index')
def dpm_index():
    question_mark = url_for("static",filename='img/question_mark.png')
    return render_template('dpm_index.html',
                           center_image=Markup("<img src=\""+question_mark+"\""+" align=\"middle\" width=160 style=\"image_rendering: pixelated;\">"),
                           caption = "Create pixel artwork using deep generative neural networks!",
                           title = 'DeepPixelMonster',
                           button1_link = url_for('gen_more'),
                           button1_message = "GENERATE" )
                           

@dpmapp.route('/about')
def dpm_about():
    #TODO: Move this caption to a separate file
    about_caption = Markup(
        "<div align=\"left\"style = \"font-size:16px\">DeepPixelMonster was developed as part of a 3-4 week <a href=\"http://insightdatascience.com/\">Insight Data Science Project</a> by <a href=\"mailto:larrydu88@gmail.com\">Lawrence Du</a>.<br><br>This site uses <a href=\"https://arxiv.org/abs/1406.2661\">Deep Convolutional Adversarial Networks</a>(DCGANs) for generating cartoon style images trained from images of Pokemon.<br><br>The (Tensorflow) source code for this project can be found at <a href=\"https://github.com/LarsDu/DeepPixelMonster\">github.com/LarsDu/DeepPixelMonster</a><br><br>The training images used were obtained from <a href=\"https://www.pkparaiso.com/xy/sprites_pokemon.php\">Pokemon Paraiso</a> with training set curation greatly aided by a SQLite database developed by <a href=\"https://github.com/veekun/pokedex\">Alex Munroe</a>.<br><br>Pokemon is a registered trademark of Nintendo Co. Ltd. and Game Freak Inc. This project is not affiliated with the copyright holders of the Pokemon franchise.<br><br>Any images generated by this DCGAN should not be used for commercial purposes without express permission of the aforementioned copyright holders due to the possibility of training set overfitting.<br><br>Special thanks to:<br><li>Taehoon Kim for <a href=\"https://github.com/carpedm20\">reference code.</a></li><li>Sarath Shekkizhar for <a href=\"https://github.com/shekkizh/TensorflowProjects\">network architecture code</a>.  </li></div>")
     
    return render_template('dpm_index.html',
                           caption = about_caption,
                           title = 'DeepPixelMonster - About')



    
    



@dpmapp.route('/gen_more')
def gen_more():
    monster_nanny.update()#clean old files out
    
    monster_list = caller.create_random(int(6*6))

    for monster in monster_list:
        #uuid_dict[monster.uuid] = monster.z_key
        monster_nanny.add(monster)
        
        
    gen_table = MonTable(monster_list,6,6)
    monster_sheet = gen_table.create_sheet()

    #print "Sheet created:",sheet_target
    b2_link =url_for('static',filename='img/live_monsters'+os.sep+monster_sheet.base_file) 
    #b2_link = monster_sheet.im_file
    b2_target = "\"_blank\""
    
    return render_template('dpm_index.html',
                        caption=Markup("Click on a monster to make more like it or press \
                        <br> <strong>GENERATE NEW</strong> to create more monsters"),
                        title='DeepPixelMonster',
                        main_table=gen_table.write_table(),
                        button1_link=url_for('gen_more'),
                        button1_message="GENERATE NEW",
                        button2_link = b2_link,
                        button2_target = b2_target,
                        button2_message="Save sheet")



@dpmapp.route('/select_fav/<image_uuid>')
def select_fav(image_uuid=None):

    monster_nanny.update()#Clean old files out
    num_zs = 12 #Number of indices to randomly slect for walk
    
    #z_key = uuid_dict[image_uuid]
    z_key = monster_nanny.get_z_key_from_uuid(image_uuid)
    num_rows=6
    num_cols=6

    #Create z indices to step through for latent walk
    latent_list=[]
    orig_img_file = url_for('static',filename='img/live_monsters/'+image_uuid+'.png')
    orig_img = Markup("<br><div align=\"center\"><a href=\""+url_for('select_fav',image_uuid=image_uuid)+"\"><img src=\""+orig_img_file
                +"\""
                +" align=\"middle\" width=100 style=\"image-rendering: pixelated;\"></a></div><br>")
    
    for row in range(num_rows):
        z_inds = []
        for _ in range(num_zs):
            rand_z_index = int(np.random.randint(0,z_dim))
            z_inds.append(rand_z_index)
        
        mr_list = caller.create_latent_series(z_key,z_inds,step_size=2.0/num_cols,
                                                  num_steps=num_cols)
        latent_list.extend(mr_list)

    for lmonster in latent_list:
        #print lmonster.image_html
        #uuid_dict[lmonster.uuid] = lmonster.z_key
        monster_nanny.add(lmonster)
    selected_table = MonTable(latent_list,num_rows,num_cols)

    sheet_monster = selected_table.create_sheet()

    b2_link = url_for('static',filename='img/live_monsters'+os.sep+str(sheet_monster.base_file))
    #b2_link = monster_sheet.im_file
    b2_target ="\"_blank\""
    
    sf_caption=Markup("Click on an image to create more monsters like that image"+
                      "<br><br>Use right click and \"Save image as\" to save image or"+
                       "<br>click the green button below to save entire sheet")
    
    return render_template('dpm_index.html',
                           title='DeepPixelMonster - Select Your Favorite Monster',
                           caption = sf_caption,
                           main_table = orig_img+selected_table.write_table(),
                           button1_link=url_for('gen_more'),
                           button1_message="GENERATE NEW",
                           button2_link = b2_link,
                           button2_target = b2_target,
                           button2_message="Save sheet")

#@dpmapp.route('/save_sheet')
#def save_sheet():
#    print "I dunno"





if __name__ == '__main__':
    dpmapp.run(debug=False)
