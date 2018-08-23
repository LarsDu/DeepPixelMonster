#import numpy
#from flask_table import Table,Col
import os
from flask import Markup,url_for
import scipy.misc as misc
import numpy as np

import uuid
from PIL import Image
import time

from collections import deque

class MonTable():
    def __init__(self, mon_images,num_rows=6,num_cols=6):

        
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_cells = (self.num_rows*self.num_cols)
        self.num_items = len(mon_images)
        if self.num_items != self.num_cells:
            print "Num items is",self.num_items,"while num cells is",self.num_cells 
        #self.im_scale_h = 128
        #self.im_scale_w = 128
        self.mon_images = mon_images
        self.dir_path='static/img/live_monsters'
        self.sheet_image = None
        self.items = []
        #self.table = self.populate_table()

    def create_sheet(self):
        rand_id = str(uuid.uuid4())
        #sheet_img_name = 'cur_sheet.png'
        sheet_img_name = rand_id+'.png'
        
            
        image_flist = [monim.im_file for monim in self.mon_images]
        images = [Image.open(fname) for fname in image_flist]
        im_w,im_h = images[0].size
        swidth = im_w*self.num_cols
        sheight = im_h*self.num_rows

        sheet_img = Image.new(mode='RGBA',size=(swidth,sheight),color=(0,0,0,0))

        for col in range(self.num_cols):
            for row in range(self.num_cols):
                #loc: tuple for upper left corner (y axis top to bottom)
                loc = (im_w*col, im_h*row)
                #print loc
                sheet_img.paste(images[col+self.num_cols*row],loc)
        #sheet_im_path = url_for('static',filename= 'img/live_monsters'+os.sep+sheet_img_name)
        sheet_im_path = 'static/img/live_monsters'+os.sep+sheet_img_name #file location
        #full_im_path = os.path.abspath(self.dir_path+os.sep+sheet_img_name)
        #print "Full sheet path",full_im_path
        #Since PIL does not overwrite images on save, we must remove previous sheets manually
        #if os.path.exists(full_im_path):
        #    os.remove(full_im_path)
        sheet_img.save(sheet_im_path)
        
        return MonImage(sheet_im_path) #z-key defaults to np array of zeros
        
    def call_mon_images(self):
        """Tell each MonImage what table each belongs to"""
        for mon_image in self.mon_images:
            mon_image.parent_table = self
        
    def write_table(self):
        """Write HTML5 table with MonImages and links"""

        table_body=''
        mon_cells = self.mon_images[0:self.num_cells]
        
        for row in range(self.num_rows):
            table_body=table_body+'<tr>'
            for col in range(self.num_cols):
                table_body=table_body+'<td>'+mon_cells[int(col*self.num_cols+row)].image_html+'</td>'
            table_body=table_body+'</tr>'
        full_table = '<table align = "center"><tbody>'+table_body+'</tbody></table>'
        #return self.table.__html__() #{{table}} from within a Jinja template
        return Markup(full_table)

    def clear(self):
        """
        Delete table images.
        """
        for mon_image in self.mon_images:
            mon_image.clear()

         
'''
class MonCollection:
    """This class stores a list of MonImages"""
    def __init__(self,mon_image_list):
        self.mon_list = mon_image_list #MonsterImages
'''   
 
class MonImage():
    """Encapsulates monster images and stores z_keys"""
    def __init__(self,im_file,z_key=np.zeros(128)):
        #dirpath relative to flask dpmapp.py

        self.im_file = im_file #Image file with path relative to dpmapp.py
        self.base_file = os.path.basename(self.im_file)#remove directory info
        self.url_path =url_for("static",filename="img/live_monsters")+os.sep+self.base_file
        self.file_path = os.path.realpath(self.im_file)
        self.dir_path = os.path.dirname(self.file_path) #full dir path
        self.uuid = str(os.path.splitext(self.base_file)[0])

        self.parent_table = None #This value is filled if the MonImage is passed to a MonTable 

        #print "uuid",self.uuid
        self.z_key = z_key
        #self.row_ind = 0
        #self.col_ind = 0
        #Pass uuid to select_fav to generate custom page
        #self.link_dest="\"select_fav/"+str(self.uuid)+"\""
        self.link_dest=url_for('select_fav',image_uuid=self.uuid)
        self.image_html = ('<a href=\"'+self.link_dest+'\">'+
                           '<img src='+self.url_path+
                         ' width=100 style=\"image-rendering:pixelated;\"></a>')

        self.birth_time = time.time()
    def set_row_col(self,row,col):
        self.row_ind = row
        self.col_ind = col
    
    def clear(self):
        """Delete image file"""
        os.remove(self.im_file)


        
        
class DcganCaller:
    def __init__(self,dcgan,flask_image_dir):
        self.image_dir = flask_image_dir
        self.dcgan = dcgan
        self.z_dim = self.dcgan.z_dim
        
    def create_latent_series(self,z_key,z_index,step_size=0.2,num_steps=8):
        """Create MonImages on disk along a specified walk_dim"""
        image_names,z_keys = self.dcgan.latent_walk(z_key,z_index,self.image_dir,
                                                    step_size,num_steps)
        mon_image_list = []
        for i,_ in enumerate(image_names):
            mon_image_list.append(MonImage(image_names[i],z_keys[i]))
        return mon_image_list

    
    def create_random(self,n):
        """Create n z_keys and generate MonImages"""
        image_names, z_keys = self.dcgan.generate_random_images(n,self.image_dir)
        mon_image_list = []
        for i,_ in enumerate(image_names):
            mon_image_list.append(MonImage(image_names[i],z_keys[i]))
        return mon_image_list


    
class FolderNanny:
    """
    Clear out old files to keep the harddrive from filling up 
    """
    def __init__(self,image_directory,time_limit):
        """
        Initialized FolderNanny
        :param image_directory: Directory of images to watch over 
        :param time_limit: Time in seconds to keep file on server before deletion 
                               (requires update() call)

        """
        
        self.time_limit = time_limit
        
        self.files = []
        self.mon_list = deque()
        self.uuid_dict = {}
    
    def add(self,monster_image):
        self.uuid_dict[monster_image.uuid]=monster_image.z_key
        self.mon_list.append(monster_image)

        
    def get_z_key_from_uuid(self,image_uuid):
        try:
            zkey = self.uuid_dict[image_uuid]
        except KeyError:
            print "{} not found. Using random z_key instead".format(image_uuid)
            zkey = np.random.rand(128)
        return zkey 
        
    def update(self):
        while True:
            try:
                mon_image = self.mon_list[0]
                elapsed_time = time.time()-mon_image.birth_time 
                if elapsed_time > self.time_limit:
                    self.mon_list.popleft()
                    print "Deleting old file {} after {} seconds".\
                        format(mon_image.file_path,elapsed_time)
                    try:
                        os.remove(mon_image.file_path)
                    except OSError:
                        print "Can't find file {}".format(mon_image.file_path)

                else:
                    #The first element is the oldest element. If it's still too young, no need
                    #to check other items
                    break #breaks while
                        
            except IndexError:
                #No more items in queue
                break
