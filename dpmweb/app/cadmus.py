#import numpy
#from flask_table import Table,Col
import os
from flask import Markup

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

        self.items = []
        #self.table = self.populate_table()

        #def populate_table(self):
             
    def write_table(self):
        """Write HTML5 table with MonImages and links"""

        table_body=''
        mon_cells = self.mon_images[0:self.num_cells]
        
        for row in range(self.num_rows):
            table_body=table_body+'<tr>'
            for col in range(self.num_cols):
                print row,col,mon_cells[(row*col+col)]
                table_body=table_body+'<td>'+mon_cells[int(row*col+col)].img_html+'</td>'
            table_body=table_body+'</tr>'
        full_table = '<table><tbody>'+table_body+'</tbody></table>'
        #return self.table.__html__() #{{table}} from within a Jinja template
        return Markup(full_table)


         
'''
class MonCollection:
    """This class stores a list of MonImages"""
    def __init__(self,mon_image_list):
        self.mon_list = mon_image_list #MonsterImages
'''   
 
class MonImage():
    """Encapsulates monster images and stores z_keys"""
    def __init__(self,im_file,z_key,local_path="static/img/live_monsters"):
        self.im_file = im_file
        self.local_path = local_path
        self.im_file_relative = self.local_path+os.sep+os.path.basename(self.im_file)
        self.uuid = os.path.splitext(os.path.basename(self.im_file))[0]
        self.z_key = z_key
        #self.row_ind = 0
        #self.col_ind = 0
        #Pass uuid to select_fav to generate custom page
        self.link_dest="\"select_fav/"+str(self.uuid)+"\""
        self.img_html = ('<a href='+self.uuid+'>'+
                          '<img src='+
                         self.im_file_relative+
                         ' width=128 style=\"image-rendering:pixelated;\"></a>')
        
    def set_row_col(self,row,col):
        self.row_ind = row
        self.col_ind = col
    
    
        
class DcganCaller:
    def __init__(self,dcgan,flask_image_dir):
        self.image_dir = flask_image_dir
        self.dcgan = dcgan
        self.z_dim = self.dcgan.z_dim
        
    def create_latent_series(self,z_key,z_index,step_size=0.05,num_steps=8):
        """Create MonImages on disk along a specified walk_dim"""
        print "LATANT DCGAN",self.dcgan.sess
        image_names,z_keys = self.dcgan.latent_walk(z_key,z_index,
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
    """Deletes images after a certain number have been created"""
    def __init__(self,image_limit,folder):
        pass
