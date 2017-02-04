'''
                                 
    def generatorA_28x28_mnist(self,z_pholder,reuse=False,is_phase_train=True):
        """
        Args:
        	z_pholder: has shape [batch_size,self.seed_size]
        	reuse: Set to False to generate new variable scopes 
                            (e.g.: a fresh copy of the network)
        	is_phase_train: Tells batch normalizers to use sampling statistics.
                                Switch this on for training and off for making sample images 

        
        The network architecture of the generator
        It should have a similar architecture to the discriminator in reverse
        This network does transposed convolution.

        Note that in Taehoon Kim's example, this entire network is cut and pasted
        into a sampler() method for the sake of setting the "is_phase_train" flags
        on all the batch_norm layers. For sampling, set is_phase_train to False.

        return tanh_act,logits,gen_nn 
        """

        print ("\n\nBuilding generatorA network for 28x28 grayscale output\n")
        
        with tf.variable_scope("generator") as scope:
            if reuse:
                #tf.get_variable_scope().reuse_variables()
                scope.reuse_variables()
                
            #nf1 = 96
            #nf2 = 192
            nf1=12
            nf2=24
            
            reshaped_z = tf.expand_dims(tf.expand_dims(z_pholder,1),1)
            #print "reshaped_z dims",reshaped_z.get_shape().as_list()
            
            #Back into GenericInput class to give input a "output" method
            z = dtl.GenericInput(reshaped_z)
            
            #Project 10 randomly generated numbers to size nf2
            batch_size = self.batch_size

            """
            Note: I figured out the dims the output shapes by running the discriminator network
            forward, and writing out the resulting dims. 
            """
            
            dc1 = dtl.Conv2d_transpose(z,
                                       filter_shape= [1,1,nf2,10],
                                       output_shape= [batch_size,1,1,nf2],
                                       strides= [1,1,1,1],
                                       padding = 'VALID',
                                       name = 'g_deconv1')
            #Error right here
            r1 = dtl.Relu(dc1)
            g1 = dtl.BatchNorm(r1,'g_bnorm1',is_phase_train)

            dc2 = dtl.Conv2d_transpose(g1,
                                       filter_shape= [1,1,nf2,nf2],
                                       output_shape= [batch_size,1,1,nf2],
                                       strides= [1,1,1,1],
                                       padding= 'VALID',
                                       name = 'g_deconv2')          
            r2 = dtl.Relu(dc2)
            g2 = dtl.BatchNorm(r2,'g_bnorm2',is_phase_train)
            
            dc3 = dtl.Conv2d_transpose(g2,
                                       filter_shape = [3,3,nf2,nf2],
                                       output_shape = [batch_size,4,4,nf2],
                                       strides = [1,2,2,1],
                                       padding = 'VALID',
                                       name = 'g_deconv3')
            r3 = dtl.Relu(dc3)
            g3 = dtl.BatchNorm(r3,'g_bnorm3',is_phase_train)

            dc4 = dtl.Conv2d_transpose(g3,
                                       filter_shape = [3,3,nf2,nf2],
                                       output_shape = [batch_size,9,9,nf2],
                                       strides = [1,2,2,1],
                                       padding = 'VALID',
                                       name = 'g_deconv4')
            r4 = dtl.Relu(dc4)
            g4 = dtl.BatchNorm(r4,'g_bnorm4',is_phase_train)

            dc5 = dtl.Conv2d_transpose(g4,
                                       filter_shape = [3,3,nf2,nf2],
                                       output_shape = [batch_size,11,11,nf2],
                                       strides = [1,1,1,1],
                                       padding = 'VALID',
                                       name = 'g_deconv5')         
            r5 = dtl.Relu(dc5)
            g5 = dtl.BatchNorm(r5,'g_bnorm5',is_phase_train)

            dc6 = dtl.Conv2d_transpose(g5,
                                       filter_shape = [3,3,nf1,nf2],
                                       output_shape = [batch_size,13,13,nf1],
                                       strides = [1,1,1,1],
                                       padding = 'VALID',
                                       name = 'g_deconv6')
            r6 = dtl.Relu(dc6)
            g6 = dtl.BatchNorm(r6,'g_bnorm6',is_phase_train)

            dc7 = dtl.Conv2d_transpose(g6,
                                       filter_shape = [3,3,1,nf1],
                                       output_shape = [batch_size,28,28,1],
                                       strides = [1,2,2,1],
                                       padding = 'VALID',
                                       name = 'g_deconv7')
            '''
            r7 = dtl.Relu(dc7)
            g7 = dtl.BatchNorm(r7,'g_bnorm7')
            
            dc8 = dtl.Conv2d_transpose(g7,
                                       filter_shape = [3,3,nf1,nf1],
                                       output_shape = [batch_size,30,30,nf1],
                                       strides = [1,1,1,1],
                                       padding = 'VALID',
                                       name = 'deconv8')
            r8 = dtl.Relu(dc8)
            g8 = dtl.BatchNorm(r8,'g_bnorm8')

            
            dc9 = dtl.Conv2d_transpose(g8,
                                       filter_shape = [3,3,nf1,1],
                                       output_shape = [batch_size,32,32,1],
                                       strides = [1,1,1,1],
                                       padding = 'VALID',
                                       name = 'deconv9')
            '''
            #nn = dtl.Network(reshaped_z,[last_layer],bounds=[0.,1.])
            return tf.nn.tanh(dc7.output)
            
'''
'''
    
    def discriminatorA_28x28_mnist(self,x_pholder,reuse=False):
        """
        The discriminator network.
        A convolutional neural network for real images
        return logits,discr_nn
        This version takes [b,28,28,1] BHWC inputs
        """

        print ("Building discriminatorA network for 28x28 grayscale input\n\n")
        with tf.variable_scope("discriminator") as scope:
            if reuse==True:
                scope.reuse_variables()


            #print "Resusing", str(tf.get_variable_scope().reuse)
            #Reshape image input and pad to 32x32
            x_image = dtl.ImageInput(x_pholder,image_shape=[28,28,1],pad_size=2)
            #x_image = dtl.ImageInput(x_pholder,image_shape=[96,96,4])

            #conv filter dimensions are  w,h,input_dims,output_dims
            #Example replacing pooling layers with strided conv layers

            #nf1=96   
            #nf2 = 192 
            nf1=12
            nf2=24
            
            #Block1
            cl1 = dtl.Conv2d(x_image, filter_shape = [3,3,1,nf1],
                             strides = [1,1,1,1],padding = 'VALID',name='d_conv1')
            r1 = dtl.Relu(cl1)
            
            bn1 = dtl.BatchNorm(r1,'d_bnorm1')

            cl2 = dtl.Conv2d(r1, filter_shape = [3,3,nf1,nf1],
                             strides=[1,1,1,1],padding = 'VALID',name='d_conv2')
            r2 = dtl.Relu(cl2)
            bn2 = dtl.BatchNorm(r2,'d_bnorm2')

            cl3s = dtl.Conv2d(r2,filter_shape=[3,3,nf1,nf1],
                              strides = [1,2,2,1],
                              padding = 'VALID',
                              name = 'd_conv3_strided')
            r3 = dtl.Relu(cl3s)
            bn3 = dtl.BatchNorm(r3,'d_bnorm3')

            #Block2
            cl4 = dtl.Conv2d(r3, filter_shape = [3,3,nf1,nf2],
                             strides=[1,1,1,1],
                             padding='VALID',name='d_conv4')
            r4 = dtl.Relu(cl4)
            bn4 = dtl.BatchNorm(r4,'d_bnorm4')

            cl5 = dtl.Conv2d(r4, filter_shape = [3,3,nf2,nf2],
                             strides=[1,1,1,1],
                             padding='VALID',name='d_conv5')

            r5 = dtl.Relu(cl5)
            bn5 = dtl.BatchNorm(r5,'d_bnorm5')

            cl6s = dtl.Conv2d(r5, filter_shape = [3,3,nf2,nf2],
                              strides=[1,2,2,1],
                              padding='VALID',name='d_conv6_strided')

            r6 = dtl.Relu(cl6s)
            bn6 = dtl.BatchNorm(r6,'d_bnorm6')

            c7 = dtl.Conv2d(r6,filter_shape=[3,3,nf2,nf2],
                            strides = [1,2,2,1],
                            padding = 'VALID',
                            name = 'd_conv7_strided')
            r7 = dtl.Relu(c7)
            bn7 = dtl.BatchNorm(r7,'d_bnorm7')

            c8 = dtl.Conv2d(r7,filter_shape =[1,1,nf2,nf2],
                            strides=[1,1,1,1],
                            padding = 'VALID',
                            name='d_conv8_1x1')
            r8 = dtl.Relu(c8)
            dtl.BatchNorm(r8,'d_bnorm8')

            c9 = dtl.Conv2d(r8,filter_shape=[1,1,nf2,10],
                            strides=[1,1,1,1],
                            padding='VALID',
                            name='d_conv9_1x1')
            r9 = dtl.Relu(c9)

            flat = dtl.Flatten(r9)

            nn = dtl.Network(x_image,[flat],bounds=[0.,1.])

            logits = flat.output
            probs = tf.nn.sigmoid(logits)

            return probs,logits,nn

''' 
    
'''
        
    def discriminatorA_96x96(self,x_pholder,reuse=False,is_phase_train=True):
        """
        Discriminator network for 96x96 images
        No labels used to train by category
        
        Notes:
        	-Batch normalize before rectifying
        	-Only works on one class
        	-Uses a fully connected layer at the very end
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                #tf.get_variable_scope().reuse_variables()
                scope.reuse_variables()
        

            #Note: I give the discriminator fewer filters than the generator
            nf1 = 24
            nf2 = 48
            nf3=384 #fc vars
            x_image = dtl.ImageInput(x_pholder,image_shape=[96,96,3],pad_size=0)

            cl1 = dtl.Conv2d(x_image,
                             filter_shape=[9,9,3,nf1],
                             strides=[1,1,1,1],
                             padding='SAME',
                             name='d_conv1')
            #bn1 =dtl.BatchNorm(cl1,'d_bnorm1',is_phase_train)
            r1 = dtl.Relu(cl1)

            #Strided Conv
            cl2 = dtl.Conv2d(r1,[5,5,nf1,nf1],[1,2,2,1],'SAME',name='d_conv2')
            #bn2 =dtl.BatchNorm(cl2,'d_bnorm2',is_phase_train)
            r2 =dtl.Relu(cl2)

            cl3 = dtl.Conv2d(r2,[5,5,nf1,nf1],[1,2,2,1],'SAME',name='d_conv3')
            #bn3 =dtl.BatchNorm(cl3,'d_bnorm3',is_phase_train)
            r3 = dtl.Relu(cl3)

            #Strided Conv
            cl4 = dtl.Conv2d(r3,[5,5,nf1,nf1],[1,2,2,1],'SAME',name='d_conv4')
            #bn4 =dtl.BatchNorm(cl4,'d_bnorm4',is_phase_train)
            r4 = dtl.Relu(cl4)

            cl5 = dtl.Conv2d(r4,[5,5,nf1,nf2],[1,2,2,1],'SAME',name='d_conv5')
            #bn5 = dtl.BatchNorm(cl5,'d_bnorm5',is_phase_train)
            r5 = dtl.Relu(cl5)

            #Strided Conv
            cl6 = dtl.Conv2d(r5,[5,5,nf2,nf2],[1,2,2,1],'SAME',name='d_conv6')
            #bn6 =dtl.BatchNorm(cl6,'d_bnorm6',is_phase_train)
            r6 = dtl.Relu(cl6)

            cl7 = dtl.Conv2d(r6,[3,3,nf2,nf2],[1,3,3,1],'SAME',name='d_conv7')
            #bn7 = dtl.BatchNorm(cl7,'d_bnorm7',is_phase_train)
            r7 = dtl.Relu(cl7)

            #Strided FC layers
            cl8 = dtl.Conv2d(r7,[1,1,nf2,nf3],[1,1,1,1],'SAME',name='d_conv8')
            #bn8 = dtl.BatchNorm(cl8,'d_bnorm8',is_phase_train)
            r8 = dtl.Relu(cl8)

            
            cl9 = dtl.Conv2d(r8,[1,1,nf3,self.seed_size],[1,1,1,1],'SAME',name='d_conv9')
            #bn9 =dtl.BatchNorm(cl9,'d_bnorm9',is_phase_train)
            #r9 = dtl.Relu(bn9)

            """
            cl10 = dtl.Conv2d(r9,[1,1,nf1,nf2],[1,1,1,1],'SAME',name='d_conv10')
            bn10 = dtl.BatchNorm(cl10,'d_bnorm10',is_phase_train)
            r10 = dtl.Relu(bn10)

            cl11 = dtl.Conv2d(r10,[1,1,nf2,self.seed_size],[1,1,1,1],'VALID',name='d_conv11')
            bn11 = dtl.BatchNorm(cl11,'dbnorm11',is_phase_train)
            r11 = dtl.Relu(bn11)
            """
            flat = dtl.Flatten(cl9)
            logits=flat.output
            
            #logits = cl9.output

            #nn = dtl.Network(x_image,[cl9],bounds=[-1.,1.])
            return tf.nn.sigmoid(logits), logits, None


'''     
'''
    def generatorA_96x96(self,z_pholder,reuse=False,is_phase_train=True):
        """
        Generator network for 96x96 images
        No labels used to train by category
        """
        with tf.variable_scope("generator") as scope:
            if reuse:
                #tf.get_variable_scope().reuse_variables()
                scope.reuse_variables()
                

            batch_size = self.batch_size


            """
            I used PaddingCalc.py to work out the dims of everything here.
                                                                  --Larry
            """

            print "\n\n\nBuilding generator network\n"

            nf1=48
            nf2=96
            nf3 = 1024
            #z input should be [b,z_size]
            #Transform to [b,1,1,z_size]

            #z is the seeding vector (the latent space)
            
            z_deflat = tf.expand_dims(tf.expand_dims(z_pholder,1),1)
            z_generic = dtl.GenericInput(z_deflat) #Wrap in a layer

            dc1 = dtl.Conv2d_transpose(z_generic,
                                       filter_shape= [1,1,nf3,self.seed_size],
                                       output_shape= [batch_size,1,1,nf3],
                                       strides= [1,1,1,1],
                                       padding = 'SAME',
                                       name = 'g_deconv1')
            bn1 = dtl.BatchNorm(dc1,'g_bnorm1',is_phase_train)
            r1 = dtl.Relu(bn1)

            dc2 = dtl.Conv2d_transpose(r1,[1,1,nf2,nf3],
                                          [batch_size,1,1,nf2],
                                          [1,1,1,1],'SAME',name='g_deconv2')
            bn2 =dtl.BatchNorm(dc2,'g_bnorm2',is_phase_train)
            r2  = dtl.Relu(bn2)

            dc3 = dtl.Conv2d_transpose(r2,[3,3,nf2,nf2],
                                          [batch_size,3,3,nf2],
                                          [1,3,3,1],'SAME',name='g_deconv3')
            bn3 =dtl.BatchNorm(dc3,'g_bnorm3',is_phase_train)
            r3  = dtl.Relu(bn3)

            dc4 = dtl.Conv2d_transpose(r3,[5,5,nf2,nf2],
                                          [batch_size,6,6,nf2],
                                          [1,2,2,1],'SAME',name='g_deconv4')
            bn4 =dtl.BatchNorm(dc4,'g_bnorm4',is_phase_train)
            r4  = dtl.Relu(bn4)

            dc5 = dtl.Conv2d_transpose(r4,[5,5,nf2,nf2],
                                          [batch_size,12,12,nf2],
                                          [1,2,2,1],'SAME',name='g_deconv5')
            bn5 = dtl.BatchNorm(dc5,'g_bnorm5',is_phase_train)
            r5  = dtl.Relu(bn5)

            dc6 = dtl.Conv2d_transpose(r5,[5,5,nf1,nf2],
                                          [batch_size,24,24,nf1],
                                          [1,2,2,1],'SAME',name='g_deconv6')
            bn6 = dtl.BatchNorm(dc6,'g_bnorm6',is_phase_train)
            r6  = dtl.Relu(bn6)

            dc7 = dtl.Conv2d_transpose(r6,[5,5,nf1,nf1],
                                          [batch_size,48,48,nf1],
                                          [1,2,2,1],'SAME',name='g_deconv7')
            bn7 = dtl.BatchNorm(dc7,'g_bnorm7',is_phase_train)
            r7  = dtl.Relu(bn7)

            dc8 = dtl.Conv2d_transpose(r7,[5,5,nf1,nf1],
                                          [batch_size,96,96,nf1],
                                          [1,2,2,1],'SAME',name='g_deconv8')
            bn8 = dtl.BatchNorm(dc8,'g_bnorm8',is_phase_train)
            r8 = dtl.Relu(bn8)

            dc9 = dtl.Conv2d_transpose(r8,[9,9,3,nf1],
                                          [batch_size,96,96,3],
                                          [1,1,1,1],'SAME',name='g_deconv9')
            #bn9 = dtl.BatchNorm(dc9,'g_bnorm9',is_phase_train)
            #r9 = dtl.Relu(bn9)

            """
            dc10 = dtl.Conv2d_transpose(r9,[3,3,nf1,nf1],
                                           [batch_size,88,88,nf1],
                                           [1,2,2,1],'VALID',name='g_deconv10')
            bn10 = dtl.BatchNorm(dc10,'g_bnorm10',is_phase_train)
            r10 = dtl.Relu(bn10)

            dc11 = dtl.Conv2d_transpose(r10,[9,9,3,nf1],
                                            [batch_size,96,96,3],
                                            [1,1,1,1],'VALID',name='g_deconv11')
            #bn11 = dtl.BatchNorm(dc11,'g_bnorm11',is_phase_train)
            #r11  = dtl.Relu(bn11)
            """

            

            return tf.nn.tanh(dc9.output) #Outputs should be 96,96,3

'''

            
    def discriminatorB_96x96(self,x_pholder,reuse=False,is_phase_train=True):
        """
        Discriminator network for 96x96 images
        No labels used to train by category
        
        This is shallower than "A" version

        Notes:
        	-Batch normalize before rectifying
        	-Only works on one class
        	-Uses a fully connected layer at the very end
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                #tf.get_variable_scope().reuse_variables()
                scope.reuse_variables()
        
        
            nf1 = 32
            nf2 = 64
            nf3 = 512

            x_image = dtl.ImageInput(x_pholder,image_shape=self.input_shape[1:],pad_size=0)

            cl1 = dtl.Conv2d(x_image,
                             filter_shape=[9,9,3,nf1],
                             strides=[1,1,1,1],
                             padding='VALID',
                             name='d_conv1')
            bn1 =dtl.BatchNorm(cl1,'d_bnorm1',is_phase_train)
            r1 = dtl.Relu(bn1)

            #Strided Conv
            cl2 = dtl.Conv2d(r1,[5,5,nf1,nf1],[1,2,2,1],'VALID',name='d_conv2')
            bn2 =dtl.BatchNorm(cl2,'d_bnorm2',is_phase_train)
            r2 =dtl.Relu(bn2)

            cl3 = dtl.Conv2d(r2,[5,5,nf1,nf1],[1,2,2,1],'VALID',name='d_conv3')
            bn3 =dtl.BatchNorm(cl3,'d_bnorm3',is_phase_train)
            r3 = dtl.Relu(bn3)

            #Strided Conv
            cl4 = dtl.Conv2d(r3,[5,5,nf1,nf2],[1,2,2,1],'VALID',name='d_conv4')
            bn4 =dtl.BatchNorm(cl4,'d_bnorm4',is_phase_train)
            r4 = dtl.Relu(bn4)

            cl5 = dtl.Conv2d(r4,[5,5,nf2,nf2],[1,2,2,1],'VALID',name='d_conv5')
            bn5 =dtl.BatchNorm(cl5,'d_bnorm5',is_phase_train)
            r5 = dtl.Relu(bn5)

            cl6 = dtl.Conv2d(r5,[2,2,nf2,nf3],[1,1,1,1],'VALID',name='d_conv6')
            bn6 =dtl.BatchNorm(cl6,'d_bnorm6',is_phase_train)
            r6 = dtl.Relu(bn6)
         
            flat = dtl.Flatten(r6)

            #Fully connected layers
            
            l7= dtl.Linear(flat,nf3,"d_linear7")
            bn7 = dtl.BatchNorm(l7,"d_bnorm7",is_phase_train)
            r7 = dtl.Relu(bn7)

            l8 = dtl.Linear(r7,nf3,"d_linear8")
            bn8 = dtl.BatchNorm(l8,"d_bnorm8",is_phase_train)
            r8 = dtl.Relu(bn8)

            l9 = dtl.Linear(r8,self.seed_size,"d_linear9")
            bn9 =dtl.BatchNorm(l9,'d_bnorm9',is_phase_train)
            r9 = dtl.Relu(bn9)

            logits= r9.output
                        
            nn = dtl.Network(x_image,[r9],bounds=[0.,1.])
            return tf.nn.sigmoid(logits), logits, nn


    
        
        

    def generatorB_96x96(self,z_pholder,reuse=False,is_phase_train=True):
        """
        Generator network for 96x96 images
        No labels used to train by category
        #This is shallower than generatorA and includes a fully connected layer
        """
        with tf.variable_scope("generator") as scope:
            if reuse:
                #tf.get_variable_scope().reuse_variables()
                scope.reuse_variables()
                

            batch_size = self.batch_size


            """
            I used PaddingCalc.py to work out the dims of everything here.
                                                                  --Larry
            """

            print "\n\n\nBuilding generator network\n"

            nf1=32
            nf2=64
            nf3=512
            #z input should be [b,seed_size]
            #Transform to [b,1,1,seed_size]

            #z_deflat = tf.expand_dims(tf.expand_dims(z_pholder,1),1)
            z_generic = dtl.GenericInput(z_pholder) #Wrap in a layer


            #Linear projection

            l1= dtl.Linear(z_generic,nf3,"g_linear1")
            bn1 = dtl.BatchNorm(l1,'g_bnorm1',is_phase_train)
            r1 = dtl.Relu(bn1)

            l2= dtl.Linear(r1,nf3,"g_linear2")
            bn2 = dtl.BatchNorm(l2,'g_bnorm2',is_phase_train)
            r2 = dtl.Relu(bn2)

            l3 = dtl.Linear(r2,nf3,"g_linear3")
            bn3 =dtl.BatchNorm(l3,'g_bnorm3',is_phase_train)
            r3 = dtl.Relu(bn3)

            #reshape to [b,1,1,nf3]
            r3_raw = tf.expand_dims(tf.expand_dims(r3.output,1),1)

            r3_generic = dtl.GenericInput(r3_raw)

           
            dc4 = dtl.Conv2d_transpose(r3_generic,
                                       filter_shape= [2,2,nf2,nf3],
                                       output_shape= [batch_size,2,2,nf2],
                                       strides= [1,1,1,1],
                                       padding = 'VALID',
                                       name = 'g_deconv4')
            bn4 = dtl.BatchNorm(dc4,'g_bnorm4',is_phase_train)
            r4 = dtl.Relu(bn4)

            
            dc5 = dtl.Conv2d_transpose(r4,[5,5,nf2,nf2],
                                          [batch_size,8,8,nf2],
                                          [1,2,2,1],'VALID',name='g_deconv5')
            bn5 =dtl.BatchNorm(dc5,'g_bnorm5',is_phase_train)
            r5  = dtl.Relu(bn5)

            
            dc6 = dtl.Conv2d_transpose(r5,[5,5,nf1,nf2],
                                          [batch_size,19,19,nf1],
                                          [1,2,2,1],'VALID',name='g_deconv6')
            bn6 =dtl.BatchNorm(dc6,'g_bnorm6',is_phase_train)
            r6  = dtl.Relu(bn6)

            dc7 = dtl.Conv2d_transpose(r6,[5,5,nf1,nf1],
                                          [batch_size,42,42,nf1],
                                          [1,2,2,1],'VALID',name='g_deconv7')
            bn7 =dtl.BatchNorm(dc7,'g_bnorm7',is_phase_train)
            r7  = dtl.Relu(bn7)

            dc8 = dtl.Conv2d_transpose(r7,[5,5,nf1,nf1],
                                          [batch_size,88,88,nf1],
                                          [1,2,2,1],'VALID',name='g_deconv8')
            bn8 = dtl.BatchNorm(dc8,'g_bnorm8',is_phase_train)
            r8  = dtl.Relu(bn8)
            
            dc9 = dtl.Conv2d_transpose(r8,[9,9,3,nf1],
                                          [batch_size,96,96,3],
                                          [1,1,1,1],'VALID',name='g_deconv9')
            #bn9 = dtl.BatchNorm(dc9,'g_bnorm9',is_phase_train)
            #r9  = dtl.Relu(bn9)

            
            return tf.nn.tanh(dc9.output) #Outputs should be 96,96,3

'''
    def discriminatorC_96x96(self,x_pholder,reuse=False,is_phase_train=True):
        """

        Discriminator network for 96x96 images
        No labels used to train by category
        No batchnorm
        This is shallower than "A" version

        Notes:
        	-Batch normalize before rectifying
        	-Only works on one class
        	-Uses a fully connected layer at the very end
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                #tf.get_variable_scope().reuse_variables()
                scope.reuse_variables()
        
        
            nf1 = 32
            nf2 = 64
            nf3 = 512

            x_image = dtl.ImageInput(x_pholder,image_shape=self.input_shape[1:],pad_size=0)

            cl1 = dtl.Conv2d(x_image,
                             filter_shape=[9,9,3,nf1],
                             strides=[1,1,1,1],
                             padding='VALID',
                             name='d_conv1')
            #bn1 =dtl.BatchNorm(cl1,'d_bnorm1',is_phase_train)
            r1 = dtl.Relu(cl1)

            #Strided Conv
            cl2 = dtl.Conv2d(r1,[5,5,nf1,nf1],[1,2,2,1],'VALID',name='d_conv2')
            #bn2 =dtl.BatchNorm(cl2,'d_bnorm2',is_phase_train)
            r2 =dtl.Relu(cl2)

            cl3 = dtl.Conv2d(r2,[5,5,nf1,nf1],[1,2,2,1],'VALID',name='d_conv3')
            #bn3 =dtl.BatchNorm(cl3,'d_bnorm3',is_phase_train)
            r3 = dtl.Relu(cl3)

            #Strided Conv
            cl4 = dtl.Conv2d(r3,[5,5,nf1,nf2],[1,2,2,1],'VALID',name='d_conv4')
            #bn4 =dtl.BatchNorm(cl4,'d_bnorm4',is_phase_train)
            r4 = dtl.Relu(cl4)

            cl5 = dtl.Conv2d(r4,[5,5,nf2,nf2],[1,2,2,1],'VALID',name='d_conv5')
            #bn5 =dtl.BatchNorm(cl5,'d_bnorm5',is_phase_train)
            r5 = dtl.Relu(cl5)

            cl6 = dtl.Conv2d(r5,[2,2,nf2,nf3],[1,1,1,1],'VALID',name='d_conv6')
            #bn6 =dtl.BatchNorm(cl6,'d_bnorm6',is_phase_train)
            r6 = dtl.Relu(cl6)
         
            flat = dtl.Flatten(r6)

            #Fully connected layers
            
            l7= dtl.Linear(flat,nf3,"d_linear7")
            #bn7 = dtl.BatchNorm(l7,"d_bnorm7",is_phase_train)
            r7 = dtl.Relu(l7)

            l8 = dtl.Linear(r7,nf3,"d_linear8")
            #bn8 = dtl.BatchNorm(l8,"d_bnorm8",is_phase_train)
            r8 = dtl.Relu(l8)

            l9 = dtl.Linear(r8,self.seed_size,"d_linear9")
            #bn9 =dtl.BatchNorm(l9,'d_bnorm9',is_phase_train)
            r9 = dtl.Relu(l9)

            logits= r9.output
                        
            nn = dtl.Network(x_image,[r9],bounds=[0.,1.])
            return tf.nn.sigmoid(logits), logits, nn

'''
'''
    def generatorC_96x96(self,z_pholder,reuse=False,is_phase_train=True):
        """
        Generator network for 96x96 images
        No labels used to train by category
        #This is shallower than generatorA and includes a fully connected layer
        """
        with tf.variable_scope("generator") as scope:
            if reuse:
                #tf.get_variable_scope().reuse_variables()
                scope.reuse_variables()
                

            batch_size = self.batch_size


            """
            I used PaddingCalc.py to work out the dims of everything here.
                                                                  --Larry
            """

            print "\n\n\nBuilding generator network\n"

            nf1=32
            nf2=64
            nf3=512
            #z input should be [b,seed_size]
            #Transform to [b,1,1,seed_size]

            #z_deflat = tf.expand_dims(tf.expand_dims(z_pholder,1),1)
            z_generic = dtl.GenericInput(z_pholder) #Wrap in a layer


            #Linear projection

            l1= dtl.Linear(z_generic,nf3,"g_linear1")
            #bn1 = dtl.BatchNorm(l1,'g_bnorm1',is_phase_train)
            r1 = dtl.Relu(l1)

            l2= dtl.Linear(r1,nf3,"g_linear2")
            #bn2 = dtl.BatchNorm(l2,'g_bnorm2',is_phase_train)
            r2 = dtl.Relu(l2)

            l3 = dtl.Linear(r2,nf3,"g_linear3")
            #bn3 =dtl.BatchNorm(l3,'g_bnorm3',is_phase_train)
            r3 = dtl.Relu(l3)

            #reshape to [b,1,1,nf3]
            r3_raw = tf.expand_dims(tf.expand_dims(r3.output,1),1)

            r3_generic = dtl.GenericInput(r3_raw)

           
            dc4 = dtl.Conv2d_transpose(r3_generic,
                                       filter_shape= [2,2,nf2,nf3],
                                       output_shape= [batch_size,2,2,nf2],
                                       strides= [1,1,1,1],
                                       padding = 'VALID',
                                       name = 'g_deconv4')
            #bn4 = dtl.BatchNorm(dc4,'g_bnorm4',is_phase_train)
            r4 = dtl.Relu(dc4)

            
            dc5 = dtl.Conv2d_transpose(r4,[5,5,nf2,nf2],
                                          [batch_size,8,8,nf2],
                                          [1,2,2,1],'VALID',name='g_deconv5')
            #bn5 =dtl.BatchNorm(dc5,'g_bnorm5',is_phase_train)
            r5  = dtl.Relu(dc5)

            
            dc6 = dtl.Conv2d_transpose(r5,[5,5,nf1,nf2],
                                          [batch_size,19,19,nf1],
                                          [1,2,2,1],'VALID',name='g_deconv6')
            #bn6 =dtl.BatchNorm(dc6,'g_bnorm6',is_phase_train)
            r6  = dtl.Relu(dc6)

            dc7 = dtl.Conv2d_transpose(r6,[5,5,nf1,nf1],
                                          [batch_size,42,42,nf1],
                                          [1,2,2,1],'VALID',name='g_deconv7')
            #bn7 =dtl.BatchNorm(dc7,'g_bnorm7',is_phase_train)
            r7  = dtl.Relu(dc7)

            dc8 = dtl.Conv2d_transpose(r7,[5,5,nf1,nf1],
                                          [batch_size,88,88,nf1],
                                          [1,2,2,1],'VALID',name='g_deconv8')
            #bn8 = dtl.BatchNorm(dc8,'g_bnorm8',is_phase_train)
            r8  = dtl.Relu(dc8)
            
            dc9 = dtl.Conv2d_transpose(r8,[9,9,3,nf1],
                                          [batch_size,96,96,3],
                                          [1,1,1,1],'VALID',name='g_deconv9')
            #bn9 = dtl.BatchNorm(dc9,'g_bnorm9',is_phase_train)
            #r9  = dtl.Relu(bn9)

            
            return tf.nn.tanh(dc9.output) #Outputs should be 96,96,3
'''
'''
    def generatorD(self,z,reuse=False,is_phase_train=True):
        ##genD
        """Stop gap method from shekkiz for proof of concept only"""
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            gen_dim=16
            batch_size,image_size,_,num_channels = self.input_shape

            z_dim = self.seed_size
            W_0 = utils.weight_variable([z_dim,
                                       64 * gen_dim / 2 * image_size / 16 * image_size / 16],
                                            name="g_weights0")
            b_0 = utils.bias_variable([64 * gen_dim / 2 * image_size / 16 * image_size / 16],
                                       name="g_bias0")
            z_0 = tf.matmul(z, W_0) + b_0
            h_0 = tf.reshape(z_0, [-1, image_size / 16, image_size / 16, 64 * gen_dim / 2])
            h_bn0 = utils.batch_norm(h_0, 64 * gen_dim / 2, is_phase_train, scope="g_bnorm0")
            h_relu0 = tf.nn.relu(h_bn0, name='g_relu0')

            W_2 = utils.weight_variable([5, 5, 64 * gen_dim / 4, 64 * gen_dim / 2],
                                        name="g_weights2")
            b_2 = utils.bias_variable([64 * gen_dim / 4], name="g_bias2")
            deconv_shape = tf.pack([tf.shape(h_relu0)[0],
                                        image_size / 8, image_size / 8, 64 * gen_dim / 4])
            h_conv_t2 = utils.conv2d_transpose_strided(h_relu0, W_2, b_2,
                                                       output_shape=deconv_shape)
            h_bn2 = utils.batch_norm(h_conv_t2, 64 * gen_dim / 4, is_phase_train,
                                         scope="g_bnorm2")
            h_relu2 = tf.nn.relu(h_bn2, name='g_relu2')


            W_3 = utils.weight_variable([5, 5, 64 * gen_dim / 8, 64 * gen_dim / 4],
                                        name="g_weights3")
            b_3 = utils.bias_variable([64 * gen_dim / 8], name="g_bias3")
            deconv_shape = tf.pack([tf.shape(h_relu2)[0], image_size / 4,
                                        image_size / 4, 64 * gen_dim / 8])
            h_conv_t3 = utils.conv2d_transpose_strided(h_relu2, W_3, b_3,
                                                       output_shape=deconv_shape)
            h_bn3 = utils.batch_norm(h_conv_t3, 64 * gen_dim / 8, is_phase_train,
                                               scope="g_bnorm3")
            h_relu3 = tf.nn.relu(h_bn3, name='g_relu3')
            #utils.add_activation_summary(h_relu3)

            W_4 = utils.weight_variable([5, 5, 64 * gen_dim / 16, 64 * gen_dim / 8],
                                        name="g_weights4")
            b_4 = utils.bias_variable([64 * gen_dim / 16], name="g_bias4")
            deconv_shape = tf.pack([tf.shape(h_relu3)[0], image_size / 2, image_size / 2,
                                        64 * gen_dim / 16])
            h_conv_t4 = utils.conv2d_transpose_strided(h_relu3, W_4, b_4,
                                                       output_shape=deconv_shape)
            h_bn4 = utils.batch_norm(h_conv_t4, 64 * gen_dim / 16, is_phase_train,
                                     scope="g_bnorm4")
            h_relu4 = tf.nn.relu(h_bn4, name='g_relu4')
            #utils.add_activation_summary(h_relu4)

            W_5 = utils.weight_variable([5, 5, num_channels, 64 * gen_dim / 16],
                                        name="g_weights5")
            b_5 = utils.bias_variable([num_channels], name="g_bias5")
            deconv_shape = tf.pack([tf.shape(h_relu4)[0], image_size, image_size, num_channels])
            h_conv_t5 = utils.conv2d_transpose_strided(h_relu4, W_5, b_5,
                                                       output_shape=deconv_shape)
            generated_image = tf.nn.tanh(h_conv_t5, name='generated_image')

            return generated_image

'''
'''
    def discriminatorD(self,input_images,reuse=False, is_phase_train=True):
        ##disD
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            gen_dim=16
            batch_size,img_size,_,num_channels = self.input_shape
            

            W_conv0 = utils.weight_variable([5, 5, num_channels, 64 * 1], name="d_weights0")
            b_conv0 = utils.bias_variable([64 * 1], name="d_bias0")
            h_conv0 = utils.conv2d_strided(input_images, W_conv0, b_conv0)
            h_bn0 = h_conv0  # utils.batch_norm(h_conv0, 64 * 1, is_phase_train, scope="d_bnorm0")
            h_relu0 = utils.leaky_relu(h_bn0, 0.2, name="d_relu0")
            #utils.add_activation_summary(h_relu0)

            W_conv1 = utils.weight_variable([5, 5, 64 * 1, 64 * 2], name="d_weights1")
            b_conv1 = utils.bias_variable([64 * 2], name="d_bias1")
            h_conv1 = utils.conv2d_strided(h_relu0, W_conv1, b_conv1)
            h_bn1 = utils.batch_norm(h_conv1, 64 * 2, is_phase_train, scope="d_bnorm1")
            h_relu1 = utils.leaky_relu(h_bn1, 0.2, name="d_relu1")
            #utils.add_activation_summary(h_relu1)

            W_conv2 = utils.weight_variable([5, 5, 64 * 2, 64 * 4], name="d_weights2")
            b_conv2 = utils.bias_variable([64 * 4], name="d_bias2")
            h_conv2 = utils.conv2d_strided(h_relu1, W_conv2, b_conv2)
            h_bn2 = utils.batch_norm(h_conv2, 64 * 4, is_phase_train, scope="d_bnorm2")
            h_relu2 = utils.leaky_relu(h_bn2, 0.2, name="d_relu2")
            #utils.add_activation_summary(h_relu2)

            W_conv3 = utils.weight_variable([5, 5, 64 * 4, 64 * 8], name="d_weights3")
            b_conv3 = utils.bias_variable([64 * 8], name="d_bias3")
            h_conv3 = utils.conv2d_strided(h_relu2, W_conv3, b_conv3)
            h_bn3 = utils.batch_norm(h_conv3, 64 * 8, is_phase_train, scope="d_bnorm3")
            h_relu3 = utils.leaky_relu(h_bn3, 0.2, name="d_relu3")
            #utils.add_activation_summary(h_relu3)

            shape = h_relu3.get_shape().as_list()
            h_3 = tf.reshape(h_relu3, [batch_size, (img_size // 16)*(img_size // 16)*shape[3]])
            W_4 = utils.weight_variable([h_3.get_shape().as_list()[1], 1], name="W_4")
            b_4 = utils.bias_variable([1], name="d_bias4")
            h_4 = tf.matmul(h_3, W_4) + b_4
            
            return tf.nn.sigmoid(h_4), h_4,h_relu3

'''
