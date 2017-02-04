import tensorflow as tf

def tanh_to_sig_scale(tanh_input,scale_multiplier=1.):
    """For image data scaled between [-1,1],
    rescale to [0,1] * scale_multiplier (e.g. 255 for RGB images)
    """
    #Due to operator overloading, this also works with tensors
    return ((tanh_input+1.)/2.)*scale_multiplier



def random_image_transforms(image):
    #input should be [0,1]
    rand_flip=True
    rand_bright=True
    rand_contrast=True
    rand_hue=True
    rand_sat=False
    do_rescale_tanh = True
    
    if rand_flip:
        image = tf.image.random_flip_left_right(image)
    if rand_bright:
        image = tf.image.random_brightness(image,max_delta=.15)
    if rand_contrast:
        image=tf.image.random_contrast(image,lower=0.80,upper=1.2)
    if rand_hue:
        image=tf.image.random_hue(image,max_delta=0.07)
    if rand_sat:
        image=tf.image.random_saturation(image,lower=.95,upper=1.05)

    # Limit pixel values to [0, 1]
    #https://github.com/tensorflow/tensorflow/issues/3816
    image = tf.minimum(image, 1.0)
    image = tf.maximum(image, 0.)
    if do_rescale_tanh:
        #Scale from [0,1] to [-1,1]
        image = (2*image)-1
    
    return image
            
