import tensorflow as tf
import os
import numpy as np
from PIL import Image
import sys
import cv2
import random
from collections import OrderedDict
import time
import matplotlib.pyplot as plt

cur_dir =  os.curdir
gt_train_path = os.path.join(cur_dir,'gtfine','train')
imgs_train_path = os.path.join(cur_dir,'leftImg8bit','train')
gt_val_path = os.path.join(cur_dir,'gtfine','val')
imgs_val_path = os.path.join(cur_dir,'leftImg8bit','val')
gt_test_path = os.path.join(cur_dir,'gtfine','test')
imgs_test_path = os.path.join(cur_dir,'leftImg8bit','test')


def prepare_ground_truth(img):
    NUM_CLASSES = 5
    new_image = np.zeros((img.shape[0], img.shape[1], NUM_CLASSES))
    # road
    road_mask = img == 7
    # sidewalk
    side_mask = img == 8
    # pedestrians[person,rider,bicycle]
    ped_mask = np.logical_or(img == 24, img == 25)
    # vehicles[car,truck,bus,caravan,trailer,train,motorcycle,license plate]
    car_mask = np.logical_or.reduce((img == 26, img == 27, img == 28,
                                      img == 29, img == 30, img == 31,
                                      img == 32, img == 33, img == -1))
    # everything else
    else_mask = np.logical_not(np.logical_or.reduce((road_mask, side_mask,
                                                     ped_mask, car_mask)))

    new_image[:,:,0] = road_mask
    new_image[:,:,1] = side_mask
    new_image[:,:,2] = ped_mask
    new_image[:,:,3] = car_mask
    new_image[:,:,4] = else_mask

    return new_image.astype(np.float32)

def visualize_img(original_image,gt,road = [0,255,0],side=[0,0,255] ,ped = [255,255,0], car=[255,0,0],blend = False,normalize=True): # numchannels 4
    # road  [0,255,0] green
    # side  [0,0,255] blue
    # ped_mask [255,255,0] yellow
    # car_mask [255,0,0] red
    gt = cv2.resize(gt,dsize=(np.shape(original_image)[0],np.shape(original_image)[1]))
    if normalize:
        original_image*=117.0
        original_image+=117.0
        original_image = original_image.astype("int")
    new_image = np.copy(original_image)
    new_image[gt[:,:,0] == 1,:] = road
    new_image[gt[:,:,1] == 1,:] = side
    new_image[gt[:,:,2] == 1,:] = ped
    new_image[gt[:,:,3] == 1,:] = car
    if blend:
        new_image = Image.blend(Image.fromarray(original_image, mode='RGB').convert('RGBA'),
                            Image.fromarray(new_image, mode='RGB').convert('RGBA'),
                            alpha=0.5)
    return new_image

def visualize_img_softmax(original_image,prediction,road = [0,255,0],side=[0,0,255] ,ped = [255,255,0], car=[255,0,0],blend = False,normalize=True): # numchannels 4
    # road  [0,255,0] green
    # side  [0,0,255] blue
    # ped_mask [255,255,0] yellow
    # car_mask [255,0,0] red
    def onehot_initialization_v2(a):
        ncols = a.max()+1
        out = np.zeros( (a.size,ncols), dtype=np.uint8)
        out[np.arange(a.size),a.ravel()] = 1
        out.shape = a.shape + (ncols,)
        return out
    pred = np.squeeze(prediction)
    pred = np.argmax(pred,axis=2)
    gt = onehot_initialization_v2(pred)
    gt = cv2.resize(gt,dsize=(np.shape(original_image)[0],np.shape(original_image)[1]))
    if normalize:
        original_image*=117.0
        original_image+=117.0
        original_image = original_image.astype("int")
    new_image = np.copy(original_image)
    new_image[gt[:,:,0] == 1,:] = road
    new_image[gt[:,:,1] == 1,:] = side
    new_image[gt[:,:,2] == 1,:] = ped
    new_image[gt[:,:,3] == 1,:] = car
    if blend:
        new_image = Image.blend(Image.fromarray(original_image, mode='RGB').convert('RGBA'),
                            Image.fromarray(new_image, mode='RGB').convert('RGBA'),
                            alpha=0.5)
    return new_image

"""
# for testing purpose
folder = os.listdir(gt_train_path)[0]
name = 'aachen_000100_000019'
im1 = np.array(Image.open(os.path.join(imgs_train_path,folder,name+'_leftImg8bit.png')))
im2 = np.array(Image.open(os.path.join(gt_train_path,folder,name+'_gtFine_color.png')))
im3 = np.array(Image.open(os.path.join(gt_train_path,folder,name+'_gtFine_instanceids.png')))
im4 = np.array(Image.open(os.path.join(gt_train_path,folder,name+'_gtFine_labelids.png')))
gt = prepare_ground_truth(im4)
new_image = visualize_img(im1,gt,normalize=False,blend=True)
plt.imshow(new_image)
"""

val_list = os.listdir(imgs_val_path)
train_list = os.listdir(imgs_train_path)

val_img_set = []
val_gt_set = []
train_img_set = []
train_gt_set = []

for folder_name in train_list:
    imgs = os.listdir(os.path.join(imgs_train_path,folder_name))
    gt = os.listdir(os.path.join(gt_train_path,folder_name))
    for img_name in imgs:
        train_img_set.append(os.path.join(folder_name,img_name))
        gt_name = img_name.replace('leftImg8bit.png','gtFine_labelIds.png')
        train_gt_set.append(os.path.join(folder_name,gt_name))
for folder_name in val_list:
    imgs = os.listdir(os.path.join(imgs_val_path,folder_name))
    gt = os.listdir(os.path.join(gt_val_path,folder_name))
    for img_name in imgs:
        val_img_set.append(os.path.join(folder_name,img_name))
        gt_name = img_name.replace('leftImg8bit.png','gtFine_labelIds.png')
        val_gt_set.append(os.path.join(folder_name,gt_name))

def preprocess_img_gt(img,fine_labels,output_size = (572,572),gt_size = (386,386)):
    #crop image 800 * 1600
    crop_img = img[24:824,224:1824]
    crop_labels = fine_labels[24:824,224:1824]
    resize_img = cv2.resize(crop_img,dsize=output_size)
    resize_label = cv2.resize(crop_labels,dsize=gt_size)
    resize_img = (resize_img - 117.0)/117.0
    gt = prepare_ground_truth(resize_label)
    return resize_img,gt

def batch_generator(batch_size = 1, output_size = (572,572)):
    total_images = len(train_img_set)
    while True:
        sample = random.sample(range(total_images),batch_size)
        images = []
        gts = []
        for index in sample:
            img,gt = preprocess_img_gt(np.array(Image.open(os.path.join(imgs_train_path,train_img_set[index]))),
                                       np.array(Image.open(os.path.join(gt_train_path,train_gt_set[index]))),
                                       output_size)
            images.append(img)
            gts.append(gt)
        yield (np.array(images),np.array(gts))



def validation_generator(output_size = (572,572)):
    for index in range(0,len(val_img_set)):
        img,gt = preprocess_img_gt(np.array(Image.open(os.path.join(imgs_val_path,val_img_set[index]))),
                                       np.array(Image.open(os.path.join(gt_val_path,val_gt_set[index]))),
                                       output_size)
        yield (np.array([img]),np.array([gt]))


#testing batch and validation generator
"""
generator = batch_generator(1)
img,gt = generator.__next__()
plt.imshow(visualize_img(img[0],gt[0]))
valgen = validation_generator()
t = time.time()
for i in range(len(val_img_set)):
    img,gt = valgen.__next__()
    pro = int((i/len(val_img_set))*100)
    sys.stdout.write("\r"+"="*pro+">{}%".format(pro))
    sys.stdout.flush()
print("\n time taken {}".format(time.time()-t))
"""





class UNET:
    def __init__(self,num_classes = 5,channels = 3,input_shape=[572,572],mode = 'train'):
        self.model_path = os.path.join(os.curdir,'model')
        self.layer_store = OrderedDict()
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.num_classes =  num_classes
        tf.reset_default_graph()
        self.channels = channels
        self.x = tf.placeholder("float", shape=[1, input_shape[0], input_shape[1], channels])
        self.y = tf.placeholder("float", shape=[None, None, None, self.num_classes])
        logits, self.vars = self.build_model(mode)
        self.loss = self.get_loss(logits)
        self.predictor = self.pixel_wise_softmax(logits)

        # These two are for validation
        self.correct_pred = tf.equal(tf.argmax(self.predictor, 3), tf.argmax(self.y, 3))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def build_model(self,mode='train'):
        if not mode in ['train','test']:
            raise ValueError("invalid mode")
        self.mode = mode
        # first
        stddev = np.sqrt(2 / ((9)*64)) #paper
        conv_1 = self.conv2d(self.x,size=64,stddev=stddev,name='conv_1')
        if mode == 'train':
            conv_1 = self.dropout(conv_1,name='drop_1')
        conv_2 = self.conv2d(conv_1,size=64,stddev=stddev,name='conv_2')
        if mode == 'train':
            conv_2 = self.dropout(conv_2,name='drop_2')
        self.layer_store[0] = conv_2
        max_pool_1 = self.max_pool_2by2(conv_2,'pool_1')
        # second
        stddev = np.sqrt(2 / ((9)*128)) #paper
        conv_3 = self.conv2d(max_pool_1,size=128,stddev=stddev,name='conv_3')
        if mode == 'train':
            conv_3 = self.dropout(conv_3,name='drop_3')
        conv_4 = self.conv2d(conv_3,size=128,stddev=stddev,name='conv_4')
        if mode == 'train':
            conv_4 = self.dropout(conv_4,name='drop_4')
        self.layer_store[1] = conv_4
        max_pool_2 = self.max_pool_2by2(conv_4,'pool_2')
        #third
        stddev = np.sqrt(2 / ((9)*256)) #paper
        conv_5 = self.conv2d(max_pool_2,size=256,stddev=stddev,name='conv_5')
        if mode == 'train':
            conv_5 = self.dropout(conv_5,name='drop_5')
        conv_6 = self.conv2d(conv_5,size=256,stddev=stddev,name='conv_6')
        if mode == 'train':
            conv_6 = self.dropout(conv_6,name='drop_6')
        self.layer_store[2] = conv_6
        max_pool_3 = self.max_pool_2by2(conv_6,'pool_3')
        #forth
        stddev = np.sqrt(2 / ((9)*512)) #paper
        conv_7 = self.conv2d(max_pool_3,size=512,stddev=stddev,name='conv_7')
        if mode == 'train':
            conv_7 = self.dropout(conv_7,name='drop_7')
        conv_8 = self.conv2d(conv_7,size=512,stddev=stddev,name='conv_8')
        if mode == 'train':
            conv_8 = self.dropout(conv_8,name='drop_8')
        self.layer_store[3] = conv_8
        max_pool_4 = self.max_pool_2by2(conv_8,'pool_4')
        #mid
        stddev = np.sqrt(2 / ((9)*1024)) #paper
        conv_9 = self.conv2d(max_pool_4, size = 1024,stddev=stddev, name='conv_9')
        if mode == 'train':
            conv_9 = self.dropout(conv_9,name='drop_9')
        conv_10 = self.conv2d(conv_9,size=1024,stddev=stddev,name='conv_10')
        if mode == 'train':
            conv_10 = self.dropout(conv_10,name='drop_10')
        #first dec
        dconv_1 = self.deconv2d(conv_10,2,stddev,'dconv_1')
        dconv_1_crop = self.crop_and_concat(self.layer_store[3],dconv_1,name='crop_1')
        stddev = np.sqrt(2 / ((9)*512))
        conv_11 = self.conv2d(dconv_1_crop,size = 512 ,stddev=stddev , name='conv_11')
        conv_12 = self.conv2d(conv_11,size = 512 ,stddev=stddev , name='conv_12')
        # second dec
        dconv_2 = self.deconv2d(conv_12,2,stddev,'dconv_2')
        dconv_2_crop = self.crop_and_concat(self.layer_store[2],dconv_2,name='crop_2')
        stddev = np.sqrt(2 / ((9)*256))
        conv_13 = self.conv2d(dconv_2_crop,size = 256 ,stddev=stddev , name='conv_13')
        conv_14 = self.conv2d(conv_13,size = 256 ,stddev=stddev , name='conv_14')
        # third dec
        dconv_3 = self.deconv2d(conv_14,2,stddev,'dconv_3')
        dconv_3_crop = self.crop_and_concat(self.layer_store[1],dconv_3,name='crop_3')
        stddev = np.sqrt(2 / ((9)*128))
        conv_15 = self.conv2d(dconv_3_crop,size = 128 ,stddev=stddev , name='conv_15')
        conv_16 = self.conv2d(conv_15,size = 128 ,stddev=stddev , name='conv_16')
        # forth dec
        dconv_4 = self.deconv2d(conv_16,2,stddev,'dconv_6')
        dconv_4_crop = self.crop_and_concat(self.layer_store[0],dconv_4,name='crop_4')
        stddev = np.sqrt(2 / ((9)*64))
        conv_17 = self.conv2d(dconv_4_crop,size = 64 ,stddev=stddev , name='conv_17')
        conv_18 = self.conv2d(conv_17,size = 64 ,stddev=stddev , name='conv_18')
        # output
        stddev = np.sqrt(2 / ((9)*self.num_classes))
        output = self.conv2d(conv_18,size = self.num_classes,stddev=stddev,name='output')
        var = tf.trainable_variables()
        return output,var
    def save(self, sess, model_path):
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path
    def restore(self, sess, model_path):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
    #Helper Function
    def deconv2d(self, layer, pool_size,stddev,name):
        x_shape = np.shape(layer)
        d_w = self.init_weight([pool_size,pool_size,x_shape[3].value//2,x_shape[3].value],stddev=stddev,name=name+'_w')
        d_b = self.init_bias(x_shape[3].value//2,name=name+'_b')
        output_shape = [x_shape[0].value, x_shape[1].value*2, x_shape[2].value*2, x_shape[3].value//2]
        dec = tf.nn.conv2d_transpose(layer, d_w, output_shape, strides=[1, pool_size, pool_size, 1], padding='VALID',name=name)
        dec = tf.nn.relu(tf.add(dec,d_b,name=name+'_add'),name=name+'_relu')
        return dec
    def crop_and_concat(self,x1,x2,name):
        x1_shape = np.shape(x1)
        x2_shape = np.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1].value - x2_shape[1].value) // 2, (x1_shape[2].value - x2_shape[2].value) // 2, 0]
        size = [-1, x2_shape[1].value, x2_shape[2].value, -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3,name=name)
    def init_weight(self,shape,stddev,name):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev), name=name)
    def init_bias(self, size ,name):
        return tf.Variable(tf.constant(0.1, shape=[size]),name=name)
    def max_pool_2by2(self, layer, name):
        return tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name=name)
    def conv2d(self, layer, size, stddev, name, kernal_shape = [3,3]):
        w = self.init_weight([kernal_shape[0],kernal_shape[1],np.shape(layer)[-1].value,size],stddev,name+'_w')
        b = self.init_bias(size,name+'_b')
        convo = tf.nn.conv2d(layer, w, strides=[1, 1, 1, 1], padding='VALID', name=name)
        return tf.nn.relu(tf.add(convo, b, name=name+'_add'),name=name+'_relu')
    def dropout(self, layer, name, keepprob = 0.5):
        return tf.nn.dropout(layer,keep_prob=keepprob,name=name)
    def pixel_wise_softmax(self, output):
        exponential_map = tf.exp(output)
        # sum-e^x
        sum_exp = tf.reduce_sum(exponential_map, 3, keepdims=True)
        # duplicate the last summed dimension
        tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output)[3]]))
        # divide e^x by sum-e^x
        return tf.div(exponential_map, tensor_sum_exp)

    def get_loss(self, logits, loss_beta=.01):
        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits))
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        cost += reg_loss
        return cost
    def train(self, data_generator, test_generator_fun, training_iters=4000, learning_rate=0.001,  restore=True):
        model_path = os.path.join(self.model_path, "unet.ckpt")

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            total = time.time()
            print("Session begun")
            sess.run(init)

            if restore:
                try:
                    self.restore(sess, model_path)
                except:
                    print("no checkpount found")
            for epoch in range(1,training_iters+1):
                x = time.time()
                x_batch, y_batch = data_generator.__next__()

                _, loss, _ = sess.run((self.optimizer, self.loss, self.vars),
                                                    feed_dict={self.x: x_batch,
                                                              self.y: y_batch})

                save_path = self.save(sess, model_path)

                pro = int((epoch/training_iters)*100)
                sys.stdout.write("\r"+"="*(pro//2)+">{}% epoch {}  loss = {} saved at{}".format(pro,epoch,loss,save_path))
                sys.stdout.flush()
                if epoch%50 == 0:
                    test_generator = test_generator_fun()
                    accuracy = 0
                    timeTotal = 0
                    for i in range(1,501):
                        x,y = test_generator.__next__()
                        t1 = time.time()
                        acc = sess.run([ self.accuracy], feed_dict={self.x: x, self.y: y})
                        timeTotal += (time.time()-t1)
                        accuracy+=acc[0]
                        prog = int((i/500)*100)
                        if prog % 10 == 0:
                            print("{}% acc = {}".format(prog,acc))
                    average_acc = accuracy/500
                    print("average accuracy = {} time taken {}".format(average_acc,timeTotal))
            print("total time {}".format(round(time.time()-total)))
    def predict(self, model_path, img, restore=True):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            if restore:
                self.restore(sess, model_path)
            prediction = sess.run([self.predictor], feed_dict={self.x: img})
        return prediction

net = UNET()
data_generator = batch_generator()
net.train(data_generator,validation_generator)
