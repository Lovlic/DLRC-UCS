# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 15:52:26 2021
    ##----## 
    # Train
    ##----## 
@author: Cai Yanan
"""

import numpy as np
from tensorflow.contrib import layers
import scipy.io as sio
import argparse
import tensorflow as tf

def next_batch(data_, _index_in_epoch ,batch_size ,num_modalities, _epochs_completed):
    _num_examples = data_['0'].shape[0] #data_['1'].shape[0] 样本数
    start = _index_in_epoch
    _index_in_epoch += batch_size
    if _index_in_epoch > _num_examples:
        # Finished epoch
        _epochs_completed += 1
        # Shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        for i in range(0,num_modalities):
            data_[str(i)] = data_[str(i)][perm]

        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples
    end = _index_in_epoch
    data={}
    for i in range(0, num_modalities):
        data[str(i)] = data_[str(i)][start:end]
    return data, _index_in_epoch, _epochs_completed

class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, num_modalities=2, learning_rate = 1e-3, batch_size = 600,\
                 reg = None, denoise = False, model_path = None, restore_path = None, logs_path = './logs'):
        #n_hidden is a arrary contains the number of neurals on every layer
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.reg = reg,
        self.model_path = model_path
        self.restore_path = restore_path
        self.kernel_size = kernel_size
        self.iter = 0
        self.num_modalities =num_modalities
        self.batch_size = batch_size
        dim_size={}
        for i in range(0, self.num_modalities):
            modality = str(i)
            # 神经网络降维X(encoder):self.z -- n*dim_size
            dim_size[modality]=self.n_input[modality][0]/8 * self.n_input[modality][1]/8 *30
            #n_input:[32 64] strides:[2 2][2 2][2 2] n_hidden:30  
            #dim_size(..latent_c..):(32/2/2/2)*(64/2/2/2)*30 = 960 
        self.dim_size=dim_size
        weights = self._initialize_weights()
        self.x={}
        Coef={} #--S
        UV={} #--U,V

        # model
        #input required to be fed
        for i in range(0, self.num_modalities):
            modality = str(i)
            self.x[modality] = tf.compat.v1.placeholder(tf.float32, [None, self.n_input[modality][0], self.n_input[modality][1], 1])

        # Encoder
        if denoise == False:
            x_input = self.x
            latents, shape = self.encoder(x_input,weights,self.num_modalities)
            
        # Initialize weights - S,U,V
        for i in range(0, self.num_modalities):
            modality = str(i)
            Coef[modality] = weights['Coef']
            #s.t. diag(S)=0
            Coef[modality] = Coef[modality] - tf.linalg.tensor_diag(tf.diag_part(Coef[modality]))
            UV[modality] = weights['UV' + modality]
        
        z={}
        z_c={}
        latent_c={}
        
        # X=XP, Y=YQ
        for i in range(0, self.num_modalities):
            modality = str(i)
            z[modality] = tf.reshape(latents[modality], [batch_size, -1]) #(样本数, ?)
            z_c[modality] = tf.matmul(Coef[modality],z[modality]) #(样本数, ?)
            latent_c[modality] = tf.reshape(z_c[modality], tf.shape(latents[modality]))  #(?, 4, 8, 30) dim_size=4*8*30 = 960 
        
        self.z = z
        self.Coef = Coef
        self.UV = UV
        
        # Decoder
        self.x_r = self.decoder(latent_c, weights, self.num_modalities, shape)
        self.saver = tf.compat.v1.train.Saver()
        
        # Loss
        # l_2 reconstruction loss  X_r-X
        self.reconst_cost_x = 0.6*tf.reduce_sum(tf.pow(tf.subtract(self.x['0'], self.x_r['0']), 2.0)) + \
                                0.1*tf.reduce_sum(tf.pow(tf.subtract(self.x['1'], self.x_r['1']), 2.0))
        self.cost = self.reconst_cost_x
        tf.compat.v1.summary.scalar("l2_loss", self.cost)
        
        # Total Loss 
        self.loss = self.cost
        
        self.merged_summary_op = tf.compat.v1.summary.merge_all()  
        
        #GradientDescentOptimizer #AdamOptimizer
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss) 
        
        tf.compat.v1.set_random_seed(1234)
        init = tf.compat.v1.global_variables_initializer()
        tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.InteractiveSession(config=tfconfig)
        self.sess.run(init)
        self.summary_writer = tf.compat.v1.summary.FileWriter(logs_path, graph=tf.compat.v1.get_default_graph())
        t_vars = tf.compat.v1.trainable_variables()
        for var in t_vars:
            print(var.name)
            print(var.shape)

    def _initialize_weights(self):
        all_weights = dict()

        for i in range(0, self.num_modalities):
            modality = str(i)
            with tf.compat.v1.variable_scope(modality):
                all_weights[modality+ '_enc_w0'] = tf.compat.v1.get_variable(modality+"_enc_w0",
                                                         shape=[self.kernel_size[0], self.kernel_size[0], 
                                                                1, self.n_hidden[0]],
                                                         initializer=layers.xavier_initializer_conv2d())
                all_weights[modality+ '_enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))

                all_weights[modality+ '_enc_w1'] = tf.compat.v1.get_variable(modality+"_enc_w1",
                                                         shape=[self.kernel_size[1], self.kernel_size[1], 
                                                                self.n_hidden[0], self.n_hidden[1]],
                                                         initializer=layers.xavier_initializer_conv2d())
                all_weights[modality+ '_enc_b1'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype=tf.float32))

                all_weights[modality+ '_enc_w2'] = tf.compat.v1.get_variable(modality+"_enc_w2",
                                                         shape=[self.kernel_size[2], self.kernel_size[2], 
                                                                self.n_hidden[1], self.n_hidden[2]],
                                                         initializer=layers.xavier_initializer_conv2d())
                all_weights[modality+ '_enc_b2'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype=tf.float32))

                all_weights[modality + '_dec_w0'] = tf.compat.v1.get_variable(modality + "_dec1_w0",
                                                         shape=[self.kernel_size[2], self.kernel_size[2], 
                                                                self.n_hidden[1], self.n_hidden[3]],
                                                         initializer=layers.xavier_initializer_conv2d())
                all_weights[modality + '_dec_b0'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype=tf.float32))

                all_weights[modality + '_dec_w1'] = tf.compat.v1.get_variable(modality + "_dec1_w1",
                                                         shape=[self.kernel_size[1], self.kernel_size[1], 
                                                                self.n_hidden[0], self.n_hidden[1]],
                                                         initializer=layers.xavier_initializer_conv2d())
                all_weights[modality + '_dec_b1'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))

                all_weights[modality + '_dec_w2'] = tf.compat.v1.get_variable(modality + "_dec1_w2",
                                                         shape=[self.kernel_size[0], self.kernel_size[0], 
                                                                1, self.n_hidden[0]],
                                                         initializer=layers.xavier_initializer_conv2d())
                all_weights[modality + '_dec_b2'] = tf.Variable(tf.zeros([1], dtype=tf.float32))
                
                all_weights['UV' + modality] = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.dim_size[modality]], \
                                                                           tf.float32), name='UV' + modality)
        all_weights['Coef'] = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef')
        
        return all_weights

    # Building the encoder
    def encoder(self,X, weights,num_modalities):
        shapes = []
        latents={}
        # Encoder Hidden layer with relu activation 
        shapes.append(X['0'].get_shape().as_list())
        for i in range(0,num_modalities):
            modality = str(i)
            layer1 = tf.nn.bias_add(tf.nn.conv2d(X[modality],weights[modality+ '_enc_w0'],strides=[1,2,2,1],padding='SAME'),\
                                    weights[modality+ '_enc_b0'])
            layer1 = tf.nn.relu(layer1)
            layer2 = tf.nn.bias_add(tf.nn.conv2d(layer1,weights[modality+ '_enc_w1'],strides=[1,2,2,1],padding='SAME'),\
                                    weights[modality+ '_enc_b1'])
            layer2 = tf.nn.relu(layer2)
            layer3 = tf.nn.bias_add(tf.nn.conv2d(layer2,weights[modality+ '_enc_w2'],strides=[1,2,2,1],padding='SAME'),\
                                    weights[modality+ '_enc_b2'])
            layer3 = tf.nn.relu(layer3)
            latents[modality] = layer3
            if (i==0):
                shapes.append(layer1.get_shape().as_list())
                shapes.append(layer2.get_shape().as_list())
                shapes.append(layer3.get_shape().as_list())
        return latents, shapes
    
    # Building the decoder
    def decoder(self,z, weights,num_modalities, shapes):
        recons={}
        # Decoder Hidden layer with relu activation 
        for i in range(0,num_modalities):
            modality = str(i)
            shape_de1 = shapes[2]
            layer1 = tf.add(tf.nn.conv2d_transpose(z[modality], weights[modality+'_dec_w0'], \
                                                   tf.stack([tf.shape(self.x['0'])[0],shape_de1[1],shape_de1[2],shape_de1[3]]),
                                                   strides=[1,2,2,1],padding='SAME'), \
                            weights[modality+'_dec_b0'])
            layer1 = tf.nn.relu(layer1)
            shape_de2 = shapes[1]
            layer2 = tf.add(tf.nn.conv2d_transpose(layer1, weights[modality+'_dec_w1'], \
                                                   tf.stack([tf.shape(self.x['0'])[0],shape_de2[1],shape_de2[2],shape_de2[3]]),
                                                   strides=[1,2,2,1],padding='SAME'), \
                            weights[modality+'_dec_b1'])
            layer2 = tf.nn.relu(layer2)
            shape_de3= shapes[0]
            layer3 = tf.add(tf.nn.conv2d_transpose(layer2, weights[modality+'_dec_w2'], \
                                                   tf.stack([tf.shape(self.x['0'])[0],shape_de3[1],shape_de3[2],shape_de3[3]]),
                                                   strides=[1,2,2,1],padding='SAME'), \
                            weights[modality+'_dec_b2'])
            layer3 = tf.nn.relu(layer3)
            recons[modality] = layer3
        return recons

    def partial_fit(self, X):
        feed_dict={}
        for i in range(0, len(X)):
            feed_dict[self.x[str(i)]] = X[str(i)]
        cost, summary, _ = self.sess.run((self.cost, self.merged_summary_op, self.optimizer), feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost

    def initlization(self):
        self.sess.run(self.init)

    def reconstruct(self, X):
        return self.sess.run((self.x_r), feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.z, feed_dict={self.x: X})

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("model restored")


def train_face(Img, CAE, n_input, batch_size,num_modalities,max_epochs):
    it = 0
    display_step = 300 #display the cost
    save_step = 600 #save the model
    _index_in_epoch = 0
    _epochs= 0

    # train the network
    while it<max_epochs:
        batch, _index_in_epoch, _epochs = next_batch(Img, _index_in_epoch , batch_size,num_modalities, _epochs)
        for i in range(0,num_modalities):
            batch[str(i)] = np.reshape(batch[str(i)],[batch_size,n_input[modality][0],n_input[modality][1],1])
        cost = CAE.partial_fit(batch)
        it = it +1
        avg_cost = cost/(batch_size)
        if it % display_step == 0:
            print ("epoch: %.1d" % _epochs)
            print  ("cost: %.4f" % avg_cost)
        if it % save_step == 0:
            CAE.save_model()
    return

if __name__ == '__main__':
    tf.compat.v1.reset_default_graph()
    parser = argparse.ArgumentParser(description='TRAIN')
    parser.add_argument('--epoch', dest='epoch', type=int, default=1800, help='# of epoch')
    parser.add_argument('--mat', dest='mat', default='C_I', help='path of the dataset')
    parser.add_argument('--model', dest='model', default='C_I',
                        help='name of the model to be saved')
    args = parser.parse_args()
    datapath = './Data/'+ args.mat + '.mat'
    data = sio.loadmat(datapath)
    num_modalities = 2
    Img = {}
    moda={}
    d=[32,64]
    n_input={}
    
    # Source->Target
    moda['0'] = data['S_train']
    moda['1'] = data['T_train']
    Label = data['Label_train']
    Label = np.array(Label)
    
    for i in range(0, num_modalities):
        I = []
        modality = str(i)
        n_input[modality] = [d[0], d[1]]
        img = moda[modality]
        for j in range(img.shape[1]):
            temp = np.reshape(img[:, j], [d[0], d[1]])
            I.append(temp)
            Img[modality] = np.array(I) #np.transpose(np.array(I),[0,2,1])
            Img[modality] = np.expand_dims(Img[modality][:], 3)  #--(j, 32, 64, 1)
        
    kernel_size = [5,3,3,3]
    n_hidden = [10, 20, 30, 30]
    batch_size = Img['0'].shape[0] #样本数
    lr = 1.0e-3 # learning rate
    model_path = './models/'+ args.model + '.ckpt'
    CAE = ConvAE(n_input = n_input, n_hidden = n_hidden, learning_rate = lr, kernel_size = kernel_size, \
                 batch_size = batch_size, model_path = model_path, restore_path = model_path, num_modalities=num_modalities)

    train_face(Img, CAE, n_input, batch_size, num_modalities, args.epoch)

    