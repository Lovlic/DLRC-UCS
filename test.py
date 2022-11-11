# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 15:52:26 2021
    ##----## 
    # Test
    ##----## 
@author: Cai Yanan
"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize
from munkres import Munkres
import argparse

class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1 = 1.0, re_constant2 = 1.0, batch_size = 100, \
                 reg = None, denoise = False, model_path = None, restore_path = None, logs_path = './logs', num_modalities=2):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.num_modalities =num_modalities
        self.batch_size = batch_size
        dim_size={}
        for i in range(0, self.num_modalities):
            modality = str(i)
            dim_size[modality]=self.n_input[modality][0]/8 * self.n_input[modality][1]/8 *30
        self.dim_size=dim_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0
        weights = self._initialize_weights()
        self.x={}
        Coef={}
        UV={}
        
        # model
        # input required to be fed
        for i in range(0, self.num_modalities):
            modality = str(i)
            self.x[modality] = tf.compat.v1.placeholder(tf.float32, [None, self.n_input[modality][0], self.n_input[modality][1], 1])
            
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, [], name='learningRate')
        
        # Encoder
        if denoise == False:
            x_input = self.x
            latents, shape = self.encoder(x_input,weights,self.num_modalities)
          
        # Initialize weights - S,U,V
        for i in range(0, self.num_modalities):
            modality = str(i)
            Coef[modality] = weights['Coef']
            # s.t. diag(S)=0
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
              
        # Loss function
        # reg losses S:Coef['0']-Coef['1']
        self.reg_losses = tf.reduce_sum(tf.pow(Coef['0'],2.0)) + tf.reduce_sum(tf.pow(Coef['1'],2.0))
        tf.compat.v1.summary.scalar("reg_loss", re_constant2 * self.reg_losses )
        
        # selfexpress losses X=XP+E1,Y=YQ+E2
        self.selfexpress_losses =  0.7 * tf.reduce_sum(tf.pow(tf.subtract(z['0'], z_c['0']), 2.0)) + \
                                   0.3 * tf.reduce_sum(tf.pow(tf.subtract(z['1'], z_c['1']), 2.0)) 
        tf.compat.v1.summary.scalar("selfexpress_loss", reg_constant1 * self.selfexpress_losses )
        
        # graph loss   ||U^T*xi-U^T*xj||^2 Pij + ||V^T*yi-V^T*yj||^2 Qij
        E1 = tf.pow(tf.subtract(tf.matmul(z['0'],tf.transpose(UV['0'])), \
                                tf.matmul(z['0'],tf.transpose(UV['0']))), 2.0)
        E2 = tf.pow(tf.subtract(tf.matmul(z['1'],tf.transpose(UV['1'])), \
                                tf.matmul(z['1'],tf.transpose(UV['1']))), 2.0)
        self.graph_loss = tf.reduce_sum(tf.matmul(E1, Coef['0'])) + tf.reduce_sum(tf.matmul(E2, Coef['1']))
        tf.compat.v1.summary.scalar("graph_loss", re_constant2 * self.graph_loss ) 

        # CCA loss X'U=Y'V+E
        self.CA_loss =  tf.reduce_sum(tf.pow(tf.subtract(tf.matmul(tf.transpose(z['0']),UV['0']), \
                                                        tf.matmul(tf.transpose(z['1']),UV['1'])), 2.0))
        tf.compat.v1.summary.scalar("CA_loss", reg_constant1 * self.CA_loss ) 

        self.z = z
        self.z_c =z_c
        self.Coef = Coef
        self.UV = UV
        
        # Decoder
        self.x_r = self.decoder(latent_c, weights, self.num_modalities, shape)

        # Loss
        # l_2 reconstruction loss  X_r-X
        self.reconst_cost_x =  0.6 * tf.reduce_sum(tf.pow(tf.subtract(self.x['0'], self.x_r['0']), 2.0)) + \
                                0.1 * tf.reduce_sum(tf.pow(tf.subtract(self.x['1'], self.x_r['1']), 2.0))    
        tf.compat.v1.summary.scalar("recons_loss", self.reconst_cost_x)
        
        # Total Loss
        self.loss =  re_constant2 * self.reg_losses + reg_constant1 * self.selfexpress_losses + \
                    re_constant2 * self.graph_loss + reg_constant1 * self.CA_loss + self.reconst_cost_x
                    
        self.merged_summary_op = tf.compat.v1.summary.merge_all()
        
        #GradientDescentOptimizer #AdamOptimizer
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss) 
    
        self.init = tf.compat.v1.global_variables_initializer()
        tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True) #运行设备不满足要求时，会自动分配GPU或者CPU
        tfconfig.gpu_options.allow_growth = True   #当使用GPU时，tensorflow运行自动慢慢达到最大GPU的内存
        self.sess = tf.compat.v1.InteractiveSession(config=tfconfig)
        self.sess.run(self.init)
        self.saver = tf.compat.v1.train.Saver([v for v in tf.compat.v1.trainable_variables() if 'Coef' not in v.name and 'UV' not in v.name])
        self.summary_writer = tf.compat.v1.summary.FileWriter(logs_path, graph=tf.compat.v1.get_default_graph())
        # UserWarning: An interactive session is already active. This can cause out-of-memory errors in some cases. 
        # You must explicitly call `InteractiveSession.close()` to release resources held by the other session(s).
        

    def _initialize_weights(self):
        all_weights = dict()

        for i in range(0, self.num_modalities):
            modality = str(i)
            with tf.compat.v1.variable_scope(modality):
                all_weights[modality + '_enc_w0'] = tf.compat.v1.get_variable(modality + "_enc_w0",
                                                                    shape=[self.kernel_size[0], self.kernel_size[0], 
                                                                           1, self.n_hidden[0]],
                                                                    initializer=layers.xavier_initializer_conv2d())
                all_weights[modality + '_enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))
                
                all_weights[modality + '_enc_w1'] = tf.compat.v1.get_variable(modality + "_enc_w1",
                                                                    shape=[self.kernel_size[1], self.kernel_size[1],
                                                                           self.n_hidden[0], self.n_hidden[1]],
                                                                    initializer=layers.xavier_initializer_conv2d())
                all_weights[modality + '_enc_b1'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype=tf.float32))

                all_weights[modality + '_enc_w2'] = tf.compat.v1.get_variable(modality + "_enc_w2",
                                                                    shape=[self.kernel_size[2], self.kernel_size[2],
                                                                           self.n_hidden[1], self.n_hidden[2]],
                                                                    initializer=layers.xavier_initializer_conv2d())
                all_weights[modality + '_enc_b2'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype=tf.float32))

                all_weights[modality + '_dec_w0'] = tf.compat.v1.get_variable(modality + "_dec1_w0",
                                                                    shape=[self.kernel_size[2], self.kernel_size[2],
                                                                           self.n_hidden[1], self.n_hidden[3]],
                                                                    initializer=layers.xavier_initializer_conv2d())
                all_weights[modality + '_dec_b0'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype=tf.float32))

                all_weights[modality + '_dec_w1'] = tf.compat.v1.get_variable(modality + "_dec1_w1",
                                                                    shape=[self.kernel_size[1], self.kernel_size[1],
                                                                           self.n_hidden[0],  self.n_hidden[1]],
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
        print('\nencoder-layer: ')
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
            print(layer1.shape)
            print(layer2.shape)
            print(layer3.shape)
            if (i==0):
                shapes.append(layer1.get_shape().as_list())
                shapes.append(layer2.get_shape().as_list())
                shapes.append(layer3.get_shape().as_list())
        return  latents, shapes
    
    # Building the decoder
    def decoder(self,z, weights,num_modalities, shapes):
        recons={}
        print('\ndecoder-layer: ')
        # Decoder Hidden layer with relu activation 
        for i in range(0,num_modalities):
            modality = str(i)
            shape_de1 = shapes[2]
            layer1 = tf.add(tf.nn.conv2d_transpose(z[modality], weights[modality+'_dec_w0'], \
                                                   tf.stack([tf.shape(self.x['0'])[0],shape_de1[1],shape_de1[2],shape_de1[3]]),\
                                                   strides=[1,2,2,1],padding='SAME'),\
                            weights[modality+'_dec_b0'])
            layer1 = tf.nn.relu(layer1)
            shape_de2 = shapes[1]
            layer2 = tf.add(tf.nn.conv2d_transpose(layer1, weights[modality+'_dec_w1'], \
                                                   tf.stack([tf.shape(self.x['0'])[0],shape_de2[1],shape_de2[2],shape_de2[3]]),\
                                                   strides=[1,2,2,1],padding='SAME'),\
                            weights[modality+'_dec_b1'])
            layer2 = tf.nn.relu(layer2)
            shape_de3= shapes[0]
            layer3 = tf.add(tf.nn.conv2d_transpose(layer2, weights[modality+'_dec_w2'], \
                                                   tf.stack([tf.shape(self.x['0'])[0],shape_de3[1],shape_de3[2],shape_de3[3]]),\
                                                   strides=[1,2,2,1],padding='SAME'),\
                            weights[modality+'_dec_b2'])
            layer3 = tf.nn.relu(layer3)
            recons[modality] = layer3
            print(layer1.shape)
            print(layer2.shape)
            print(layer3.shape)
        return recons

    def partial_fit(self, X ,lr):
        feed_dict={}
        feed_dict[self.learning_rate]= lr
        for i in range(0, len(X)):
            feed_dict[self.x[str(i)]] = X[str(i)]
        cost, summary, _, Coef = self.sess.run(
                (self.loss, self.merged_summary_op, self.optimizer, self.Coef), feed_dict=feed_dict) 
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost, Coef
    
    def initlization(self):
        self.sess.run(self.init)
    
    def reconstruct(self,X):
        feed_dict = {}
        for i in range(0, len(X)):
            feed_dict[self.x[str(i)]] = X[str(i)]
        return self.sess.run(self.x_r, feed_dict = feed_dict)
    
    def transform(self, X):
        feed_dict = {}
        for i in range(0, len(X)):
            feed_dict[self.x[str(i)]] = X[str(i)]
        return self.sess.run(self.z, feed_dict = feed_dict)

    def save_model(self):
        save_path = self.saver.save(self.sess,self.model_path)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("model restored")
        
def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)

    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2   

def thrC(C,ro):  
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))  #np.argsort(a, axis=0) 对a按列进行排序 
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C
    return Cp

# SpectralClustering (https://vimsky.com/examples/detail/python-method-sklearn.cluster.SpectralClustering.html)
def post_proC(C, K, d=6, alpha=8):
    # C: coefficient matrix, K: number of clusters(38), d: dimension of each subspace
    C = 0.5*(C + C.T)
    r = d*K + 1
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    U = U[:,::-1]    
    S = np.sqrt(S[::-1])
    S = np.diag(S)    
    U = U.dot(S)    
    U = normalize(U, norm='l2', axis = 1)       
    Z = U.dot(U.T)
    Z = Z * (Z>0)    
    L = np.abs(Z ** alpha)
    L = L/L.max()   
    L = 0.5 * (L + L.T)    
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

def err_rate(gt_s, s):
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:])
    nmi = normalized_mutual_info_score(gt_s[:], c_x[:])
    ari = adjusted_rand_score(gt_s[:], c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate, nmi, ari, c_x
      
def test_face(Img, Label, CAE, num_class,num_modalities,max_step):
    alpha = max(0.4 - (num_class-1)/10 * 0.1, 0.1)
    for j in range(0,num_modalities):
        modality=str(j)
        Img[modality] = np.array(Img[modality])
        Img[modality] = Img[modality].astype(float)

    label = np.array(Label[:])
    label = label - label.min() + 1
    label = np.squeeze(label)

    CAE.initlization()
    CAE.restore() # restore from pre-trained model

    display_step = 5
    lr = 1.0e-3
    acc=[]
    nmi=[]
    ari=[]
    pre=[]
    
    # fine-tune network
    epoch = 0
    while epoch < max_step:
        epoch = epoch + 1
        cost, Coef = CAE.partial_fit(Img, lr)

        if epoch % display_step == 0:
            print ("epoch: %.1d" % epoch, "cost: %.4f" % (cost/float(batch_size)))
            for i in range(0, num_modalities):
                modality = str(i)
                Coef[modality] = thrC(Coef[modality],alpha)
            y_x, _ = post_proC(Coef['0'], label.max())  #S -clustering
            missrate_x, nm, ar, pr = err_rate(label, y_x)
            ac = 1 - missrate_x
            print ("accuracy: %.4f" % ac, "NMI: %.4f" % nm, "ARI: %.4f" % ar)
            acc.append(ac)
            nmi.append(nm)
            ari.append(ar)
            pre.append(pr)
            
    maxn = np.array([acc]).argmax(axis=1)
    print("Total %d subjects:" % num_class) 
    print("The highest accuracy in %d epoch" % ((maxn[0] + 1) *display_step))
    print("ACC: %.4f%%" % (acc[maxn[0]]*100))
    print("NMI: %.4f%%" % (nmi[maxn[0]]*100))
    print("ARI: %.4f%%" % (ari[maxn[0]]*100))

    return acc[maxn[0]], Coef, nmi[maxn[0]], ari[maxn[0]], pre[maxn[0]]
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TEST')
    parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
    parser.add_argument('--mat', dest='mat', default='C_I', help='path of the dataset')
    parser.add_argument('--model', dest='model', default='C_I',
                        help='name of the model to be saved')
    args = parser.parse_args()
    
    # load face images and labels
    datapath = './Data/'+ args.mat + '.mat'
    data = sio.loadmat(datapath)
    num_modalities = 2
    Img = {}
    moda={}
    d=[32,64]
    n_input={}
    
    # Source->Target
    moda['0'] = data['S_test']
    moda['1'] = data['T_test']
    Label = data['Label_test']
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
    all_subjects = Label.max()  
    num_class = all_subjects + 1
    batch_size = Img['0'].shape[0]  #样本数
    reg1 = 15.5 
    reg2 = 15.5 * 10 ** (num_class / 10.0 - 6.0) 
    model_path = './models_DSC/' + args.model + '.ckpt'
    restore_path = './models/' + args.model + '.ckpt'
    logs_path = './logs'
    tf.compat.v1.reset_default_graph()
    CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2, \
                 kernel_size=kernel_size, batch_size=batch_size, model_path=model_path, \
                 restore_path=restore_path, logs_path=logs_path,num_modalities=num_modalities)

    ACC, S, NMI, ARI, pre = test_face(Img, Label, CAE, num_class, num_modalities, args.epoch)
    pre.astype(np.int32)
    result_path = './models_DSC/results_' + args.model + '.mat'
    sio.savemat(result_path, dict(P=S['0'], Q=S['1'], ACC=ACC, NMI=NMI, ARI=ARI, pre=pre.astype(np.uint8)))
