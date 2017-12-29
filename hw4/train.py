import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import sys, os
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
import skimage.io
import time
import datetime
from util import gen_test
import scipy.misc

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x
)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

class G_conv(object):
    def __init__(self):
        self.name = 'G_conv'
        self.size = 4
        self.channel = 3

    def __call__(self, z):
        with tf.variable_scope(self.name) as scope:
            g = tcl.fully_connected(z, self.size * self.size * 256, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            g = tf.reshape(g, (-1, self.size, self.size, 256))  # size
            g = tcl.conv2d_transpose(g, 128, 5, stride=2, # size*2
                                    activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 64, 5, stride=2, # size*4
                                    activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 32, 5, stride=2, # size*8
                                    activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            
            g = tcl.conv2d_transpose(g, self.channel, 5, stride=2, # size*16
                                        activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            return g
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_conv(object):
    def __init__(self):
        self.name = 'D_conv'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 64
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
                        stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 16x16x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=size * 4, kernel_size=4, # 8x8x256
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=size * 8, kernel_size=4, # 4x4x512
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

            shared = tcl.flatten(shared)
    
            d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
            q = tcl.fully_connected(shared, 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            q = tcl.fully_connected(q, 23, activation_fn=None) # 10 classes
            return d, q
            
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

# for test
def sample_y(m, n, ind):
    y = np.zeros([m,n])
    for i in range(m):
        y[i,i/4] = 1
    return y

def concat(z,y):
    return tf.concat([z,y],1)

def conv_concat(x,y):
    bz = tf.shape(x)[0]
    y = tf.reshape(y, [bz, 1, 1, 23])
    return tf.concat([x, y*tf.ones([bz, 64, 64, 23])], 3)  # bzx28x28x11

def concat(z,y):
    return tf.concat([z,y],1)

class WGAN():
    def __init__(self, generator, discriminator, train_X, train_y, test_y, ckpt):
        self.generator = generator
        self.discriminator = discriminator
        self.train_X = train_X
        self.train_y = train_y
        self.test_y = test_y

        # data
        self.z_dim = 100
        self.y_dim = 23 # condition
        self.size = 64
        self.channel = 3

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])

        # nets
        self.G_sample = self.generator(concat(self.z, self.y))

        self.D_real, _ = self.discriminator(conv_concat(self.X, self.y))
        self.D_fake, _ = self.discriminator(conv_concat(self.G_sample, self.y), reuse = True)
        
        # loss
        self.D_loss = - tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake)
        self.G_loss = - tf.reduce_mean(self.D_fake)
        #self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        #self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

        # solver
        self.D_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.G_loss, var_list=self.generator.vars)

        self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.discriminator.vars]
        
        self.saver = tf.train.Saver(max_to_keep=10)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        self.ckpt_dir = ckpt
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

    def test(self, path):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        ID, y_s = gen_test(path)
        if not os.path.exists('samples'):
            os.makedirs('samples')
        for i in range(5):
            samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(y_s.shape[0], self.z_dim)})
            for (c,s) in zip(ID, samples):
                scipy.misc.imsave('samples/sample_{}_{}.png'.format(c, i+1), s)
                #skimage.io.imsave('samples/sample_{}_{}.png'.format(c, i+1), s)

    def train(self, sample_dir, training_epoches = 1000000, batch_size = 64):
        fig_count = 0

        self.sess.run(tf.global_variables_initializer())

        counter = 0
        nb_batches = int(self.train_X.shape[0] / batch_size)
        t0 = time.time()
        
        for epoch in range(training_epoches):
            # update D
            
            #X_b,y_b = self.data(batch_size)
            n_d = 100 if epoch < 50 or (epoch+1) % 500 == 0 else 5
            for _ in range(n_d):
            	
                index = counter % nb_batches
                counter += 1
                X_b = self.train_X[index * batch_size:(index + 1) * batch_size]
                y_b = self.train_y[index * batch_size:(index + 1) * batch_size]

                self.sess.run(self.clip_D)
                self.sess.run(
                    self.D_solver,
                    feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(batch_size, self.z_dim)}
                )
            # update G
            k = 1
            for _ in range(k):
                self.sess.run(
                    self.G_solver,
                    feed_dict={self.y:y_b, self.z: sample_z(batch_size, self.z_dim)}
                )
            
            # save img, model. print loss
            if epoch % 500 == 0 or epoch < 100:
                D_loss_curr = self.sess.run(
                        self.D_loss,
                        feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
                G_loss_curr = self.sess.run(
                        self.G_loss,
                        feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; {}'.format(epoch, D_loss_curr, G_loss_curr, ((str(datetime.datetime.now())).split(' ')[1]).split('.')[0]))

                if epoch % 1000 == 0:
                    y_s = self.test_y
                    samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(1, self.z_dim)})
                    skimage.io.imsave(os.path.join(sample_dir,'epoch_{0:03d}_generated.png'.format(epoch)), samples[0])

                if epoch % 1000 == 0:
                    self.saver.save(self.sess, os.path.join(self.ckpt_dir, "wgan_conv_{0:03d}.ckpt").format(epoch))


if __name__ == '__main__':
    # save generated images
    sample_dir = 'wgraph/'

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    # param
    generator = G_conv()
    discriminator = D_conv()

    train_X = np.load('data/img.npy')
    train_y = np.load('data/text.npy')

    test_y = np.load('data/special_text.npy')
    # run
    wgan_c = WGAN(generator, discriminator, train_X, train_y, test_y, 'trained_model')
    if sys.argv[1] == 'test':
        wgan_c.test(sys.argv[2])
    else:
        wgan_c.train(sample_dir)
