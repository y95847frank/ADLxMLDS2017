from agent_dir.agent import Agent
import tensorflow as tf
import gym

import numpy as np
import random as ran
import datetime

from collections import deque

MINIBATCH = 32
REPLAY_MEMORY = deque()

HISTORY_STEP =4
FRAMESKIP = 4
TRAIN_INTERVAL = 4
NO_STEP = 30
TRAIN_START = 10000
FINAL_EXPLORATION = 0.05
TARGET_UPDATE = 1000
UPDATE = 4

MEMORY_SIZE = 10000
EXPLORATION = 1000000
START_EXPLORATION = 1.


OUTPUT = 3
HEIGHT =84
WIDTH = 84

LEARNING_RATE = 0.0001

DISCOUNT = 0.99
model_path = "save/Breakout.ckpt"

def model(input1, f1, f2, f3, w1, w2):
    c1 = tf.nn.relu(tf.nn.conv2d(input1, f1, strides=[1, 4, 4, 1],data_format="NHWC", padding = "VALID"))
    c2 = tf.nn.relu(tf.nn.conv2d(c1, f2, strides=[1, 2, 2, 1],data_format="NHWC", padding="VALID"))
    c3 = tf.nn.relu(tf.nn.conv2d(c2, f3, strides=[1,1,1,1],data_format="NHWC", padding="VALID"))

    l1 = tf.reshape(c3, [-1, w1.get_shape().as_list()[0]])
    l2 = tf.nn.relu(tf.matmul(l1, w1))

    pyx = tf.matmul(l2, w2)
    return pyx

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        ##################
        # YOUR CODE HERE #
        tf.reset_default_graph()
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            self.X = tf.placeholder("float", [None, 84, 84, 4])

            f1 = tf.get_variable("f1", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            f2 = tf.get_variable("f2", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            f3 = tf.get_variable("f3", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

            w1 = tf.get_variable("w1", shape=[7*7*64,512], initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.get_variable("w2", shape=[512, OUTPUT], initializer=tf.contrib.layers.xavier_initializer())

            self.py_x = model(self.X, f1, f2, f3 , w1, w2)

            f1_r = tf.get_variable("f1_r", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            f2_r = tf.get_variable("f2_r", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            f3_r = tf.get_variable("f3_r", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

            w1_r = tf.get_variable("w1_r", shape=[7*7*64,512], initializer=tf.contrib.layers.xavier_initializer())
            w2_r = tf.get_variable("w2_r", shape=[512, OUTPUT], initializer=tf.contrib.layers.xavier_initializer())

            py_x_r = model(self.X, f1_r, f2_r,f3_r, w1_r, w2_r)
            a= tf.placeholder(tf.int64, [None])
            y = tf.placeholder(tf.float32, [None])
            a_one_hot = tf.one_hot(a, OUTPUT, 1.0, 0.0)
            q_value = tf.reduce_sum(tf.multiply(self.py_x, a_one_hot), reduction_indices=1)
            error = tf.abs(y - q_value)

            quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
            linear_part = error - quadratic_part
            loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

            optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE,momentum=0.95,epsilon= 0.01)
            train = optimizer.minimize(loss)

            self.saver = tf.train.Saver(max_to_keep=None)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.saver.restore(self.sess, save_path = 'breakout/Breakout.ckpt-172')
        


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        frame = 0
        e = 1.
        X = tf.placeholder("float", [None, 84, 84, 4])

        f1 = tf.get_variable("f1", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        f2 = tf.get_variable("f2", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        f3 = tf.get_variable("f3", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        w1 = tf.get_variable("w1", shape=[7*7*64,512], initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable("w2", shape=[512, OUTPUT], initializer=tf.contrib.layers.xavier_initializer())

        py_x = model(X, f1, f2, f3 , w1, w2)

        f1_r = tf.get_variable("f1_r", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        f2_r = tf.get_variable("f2_r", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        f3_r = tf.get_variable("f3_r", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        w1_r = tf.get_variable("w1_r", shape=[7*7*64,512], initializer=tf.contrib.layers.xavier_initializer())
        w2_r = tf.get_variable("w2_r", shape=[512, OUTPUT], initializer=tf.contrib.layers.xavier_initializer())

        py_x_r = model(X, f1_r, f2_r,f3_r, w1_r, w2_r)

        rlist=[0]
        recent_rlist=[0]

        episode = 0
        epoch = 0
        epoch_score = deque()
        epoch_Q = deque()
        epoch_on = False
        average_Q = deque()
        average_reward = deque()
        no_life_game = False

        a= tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])
        a_one_hot = tf.one_hot(a, OUTPUT, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(py_x, a_one_hot), reduction_indices=1)
        error = tf.abs(y - q_value)

        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE,momentum=0.95,epsilon= 0.01)
        train = optimizer.minimize(loss)

        saver = tf.train.Saver(max_to_keep=None)

        fp = open('reward_breakout.log', 'a')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, save_path = 'save/Breakout.ckpt-95')
            sess.run(tf.global_variables_initializer())
            sess.run(w1_r.assign(w1))
            sess.run(w2_r.assign(w2))
            sess.run(f1_r.assign(f1))
            sess.run(f2_r.assign(f2))
            sess.run(f3_r.assign(f3))

            while np.mean(recent_rlist) < 500 :
                episode += 1

                if len(recent_rlist) > 100:
                    del recent_rlist[0]

                rall = 0
                d = False
                ter = False
                count = 0
                s = self.env.reset()
                avg_max_Q = 0
                avg_loss = 0

                while not d :
                    # env.render()

                    frame +=1
                    count+=1

                    if e > FINAL_EXPLORATION and frame > TRAIN_START:
                        e -= 4 * (START_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION

                    state = np.reshape(s, (1, 84, 84, 4))
                    Q = sess.run(py_x, feed_dict = {X : state})
                    average_Q.append(np.max(Q))
                    avg_max_Q += np.max(Q)

                    if e > np.random.rand(1):
                        action = np.random.randint(OUTPUT)
                    else:
                        action = np.argmax(Q)

                    if action == 0:
                        real_a = 1
                    elif action == 1:
                        real_a = 2
                    else:
                        real_a = 3

                    s1, r, d, l = self.env.step(real_a)
                    ter = d
                    reward= np.clip(r, -1,1)

                    REPLAY_MEMORY.append((np.copy(s), np.copy(s1), action ,reward, ter))

                    s = s1

                    if len(REPLAY_MEMORY) > MEMORY_SIZE:
                        REPLAY_MEMORY.popleft()
                    rall += r

                    if frame > TRAIN_START and frame % UPDATE == 0:
                        s_stack = deque()
                        a_stack = deque()
                        r_stack = deque()
                        s1_stack = deque()
                        d_stack = deque()
                        y_stack = deque()

                        sample = ran.sample(REPLAY_MEMORY, MINIBATCH)

                        for _s, s_r, a_r, r_r, d_r in sample:
                            s_stack.append(_s)
                            a_stack.append(a_r)
                            r_stack.append(r_r)
                            s1_stack.append(s_r)
                            d_stack.append(d_r)

                        d_stack = np.array(d_stack) + 0

                        Q1 = sess.run(py_x_r, feed_dict={X: np.float32(np.array(s1_stack))})

                        y_stack = r_stack + (1 - d_stack) * DISCOUNT * np.max(Q1, axis=1)

                        sess.run(train, feed_dict={X: np.array(s_stack), y: y_stack, a: a_stack})

                        if frame % TARGET_UPDATE == 0 :
                            sess.run(w1_r.assign(w1))
                            sess.run(w2_r.assign(w2))
                            sess.run(f1_r.assign(f1))
                            sess.run(f2_r.assign(f2))
                            sess.run(f3_r.assign(f3))

                    if (frame - TRAIN_START) % 50000 == 0:
                        epoch_on = True

                if epoch_on:
                    epoch += 1
                    epoch_score.append(np.mean(average_reward))
                    epoch_Q.append(np.mean(average_Q))

                    epoch_on = False
                    average_reward = deque()
                    average_Q = deque()

                    save_path = saver.save(sess, model_path, global_step=(epoch-1))
                    print("Model(episode :",episode, ") saved in file: ", save_path , " Now time : " ,datetime.datetime.now())

                recent_rlist.append(rall)
                rlist.append(rall)
                average_reward.append(rall)

                if episode % 10 == 0:
                    print("Episode:{0:6d} | Frames:{1:9d} | Steps:{2:5d} | Reward:{3:3.0f} | e-greedy:{4:.5f} | Avg_Max_Q:{5:2.5f} | "
                      "Recent reward:{6:.5f}  ".format(episode,frame, count, rall, e, avg_max_Q/float(count),np.mean(recent_rlist)))
                    s = "%d\t%lf\n" % (episode, np.mean(recent_rlist))
                    fp.write(s)
        
        ##################


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        s = np.reshape(observation, (1, 84, 84, 4))
        
        Q = self.sess.run(self.py_x, feed_dict = {self.X : s})
        
        action = np.argmax(Q)

        if action == 0:
            real_a = 1
        elif action == 1:
            real_a = 2
        else:
            real_a = 3

        return real_a
        ##################
        #return self.env.get_random_action()
