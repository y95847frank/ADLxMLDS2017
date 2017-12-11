from agent_dir.agent import Agent
import numpy as np
import tensorflow as tf
import gym

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        # hyperparameters
        self.n_obs = 80 * 80           # dimensionality of observations
        self.h = 200                   # number of hidden layer neurons
        self.n_actions = 3             # number of available actions
        self.learning_rate = 1e-4
        self.gamma = .99               # discount factor for reward
        self.decay = 0.99              # decay rate for RMSProp gradients
        self.save_path='pong/pong.ckpt'

        # gamespace 
        self.env = gym.make("Pong-v0") # environment info
        self.observation = self.env.reset()
        self.prev_x = None
        self.xs,self.rs,self.ys = [],[],[]
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 0

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            tf_model = {}
            with tf.variable_scope('layer_one',reuse=False):
                xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(self.n_obs), dtype=tf.float32)
                tf_model['W1'] = tf.get_variable("W1", [self.n_obs, self.h], initializer=xavier_l1)
            with tf.variable_scope('layer_two',reuse=False):
                xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(self.h), dtype=tf.float32)
                tf_model['W2'] = tf.get_variable("W2", [self.h,self.n_actions], initializer=xavier_l2)

            # tf operations
            def tf_discount_rewards(tf_r): #tf_r ~ [game_steps,1]
                discount_f = lambda a, v: a*self.gamma + v;
                tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[True, False]))
                tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
                return tf_discounted_r

            def tf_policy_forward(x): #x ~ [1,D]
                h = tf.matmul(x, tf_model['W1'])
                h = tf.nn.relu(h)
                logp = tf.matmul(h, tf_model['W2'])
                p = tf.nn.softmax(logp)
                return p


            # tf placeholders
            self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_obs],name="tf_x")
            tf_y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions],name="tf_y")
            tf_epr = tf.placeholder(dtype=tf.float32, shape=[None,1], name="tf_epr")

            # tf reward processing (need tf_discounted_epr for policy gradient wizardry)
            tf_discounted_epr = tf_discount_rewards(tf_epr)
            tf_mean, tf_variance= tf.nn.moments(tf_discounted_epr, [0], shift=None, name="reward_moments")
            tf_discounted_epr -= tf_mean
            tf_discounted_epr /= tf.sqrt(tf_variance + 1e-6)

            # tf optimizer op
            self.tf_aprob = tf_policy_forward(self.tf_x)
            loss = tf.nn.l2_loss(tf_y-self.tf_aprob)
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay)
            tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=tf_discounted_epr)
            train_op = optimizer.apply_gradients(tf_grads)

            # tf graph initialization
            #sess = tf.InteractiveSession()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.InteractiveSession(config=config)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            #tf.initialize_all_variables().run()

            # try load saved model
            saver = tf.train.Saver(tf.global_variables())
            try:
                save_dir = '/'.join(self.save_path.split('/')[:-1])
                ckpt = tf.train.get_checkpoint_state(save_dir)
                load_path = ckpt.model_checkpoint_path
                saver.restore(self.sess, load_path)
            except:
                print ("no saved model to load. starting new session")
                load_was_success = False
            else:
                print ("loaded model: {}".format(load_path))
                saver = tf.train.Saver(tf.global_variables())
                #self.episode_number = int(load_path.split('-')[-1])

        ##################
        


        ##################

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

        # initialize model
        tf_model = {}
        with tf.variable_scope('layer_one',reuse=False):
            xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(self.n_obs), dtype=tf.float32)
            tf_model['W1'] = tf.get_variable("W1", [self.n_obs, self.h], initializer=xavier_l1)
        with tf.variable_scope('layer_two',reuse=False):
            xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(self.h), dtype=tf.float32)
            tf_model['W2'] = tf.get_variable("W2", [self.h,self.n_actions], initializer=xavier_l2)

        # tf operations
        def tf_discount_rewards(tf_r): #tf_r ~ [game_steps,1]
            discount_f = lambda a, v: a*self.gamma + v;
            tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[True, False]))
            tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
            return tf_discounted_r

        def tf_policy_forward(x): #x ~ [1,D]
            h = tf.matmul(x, tf_model['W1'])
            h = tf.nn.relu(h)
            logp = tf.matmul(h, tf_model['W2'])
            p = tf.nn.softmax(logp)
            return p

        # downsampling
        def prepro(I):
            """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
            I = I[35:195] # crop
            I = I[::2,::2,0] # downsample by factor of 2
            I[I == 144] = 0  # erase background (background type 1)
            I[I == 109] = 0  # erase background (background type 2)
            I[I != 0] = 1    # everything else (paddles, ball) just set to 1
            return I.astype(np.float).ravel()

        # tf placeholders
        tf_x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_obs],name="tf_x")
        tf_y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions],name="tf_y")
        tf_epr = tf.placeholder(dtype=tf.float32, shape=[None,1], name="tf_epr")

        # tf reward processing (need tf_discounted_epr for policy gradient wizardry)
        tf_discounted_epr = tf_discount_rewards(tf_epr)
        tf_mean, tf_variance= tf.nn.moments(tf_discounted_epr, [0], shift=None, name="reward_moments")
        tf_discounted_epr -= tf_mean
        tf_discounted_epr /= tf.sqrt(tf_variance + 1e-6)

        # tf optimizer op
        tf_aprob = tf_policy_forward(tf_x)
        loss = tf.nn.l2_loss(tf_y-tf_aprob)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay)
        tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=tf_discounted_epr)
        train_op = optimizer.apply_gradients(tf_grads)

        # tf graph initialization
        #sess = tf.InteractiveSession()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        #tf.initialize_all_variables().run()

        # try load saved model
        saver = tf.train.Saver(tf.global_variables())
        load_was_success = True # yes, I'm being optimistic
        try:
            save_dir = '/'.join(self.save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            saver.restore(sess, load_path)
        except:
            print ("no saved model to load. starting new session")
            load_was_success = False
        else:
            print ("loaded model: {}".format(load_path))
            saver = tf.train.Saver(tf.global_variables())
            self.episode_number = int(load_path.split('-')[-1])


        # training loop
        pg_log = open("model_pg/pg.log", "a")
        while True:
        #     if True: env.render()

            # preprocess the observation, set input to network to be difference image
            cur_x = prepro(self.observation)
            x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.n_obs)
            self.prev_x = cur_x

            # stochastically sample a policy from the network
            feed = {tf_x: np.reshape(x, (1,-1))}
            aprob = sess.run(tf_aprob,feed) ; aprob = aprob[0,:]
            action = np.random.choice(self.n_actions, p=aprob)
            label = np.zeros_like(aprob) ; label[action] = 1

            # step the environment and get new measurements
            self.observation, reward, done, info = self.env.step(action)
            self.reward_sum += reward
            
            # record game history
            self.xs.append(x) ; self.ys.append(label) ; self.rs.append(reward)
            
            if done:
                # update running reward
                self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
                
                # parameter update
                feed = {tf_x: np.vstack(self.xs), tf_epr: np.vstack(self.rs), tf_y: np.vstack(self.ys)}
                _ = sess.run(train_op,feed)
                
                # print progress console
                if self.episode_number % 10 == 0:
                    print ('ep {}: reward: {}, mean reward: {:3f}'.format(self.episode_number, self.reward_sum, self.running_reward))
                else:
                    print ('\tep {}: reward: {}'.format(self.episode_number, self.reward_sum))
                
                # bookkeeping
                self.xs,self.rs,self.ys = [],[],[] # reset game history
                self.episode_number += 1 # the Next Episode
                self.observation = self.env.reset() # reset env
                self.reward_sum = 0
                if self.episode_number % 50 == 0:
                    saver.save(sess, self.save_path, global_step=self.episode_number)
                    print ("SAVED MODEL #{}".format(self.episode_number))
                if self.episode_number % 500 == 0:
                    pg_log.write('ep {}: reward: {}\n'.format(self.episode_number, self.reward_sum))
        ##################


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        # preprocess the observation, set input to network to be difference image

        # downsampling
        def prepro(I):
            """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
            I = I[35:195] # crop
            I = I[::2,::2,0] # downsample by factor of 2
            I[I == 144] = 0  # erase background (background type 1)
            I[I == 109] = 0  # erase background (background type 2)
            I[I != 0] = 1    # everything else (paddles, ball) just set to 1
            return I.astype(np.float).ravel()

        cur_x = prepro(observation)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.n_obs)
        self.prev_x = cur_x

        # stochastically sample a policy from the network
        feed = {self.tf_x: np.reshape(x, (1,-1))}
        aprob = self.sess.run(self.tf_aprob,feed) ; aprob = aprob[0,:]
        action = np.random.choice(self.n_actions, p=aprob)
        return action+1

        ##################
        #return self.env.get_random_action()

