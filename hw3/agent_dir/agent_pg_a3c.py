from agent_dir.agent import Agent
from agent_dir.a3c import Agents
from agent_dir.a3c import A3CNetwork
from agent_dir.a3c import pipeline
import numpy as np
import tensorflow as tf
import gym
import os
from scipy.misc import imresize

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        # hyperparameters
        

        if args.test_pg:
            #you can load your model here
            tf.reset_default_graph()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.InteractiveSession(config=config)
            coord = tf.train.Coordinator()

            save_path = "a3c_pong/model.ckpt"
            n_threads = 1
            self.input_shape = [80, 80, 1]
            self.output_dim = 3  # {1, 2, 3}
            self.global_network = A3CNetwork(name="global",
                                        input_shape=self.input_shape,
                                        output_dim=self.output_dim)

            if tf.train.get_checkpoint_state(os.path.dirname(save_path)):
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
                saver = tf.train.Saver(var_list=var_list)
                saver.restore(self.sess, save_path)
                print("Model restored to global")
            else:
                init = tf.global_variables_initializer()
                self.sess.run(init)
                print("No model is found")
            

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
        self.init_move = True


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)
        coord = tf.train.Coordinator()

        save_path = "a3c_pong/model.ckpt"
        n_threads = 8 
        input_shape = [80, 80, 1]
        output_dim = 3  # {1, 2, 3}
        global_network = A3CNetwork(name="global",
                                    input_shape=input_shape,
                                    output_dim=output_dim)

        thread_list = []
        env_list = []
        
        try:
            for id in range(n_threads):
                env = gym.make("Pong-v0")

                #if id == 0:
                #    env = gym.wrappers.Monitor(env, "monitors", force=True)

                single_agent = Agents(env=env,
                                     session=sess,
                                     coord=coord,
                                     name="thread_{}".format(id),
                                     global_network=global_network,
                                     input_shape=input_shape,
                                     output_dim=output_dim)
                thread_list.append(single_agent)
                env_list.append(env)

            if tf.train.get_checkpoint_state(os.path.dirname(save_path)):
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
                saver = tf.train.Saver(var_list=var_list)
                saver.restore(sess, save_path)
                print("Model restored to global")
            else:
                init = tf.global_variables_initializer()
                sess.run(init)
                print("No model is found")

            for t in thread_list:
                t.start()

            print("Ctrl + C to close")
            coord.wait_for_stop()

        except KeyboardInterrupt:
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
            saver = tf.train.Saver(var_list=var_list)
            saver.save(sess, save_path)
            print()
            print("=" * 10)
            print('Checkpoint Saved to {}'.format(save_path))
            print("=" * 10)

            print("Closing threads")
            coord.request_stop()
            coord.join(thread_list)

            print("Closing environments")
            for env in env_list:
                env.close()

            sess.close()
            
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
        if self.init_move:
            self.last_observation = pipeline(observation)
            self.init_move = False
            action = self.env.action_space.sample()
            return action
        else :    
            state = pipeline(observation)
            diff = state - self.last_observation
            self.last_observation = state
            
            states = np.reshape(diff, [-1, *self.input_shape])
            feed = {
                self.global_network.states: states
            }

            action = self.sess.run(self.global_network.action_prob, feed)
            action = np.squeeze(action)

            return np.argmax(action) + 1
        #return action

        ##################
        #return self.env.get_random_action()

