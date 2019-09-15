import gym
import numpy as np
import micronet.experiments.sequential_pool.sequential_pool as sp
import micronet.experiments.sequential_pool.efficientnet_utils as efnet_utils
import tensorflow as tf


class PoolEnv(gym.Env):
    """Efficientnet pool layer evaluation order enviroment.

    Observations:
        * current pool evaluations, 2^(7x7) possible states encoded into 4
          integers.
        * a few summaries of pool output activations:
            * sum
            * mean
            * SD

    Actions:
        * 0-49, representing adding the chosen pixel to the current evaluation
          mask. Where the mask represents pixels that _are_ evaluated.
        * 50, meaning that agent chooses to use the current mask to estimate
          the image classification.

    Rewards:
        * 100 if the classify action is taken and the classification is correct.
        * 50 if the classify action is taken and the classification is incorrect
          but equal to the best guess (guess given all pixels are used).
        * 0 otherwise.

    The episode terminates if the classify action is taken (50).
    """

    NUM_PIXELS = 7*7
    NUM_ACTIONS = NUM_PIXELS + 1
    SUCCESS_REWARD = 100.0
    BEST_EFFORT_REWARD = 50.0
    FAILURE_REWARD = 0.0

    def __init__(self, tf_sess):
        self.batch_size = 1
        self._tf_sess = tf_sess
        self._img_tensor = None
        self._label_tensor = None
        self._best_prediction = None
        self._encoded_state_tensor = None
        self._prediction_tensor = None
        self._pool_input_tensor = None
        self._pool_input_placeholder = None
        self._create_model_base()
        self._create_classification_head()
        # doing it in main where we can place it last in the initialization
        # order.
        # self.load_weights()

        self._img = None
        self._label = None
        self._mask = None
        self._best_prediction = None
        self._unmasked_pool_input = None

        self.action_space = gym.spaces.Discrete(self.NUM_ACTIONS)
        # high = np.array([
        #     sp.MaskEncoding.max_bottom,
        #     sp.MaskEncoding.max_right,
        #     sp.MaskEncoding.max_center,
        #     sp.MaskEncoding.max_count,
        #     np.finfo(np.float32).max, # sum
        #     np.finfo(np.float32).max, # mean
        #     np.finfo(np.float32).max  # SD
        # ])
        # self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        # encoded state
        pool_output_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                           shape=(1280,), dtype=np.float32)
        mask_space = gym.spaces.MultiBinary(n=sp.NUM_ACTIONS)
        self.observation_space = gym.spaces.Tuple(
            (pool_output_space, mask_space))

        # Stats
        self._accuracy = 0.0
        self._accuracy_alpha = 0.01
        self._pixel_tally = 0

    def reset(self):
        # Get the next image and its label.
        self._img, self._label = self._tf_sess.run([self._img_tensor,
                                                    self._label_tensor])
        # Calculate best prediction and the un-masked pool inputs.
        self._unmasked_pool_input = self._tf_sess.run(
            self._pool_input_tensor,
            feed_dict={self._img_placeholder: self._img})
        self._best_prediction = self._tf_sess.run(
            self._prediction_tensor,
            feed_dict={self._pool_input_placeholder: self._unmasked_pool_input})

        # Reset mask.
        self._mask = np.full((sp.NUM_PIXELS, ), False)

        # Reset stats
        self._pixel_tally = 0

    def step(self, action):
        if not self.action_space.contains(action):
            raise Exception('Invalid action: {}'.format(action))
        already_unmasked = self._mask[action]
        if already_unmasked:
            raise Exception('Action already taken: {}'.format(action))
        # Update mask.
        self._mask[action] = True
        # Calculate pool output.
        pool_input = self._unmasked_pool_input * self._mask.as_type(np.float32)
        pool_output = self._tf_sess.run(
            self._pool_ouput_tensor,
            feed_dict={self._pool_input_placeholder: pool_input})
        action_mask = np.zeros(sp.NUM_ACTIONS)
        action_mask[:-1] = self._mask
        #observations = (self._mask, pool_output)
        observations = (action_mask, pool_output)
        if action == sp.STOP_ACTION:
            done = True
            reward = self.calculate_reward()
            aux_data = None
        else:
            done = False
            reward = 0
            # Aux data
            correct = float(reward == self.SUCCESS_REWARD)
            self._accuracy = self._accuracy - \
                             self._accuracy_alpha * (correct - self._accuracy)
            aux_data = {'accuracy': self._accuracy,
                        'num_eval_pixels': self._pixel_tally}
        self._pixel_tally += 1
        return observations, reward, done, aux_data

    def calculate_reward(self):
        pool_inputs = self._unmasked_pool_input * self._mask.as_type(np.float32)
        prediction = self._tf_sess.run(
            [self._prediction_tensor],
            feed_dict={self._pool_input_placeholder: pool_inputs})
        if prediction == self._label:
            reward = self.SUCCESS_REWARD
        elif prediction == self._best_prediction:
            reward = self.BEST_EFFORT_REWARD
        else:
            reward = self.FAILURE_REWARD
        return reward

    def _create_input_net(self):
        iterator = sp.image_iterator()
        self._img_tensor, self._label_tensor = iterator.get_next()

    def _create_model_base(self):
        self._img_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=(self.batch_size, *sp.IMAGE_SHAPE),
            name='img_placeholder')
        # Note: it probably makes more sense having the encoding done by the
        # agent. However, this way reduces data communication.
        self._pool_input_tensor = sp.efficientnet_until_pool(
            self._img_placeholder)

    def _create_classification_head(self):
        # A little hacky for the moment. Recreate the scope so that we can
        # populate the network.
        self._pool_input_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=(None, 7, 7, 1280)
        )
        with tf.variable_scope('efficientnet-b0'):
            with tf.variable_scope('head'):
                pool_ouput_tensor_full = tf.keras.layers.GlobalAveragePooling2D(
                    data_format='channels_first')(self._pool_input_placeholder)
                classes = 1000
                self._pool_ouput_tensor = tf.squeeze(pool_ouput_tensor_full)
                logits = tf.keras.layers.Dense(classes)(pool_ouput_tensor_full)
                assert len(logits.shape) == 2
                assert logits.shape[1] == classes
                self._prediction_tensor = tf.arg_max(logits, 1)
                # self._dropout = tf.keras.layers.Dropout(self._global_params.dropout_rate)

    def load_weights(self):
        eval_ckpt_driver = efnet_utils.EvalCkptDriver('efficientnet-b0')
        b0_path = '/home/k/Sync/micronet/resources/efficientnet-b0/'
        eval_ckpt_driver.restore_model(self._tf_sess, b0_path, enable_ema=True,
                                       export_ckpt=None)
