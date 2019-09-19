import gym
import numpy as np
import micronet.experiments.sequential_pool.sequential_pool as sp
import micronet.experiments.sequential_pool.efficientnet_utils as efnet_utils
import tensorflow as tf
import random


class RandomPixelOrder:

    def __iter__(self):
        self._ordered_indices = list(range(0, sp.NUM_PIXELS))
        random.shuffle(self._ordered_indices)
        return iter(self._ordered_indices)

    def __next__(self):
        raise NotImplemented


class SpreadPixelOrder:
    ORDER_MAP = np.array(
        [
            [9,  36, 47, 13, 34, 19, 11],
            [42,  5, 17, 25, 38,  7, 45],
            [28, 44,  1, 29,  3, 46, 33],
            [16, 24, 21,  0, 31, 23, 15],
            [32, 39,  4, 30,  2, 43, 27],
            [48,  8, 37, 26, 18,  6, 40],
            [12, 20, 35, 14, 41, 22, 10]
        ])

    def __iter__(self):
        self._i = 0
        _, self.ordered_indexes = zip(*sorted([(val, idx) for idx, val in
                                      np.ndenumerate(self.ORDER_MAP)]))
        return self

    @staticmethod
    def _coord_to__index(x, y):
        return x + y * sp.MASK_WIDTH

    def __next__(self):
        if self._i >= len(self.ordered_indexes):
            raise StopIteration
        res = self._coord_to__index(*self.ordered_indexes[self._i])
        self._i += 1
        return res

class PoolEnv(gym.Env):
    """Efficientnet pool layer evaluation order enviroment.

    Observations:
        * num of pixels evaluated so far (int).
        * a few summaries of pool output activations:
            * sum
            * mean
            * SD

    Actions:
        * 0- evaluate another pixel (the enviroment is configured with an
                evaluation order).
        * 1- classify and end episode.

    Rewards:
        * 100 if the classify action is taken and the classification is correct.
        * 0 otherwise.
    """

    NUM_PIXELS = 7*7
    NUM_ACTIONS = 2 # eval or classify
    STOP_ACTION = 1
    SUCCESS_REWARD = 100.0
    BEST_EFFORT_REWARD = 0.0 # 50.0
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
        self._create_input_net()
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
        # self._pixel_order = SpreadPixelOrder()
        self._pixel_order = RandomPixelOrder()
        self._pixel_iter = None

        self.action_space = gym.spaces.Discrete(self.NUM_ACTIONS)
        bounds_low = np.array([0, *([-np.inf] * 1280)])
        bounds_high = np.array([sp.NUM_PIXELS, *([np.inf]* 1280)])
        # First element is the number of pixels evaluated.
        # We can't use a tuple space, as they are not supported by the
        # deepq algorithm as implemented by openai, and I want to stick with
        # the standard implementation this time.
        self.observation_space = gym.spaces.Box(low=bounds_low,
                                                high=bounds_high,
                                                dtype=np.float32)
        self._last_observation = None

        self._accuracy_target = 0.765
        self._step_reward = -1.5

        # Stats
        self._accuracy = self._accuracy_target
        self._ave_pixels = 0.0
        self._alpha = 0.001
        self._actions_so_far = 0
        self.best_effort_accuracy = 0.80
        self.best_effort_accuracy_lower_bound = 0.4

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

        # Keep track of the best-effort accuracy so that we can add some sanity
        # checks to insure the classification net is correctly initialized.
        is_best_effort_correct = self._best_prediction == self._label
        self.best_effort_accuracy = \
            self.best_effort_accuracy + self._alpha * \
            (int(is_best_effort_correct - self.best_effort_accuracy))
        assert self.best_effort_accuracy > self.best_effort_accuracy_lower_bound


        # Reset mask.
        self._mask = np.full((sp.NUM_PIXELS, ), False)

        # Reset pixel order.
        self._pixel_iter = iter(self._pixel_order)

        # Reset counters.
        self._actions_so_far = 0

        # First observation
        obs = self._observation(self._unmasked_pool_input)
        self._last_observation = obs
        return obs

    def step(self, action):
        if not self.action_space.contains(action):
            raise Exception('Invalid action: {}'.format(action))
        self._actions_so_far += 1
        if action != self.STOP_ACTION:
            next_pixel = next(self._pixel_iter)
            assert not self._mask[next_pixel]
            self._mask[next_pixel] = True
        remaining_pixels = sp.NUM_PIXELS - self._actions_so_far
        fin = remaining_pixels == 0 or action == self.STOP_ACTION
        if fin:
            done = True
            reward = self.calculate_reward()
            # Aux data
            correct = float(reward == self.SUCCESS_REWARD)
            self._accuracy = self._accuracy + \
                             self._alpha * (correct - self._accuracy)
            pixel_count = np.sum(self._mask.astype(np.int32))
            self._ave_pixels = self._ave_pixels + \
                             self._alpha * (pixel_count - self._ave_pixels)
            aux_data = {'accuracy': self._accuracy,
                        'num_eval_pixels': pixel_count}
        else:
            done = False
            reward = self._step_reward
            aux_data = None
        # Calculate pool output.
        pool_input = self._unmasked_pool_input * \
                     self._reshape_for_mask_op(self._mask)
        obs = self._observation(pool_input)
        self._last_observation = obs
        return obs, reward, done, aux_data

    def log_stats(self, logger):
        logger.record_tabular("accuracy", self._accuracy)
        logger.record_tabular("ave pixels", self._ave_pixels)
        logger.record_tabular("step_reward", self._step_reward)

    def _observation(self, pool_input):
        pool_output = self._tf_sess.run(
            self._pool_ouput_tensor,
            feed_dict={self._pool_input_placeholder: pool_input})
        obs = np.array([self._actions_so_far])
        obs = np.concatenate((obs, pool_output))
        return obs

    @staticmethod
    def _reshape_for_mask_op(mask):
        return np.reshape(mask.astype(np.float32), (7,7,1))

    def calculate_reward(self):
        # Shortcut for the case where no pixels are evaluated before
        # classifying. This is a degenerative case and should probably be
        # prevented elsewhere.
        if np.sum(self._mask) == 0:
            return self._step_reward
        pool_inputs = self._unmasked_pool_input * \
                      self._reshape_for_mask_op(self._mask)
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
        #with tf.variable_scope(head_scope, auxiliary_name_scope=False) as vs:
        #    with tf.name_scope(vs.original_name_scope):
        # Using trailing '/' trick to prevent new scope creation.
        with tf.name_scope('efficientnet-b0/head/'):
                pool_ouput_tensor_full = tf.keras.layers.GlobalAveragePooling2D(
                    data_format='channels_last')(self._pool_input_placeholder)
                classes = 1000
                self._pool_ouput_tensor = tf.squeeze(pool_ouput_tensor_full)
                logits = tf.keras.layers.Dense(classes, name='dense')(pool_ouput_tensor_full)
                assert len(logits.shape) == 2
                assert logits.shape[1] == classes
                self._prediction_tensor = tf.arg_max(logits, 1)
                # self._dropout = tf.keras.layers.Dropout(self._global_params.dropout_rate)

    def load_weights(self):
        eval_ckpt_driver = efnet_utils.EvalCkptDriver('efficientnet-b0')
        b0_path = './resources/efficientnet-b0/'
        eval_ckpt_driver.restore_model(self._tf_sess, b0_path, enable_ema=True,
                                       export_ckpt=None)
