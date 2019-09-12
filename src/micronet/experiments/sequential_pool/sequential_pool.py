import numpy as np
import tensorflow as tf
import micronet.dataset.imagenet as imagenet_ds
import micronet.experiments.sequential_pool.efficientnet_builder as efnet_builder
MASK_WIDTH = 7
MASK_HEIGHT = 7
NUM_PIXELS = MASK_WIDTH * MASK_HEIGHT
NUM_ACTIONS = NUM_PIXELS + 1
STOP_ACTION = NUM_ACTIONS - 1
SUCCESS_ACCURACY = 0.76
SUCCESS_REWARD = 100
BEST_EFFORT_REWARD = 50
DISCOUNT_RATE = 0.95
ALPHA = 0.2
IMAGE_SHAPE = (224, 224, 3)
ENCODED_STATE_SHAPE = [7] # 4 for MaskEncoding, 3 for PoolEncoding.
b0_path = '/home/k/Sync/micronet/resources/efficientnet-b0/'
b0_meta_path = b0_path + 'model.ckpt.meta'
imported_effnet = tf.train.import_meta_graph(b0_meta_path)


def action_coords(idx_):
    return (idx_ / MASK_WIDTH, idx_ % MASK_WIDTH)


class MaskEncoding:

    buckets_flat = np.array([
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 4, 4, 5, 1, 1],
        [0, 0, 4, 5, 5, 1, 1],
        [2, 2, 6, 7, 7, 3, 3],
        [2, 2, 2, 3, 3, 3, 3],
        [2, 2, 2, 3, 3, 3, 3],
    ])

    buckets = np.array([
        [
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1]
        ], [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ], [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]
    ])
    right_idx = 0
    bottom_idx = 1
    center_idx = 2

    def __init__(self, bottom_count, right_count, center_count, total):
        self.bottom_count = bottom_count
        self.right_count = right_count
        self.center_count = center_count
        self.total = total

    def __hash__(self):
        return hash((self.bottom_count, self.right_count, self.center_count,
                    self.total))

    def __eq__(self, other):
        return (self.bottom_count, self.right_count,
                self.center_count, self.total) == \
               (other.bottom_count, other.right_count,
                other.center_count, other.total)

    @classmethod
    def from_mask(cls, mask):
        """Encode a 7x7 mask into 4 integers.

        params:
            mask: 2D array, 7x7 mask for the global pool layer outputs.
        returns:


        The integers are:

            * bottom count
            * right count
            * center count
            * total

        Top, left and outer counts can be inferred by subtracting
        top, left and center from the total.

        This encoding was chosen so that:
          * it would be easy to find all states for a given total mask
            count.
          * coarse coding is used instead of state aggregation. I have seen
            claims that coarse coding is typically more effective than
            state aggregation. This, however, does warrant some investigation.

        Only 5-bits of each integer are used.

        Coarse code
        -----------

             0     1     2     3     4     5     6
          +-----+-----+-----+-----+-----+-----+-----+
        0 | 000 | 000 | 000 | 000 | 001 | 001 | 001 |
          +-----+-----+-----+-----+-----+-----+-----+
        1 | 000 | 000 | 000 | 000 | 001 | 001 | 001 |
          +-----+-----+-----+-----+-----+-----+-----+
        2 | 000 | 000 | 100 | 100 | 101 | 001 | 001 |
          +-----+-----+-----+-----+-----+-----+-----+
        3 | 000 | 000 | 100 | 111 | 101 | 001 | 001 |
          +-----+-----+-----+-----+-----+-----+-----+
        4 | 010 | 010 | 110 | 111 | 111 | 011 | 011 |
          +-----+-----+-----+-----+-----+-----+-----+
        5 | 010 | 010 | 010 | 011 | 011 | 011 | 011 |
          +-----+-----+-----+-----+-----+-----+-----+
        6 | 010 | 010 | 010 | 011 | 011 | 011 | 011 |
          +-----+-----+-----+-----+-----+-----+-----+

        As decimal
        ----------
        Note, this is not a state aggregation, but the decimal
        representation of the coarse code.

             0     1     2     3     4     5     6
          +-----+-----+-----+-----+-----+-----+-----+
        0 |  0  |  0  |  0  |  0  |  1  |  1  |  1  |
          +-----+-----+-----+-----+-----+-----+-----+
        1 |  0  |  0  |  0  |  0  |  1  |  1  |  1  |
          +-----+-----+-----+-----+-----+-----+-----+
        2 |  0  |  0  |  4  |  4  |  5  |  1  |  1  |
          +-----+-----+-----+-----+-----+-----+-----+
        3 |  0  |  0  |  4  |  5  |  5  |  1  |  1  |
          +-----+-----+-----+-----+-----+-----+-----+
        4 |  2  |  2  |  6  |  7  |  7  |  3  |  3  |
          +-----+-----+-----+-----+-----+-----+-----+
        5 |  2  |  2  |  2  |  3  |  3  |  3  |  3  |
          +-----+-----+-----+-----+-----+-----+-----+
        6 |  2  |  2  |  2  |  3  |  3  |  3  |  3  |
          +-----+-----+-----+-----+-----+-----+-----+
        """
        mask = np.array(mask)

        right_count = np.sum(np.multiply(np.bitwise_and(
            cls.buckets_flat, b'001'), mask))
        bottom_count = np.sum(np.multiply(np.right_shift(
            np.bitwise_and(cls.buckets_flat, b'0101'), 1), mask))
        center_count = np.sum(np.multiply(np.right_shift(
            np.bitwise_and(cls.buckets_flat, b'0101'), 1), mask))
        total = np.sum(mask)
        assert total <= NUM_PIXELS
        return MaskEncoding(right_count, bottom_count, center_count, total)

    @classmethod
    def encode_net(cls, mask):
        # square_mask = tf.reshape(mask, (-1, 7, 7))
        flattened_buckets = np.reshape(cls.buckets, (3, NUM_PIXELS))
        right_count = tf.math.reduce_sum(
            mask * flattened_buckets[cls.right_idx], axis=[1])
        bottom_count = tf.math.reduce_sum(
            mask * flattened_buckets[cls.bottom_idx], axis=[1])
        center_count = tf.math.reduce_sum(
            mask * flattened_buckets[cls.center_idx], axis=[1])
        total = tf.math.reduce_sum(mask, axis=[1])
        return tf.stack([right_count, bottom_count, center_count, total],
                         axis=1)


class PoolEncoding:

    @staticmethod
    def encode_net(pool_output):
        """

        :param pool_output: 320xb tensor.
        :return: 3xb tensor (average, total, SD).
        """
        abs_pool = tf.abs(pool_output)
        # axis 0 is the batch axis.
        sum = tf.math.reduce_sum(abs_pool, axis=[1])
        mean, variance = tf.nn.moments(abs_pool, axes=[1])
        sd = tf.sqrt(variance)
        # We use stack, as we wish there to be an extra dimension:
        # previously, all were (batch,), after we want (batch, 3)
        state = tf.stack((sum, mean, sd), axis=1)
        return state


# def action_value_model(state, scope_name='action_value'):
def action_value_model(scope_name='action_value'):
    with tf.variable_scope(scope_name):
        l = tf.keras.layers
        inputs = tf.keras.Input(shape=(None, *ENCODED_STATE_SHAPE))
        # TODO: do we need a stop gradient on 'state'?
        initializer = tf.keras.initializers.Zeros()
        x = l.Dense(64,
                    activation='relu',
                    kernel_initializer=initializer,
                    bias_initializer=initializer)(inputs)
        # TODO: use the layer_norm?
        # x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
        x = l.Dense(64,
                    activation='relu',
                    kernel_initializer=initializer,
                    bias_initializer=initializer)(x)
        # x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
        x = l.Dense(NUM_ACTIONS,
                    activation=None,
                    kernel_initializer=initializer,
                    bias_initializer=initializer)(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def state_net(image, mask):
    logits, endpoints = efnet_builder.build_model(
        image, model_name='efficientnet-b0', mask=mask, training=False)
    prediction = tf.argmax(logits, axis=1)
    pool_outputs = endpoints['global_pool']
    mask_state = MaskEncoding.encode_net(mask)
    pool_state = PoolEncoding.encode_net(pool_outputs)
    state = tf.concat([mask_state, pool_state], axis=1)
    return state, prediction


def accuracy(mask, num_images=2**(10+10+10)):
    return 0


def reward(to_mask) -> float:
    """Calculates the reward received when transitioning to `to_mask`.

    :param to_mask: the mask/state that is transitioned to.
    :return: the reward received when making the transition to `to_mask`.
    """
    acc = accuracy(to_mask)
    if acc >= SUCCESS_ACCURACY:
        return 1.0
    else:
        return 0.0


_mask_values = {}
def value(mask) -> float:
    # Not sure if we need the encoding yet.
    # encoding = MaskEncoding.from_mask(mask)
    mask_as_tuple = tuple(map(tuple, mask))
    return _mask_values[mask_as_tuple]


def random_start_mask(batch_size):
    if batch_size != 1:
        raise Exception('Batching is not yet supported.')
    one_idx = tf.random.uniform(shape=0, minval=0, maxval=NUM_PIXELS,
                                dtype=tf.int8) # use the seed?
    mask_flattened = tf.one_hot(indices=one_idx, depth=NUM_PIXELS, on_value=1,
                                off_value=0, dtype=tf.int8)
    # mask = tf.reshape(mask_flattened, (MASK_WIDTH, MASK_HEIGHT, batch_size))
    mask = tf.reshape(mask_flattened, (-1, NUM_PIXELS))
    return mask, one_idx


def random_start_mask_np(batch_size):
    if batch_size != 1:
        raise Exception('Batching is not yet supported.')
    mask = np.zeros(NUM_PIXELS, dtype=np.int8)
    rand_idx = np.random.randint(0, NUM_PIXELS, dtype=np.int8)
    mask[rand_idx] = 1
    # mask = np.reshape(mask, [MASK_WIDTH, MASK_HEIGHT])
    mask = np.expand_dims(mask, axis=0)
    return mask, rand_idx


def policy_net(action_values, current_mask):
    e = np.random.ranf()
    greedy_rate = 0.95
    if e > greedy_rate:
        action = tf.random.uniform(shape=0, minval=0, maxval=NUM_ACTIONS)
    else:
        allowed_actions = tf.bitwise.invert(current_mask)
        actions_with_disallowed_zeroed = action_values * allowed_actions
        action = tf.argmax(action_values, axis=0)
        # How to make this op a dependency?
        # op = tf.debugging.Assert(tf.assert_non_negative(action_values[action]))
    return action


def policy(action_values, current_mask):
    # Only works with batch_size 1, currently.
    assert action_values.shape[0] == 1
    e = np.random.ranf()
    greedy_rate = 0.95
    def action_available(a):
        return a == STOP_ACTION or not current_mask[0, a]
    action = None
    if e > greedy_rate:
        while not action or action_available(action):
            action = np.random.randint(0, NUM_ACTIONS, dtype=np.int8)
    else:
        # This won't work as we have negative values.
        # allowed_actions = np.ones(NUM_ACTIONS, dtype=np.int8)
        # allowed_actions[0:NUM_PIXELS] = np.invert(current_mask.astype(bool))
        # allowed_action_values = action_values * allowed_actions
        # action = np.argmax(allowed_action_values, axis=1)
        # action = action[0]
        action_values = action_values[0]
        max_val = np.NINF
        action = -1
        for a, val in enumerate(action_values):
            if val > max_val and action_available(a):
                max_val = val
                action = a
        assert action >= 0
        assert action_available(action)
    return action


def sarsa2():
    num_episodes = 10*1000
    batch_size = 1
    start_state_action_values = [0] * 50
    image_fn = imagenet_ds.create_train_input(IMAGE_SHAPE[0],
                                              num_parallel_calls=1,
                                              for_tpu=False).input_fn
    params = {'batch_size': 1}
    dataset = image_fn(params)
    iterator = dataset.make_one_shot_iterator()
    image_input, y = iterator.get_next()
    image_var = tf.Variable(
        tf.zeros((batch_size, *IMAGE_SHAPE), dtype=tf.float32),
        trainable=False)
    y_var = tf.Variable([0]*batch_size, dtype=tf.int32, trainable=False)
    assign_image = image_var.assign(image_input)
    assign_y = y_var.assign(y)
    # State encoding
    mask_placeholder = tf.placeholder(dtype=tf.float32,
                                      shape=(batch_size, NUM_PIXELS),
                                      name='mask')
    state, prediction = state_net(image_var, mask_placeholder)
    state_placeholder = tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size, *ENCODED_STATE_SHAPE))
    # This might be able to be calculated at the same time as state.
    trainable_scope = 'action_value_fn'
    action_vals = action_value_model(scope_name=trainable_scope)(state_placeholder)
    W_variables = tf.trainable_variables(scope=trainable_scope)
    action_placeholder = tf.placeholder(dtype=tf.int32, shape=())
    # Batch broken here.
    chosen_action_val = tf.gather(tf.gather(action_vals, 0), action_placeholder)
    dQdW = tf.gradients(chosen_action_val, W_variables)
    max_norm = 5
    dQdW, _ = tf.clip_by_global_norm(dQdW, clip_norm=max_norm)
    # next_action = policy(action_val, mask_var)
    accuracy = 0
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer()])
        imported_effnet.restore(sess, tf.train.latest_checkpoint(b0_path))
        for e in range(num_episodes):
            # Initialize image, y and calculate prediction.
            # sess.run([y_var.initializer, image_var.initializer])
            sess.run([assign_image, assign_y])
            # initialize graph.
            full_mask = np.ones((1, NUM_PIXELS))
            y_eval, best_prediction = sess.run(
                [y, prediction], feed_dict={mask_placeholder: full_mask})
            mask = np.zeros((1, NUM_PIXELS), dtype=np.int8)
            A = np.random.randint(0, NUM_PIXELS, dtype=np.int8)
            finished = False
            first_loop = True
            S = None
            while not finished:
                if A == STOP_ACTION:
                    finished = True
                else:
                    mask[0, A] = 1
                S_next, prediction_val, W_current = \
                    sess.run([state, prediction, W_variables],
                             feed_dict={mask_placeholder: mask})
                # Observe reward
                if finished:
                    if prediction_val == y_eval:
                        reward = SUCCESS_REWARD
                    elif prediction_val == best_prediction:
                        reward = BEST_EFFORT_REWARD
                    else:
                        reward = 0
                    # Log accuracy
                    sample = int(prediction == y_eval)
                    accuracy = accuracy + 0.01 * (sample - accuracy)
                    action_count = np.sum(mask)
                    print('({}) accuracy: {}, actions: {}'.format(
                        e, accuracy, action_count))
                else:
                    reward = 0
                # Choose next action
                if finished:
                    Q_next = 0
                else:
                    next_Qs = sess.run(action_vals,
                                       feed_dict={state_placeholder: S_next})
                    A_next = policy(next_Qs, mask)
                    # We break the batch dimension here.
                    assert next_Qs.shape[0] == 1
                    Q_next = next_Qs[0][A_next]
                # Calculate td-error.
                if first_loop:
                    first_loop = False
                    Q = start_state_action_values[A]
                    td_error = reward + DISCOUNT_RATE * Q_next - Q
                    # Basic table-based value function.
                    start_state_action_values[A] = \
                        start_state_action_values[A] + ALPHA * td_error
                else:
                    # This can be combined with the previous invocation.
                    assert S is not None
                    Qs, dQdW_val = sess.run([action_vals, dQdW],
                                  feed_dict = {state_placeholder: S,
                                               action_placeholder: A})
                    # We break the batch dimension here.
                    Q = Qs[0][A]
                    # dQdW_val = dQdW_val * max_norm / np.linalg.norm(dQdW_val)
                    td_error = reward + DISCOUNT_RATE * Q_next - Q
                    assign_ops = []
                    for w_var, w_cur_val, w_grad_val in \
                            zip(W_variables, W_current, dQdW_val):
                        w_next_val = w_cur_val + ALPHA * td_error * w_grad_val
                        # print(w_next_val)
                        assign_ops.append(w_var.assign(w_next_val))
                    sess.run(assign_ops)
                S = S_next
                A = A_next


def sarsa():
    num_episodes = 10*1000
    batch_size = 1
    image_size = 224

    # Keep the start state action values separate from the function
    # approximation, as the start state might not be well generalized by the
    # function approximation NN.
    start_state_action_values = [0] * 50

    image_fn = imagenet_ds.create_train_input(image_size,
                                              num_parallel_calls=1,
                                              for_tpu=False).input_fn
    params = {'batch_size': 1}
    dataset = image_fn(params)
    iterator = dataset.make_one_shot_iterator()
    image_input, y = iterator.get_next()
    start_mask, start_idx = random_start_mask(batch_size)
    ones_mask = np.ones((MASK_WIDTH, MASK_HEIGHT), dtype=np.int8)
    mask_var = tf.Variable(tf.ones([MASK_WIDTH, MASK_HEIGHT]))
    action_var = tf.Variable(0)
    state, masked_predict = state_net(image_input, mask_var)
    action_val = action_value_model(state)()
    next_action = policy_net(action_val, mask_var)
    # We are duplicating the efficientnet network here!
    # m = tf.stack([ones_mask, start_mask], axis=2)
    _, best_predict = state_net(image_input, ones_mask)

    for episode in range(num_episodes):
        # increment the image (manually)!
        with tf.Session() as sess:
            # FIXME, not how it works.
            mask_var.assign(start_mask)
        first_iter = True
        is_terminal = False
        current_action = start_idx
        current_action_val = start_state_action_values[current_action]
        while not is_terminal:
            next_action_evaluated, next_state_evaluated, \
            next_action_val_evaluated, mask_evaluated, best_predict_evaluated, \
            masked_predict_evaluated = sess.run([next_action,
                 state, action_val, start_mask, best_predict, masked_predict])
            assert 0 <= next_action_evaluated < NUM_ACTIONS
            is_terminal = next_action_evaluated == STOP_ACTION
            # The calculation of the reward could be put in the graph.
            gradient = calc_gradients()
            if is_terminal:
                next_state_val = 0 # end state.
                is_correct = masked_predict == y
                is_best_effort = masked_predict == best_predict
                if is_correct:
                    reward = SUCCESS_REWARD
                elif is_best_effort:
                    reward = BEST_EFFORT_REWARD
                else:
                    reward = 0
                td_error = reward - action_val
            else:
                reward = 0
                target = next_action_val_evaluated[next_action_evaluated]
                if first_iter:
                    td_error = reward + DISCOUNT_RATE * target - current_action_val
                    start_state_action_values[current_action] = \
                        start_state_action_values[current_action] + ALPHA * td_error
                else:
                    td_error = reward + DISCOUNT_RATE * target - current_action_val
                    # update_W
                next_mask_entry = action_coords(next_action_evaluated)
                # current_action_val =
                assert mask_evaluated[next_mask_entry] == 0, \
                    'The action must choose an unevaluated pixel.'


if __name__ == '__main__':
    sarsa2()





