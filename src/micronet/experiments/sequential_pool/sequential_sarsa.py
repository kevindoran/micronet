import numpy as np
import tensorflow as tf
import micronet.dataset.imagenet as imagenet_ds
import micronet.experiments.sequential_pool.efficientnet_utils as efnet_utils
import micronet.experiments.sequential_pool.sequential_pool as seq_pool
import micronet.experiments.sequential_pool.efficientnet_builder as efnet_builder

SUCCESS_ACCURACY = 0.76
SUCCESS_REWARD = 100
BEST_EFFORT_REWARD = 50
DISCOUNT_RATE = 0.96
ALPHA = 0.2


def action_coords(idx_):
    return (idx_ / seq_pool.MASK_WIDTH, idx_ % seq_pool.MASK_WIDTH)


# def action_value_model(state, scope_name='action_value'):
def action_value_model(scope_name='action_value'):
    with tf.variable_scope(scope_name):
        l = tf.keras.layers
        inputs = tf.keras.Input(shape=(None, *seq_pool.ENCODED_STATE_SHAPE))
        # TODO: do we need a stop gradient on 'state'?
        x = l.Dense(64,
                    activation='relu')(inputs)
        # TODO: use the layer_norm?
        # x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
        x = l.Dense(64,
                    activation='relu')(x)
        # x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
        x = l.Dense(seq_pool.NUM_ACTIONS,
                    activation=None)(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


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
    one_idx = tf.random.uniform(shape=0, minval=0, maxval=seq_pool.NUM_PIXELS,
                                dtype=tf.int8) # use the seed?
    mask_flattened = tf.one_hot(indices=one_idx, depth=seq_pool.NUM_PIXELS,
                                on_value=1, off_value=0, dtype=tf.int8)
    # mask = tf.reshape(mask_flattened, (MASK_WIDTH, MASK_HEIGHT, batch_size))
    mask = tf.reshape(mask_flattened, (-1, seq_pool.NUM_PIXELS))
    return mask, one_idx


def random_start_mask_np(batch_size):
    if batch_size != 1:
        raise Exception('Batching is not yet supported.')
    mask = np.zeros(seq_pool.NUM_PIXELS, dtype=np.int8)
    rand_idx = np.random.randint(0, seq_pool.NUM_PIXELS, dtype=np.int8)
    mask[rand_idx] = 1
    # mask = np.reshape(mask, [MASK_WIDTH, MASK_HEIGHT])
    mask = np.expand_dims(mask, axis=0)
    return mask, rand_idx


def policy_net(action_values, current_mask):
    e = np.random.ranf()
    greedy_rate = 0.95
    if e > greedy_rate:
        action = tf.random.uniform(shape=0, minval=0,
                                   maxval=seq_pool.NUM_ACTIONS)
    else:
        allowed_actions = tf.bitwise.invert(current_mask)
        actions_with_disallowed_zeroed = action_values * allowed_actions
        action = tf.argmax(action_values, axis=0)
        # How to make this op a dependency?
        # op = tf.debugging.Assert(tf.assert_non_negative(action_values[action]))
    return action


def state_net(image, mask):
    def normalize_features(features, mean_rgb, stddev_rgb):
        """Normalize the image given the means and stddevs."""
        # TODO: support GPU by using shape=[3, 1, 1]
        features -= tf.constant(mean_rgb, shape=[1, 1, 3],
                                dtype=features.dtype)
        features /= tf.constant(stddev_rgb, shape=[1, 1, 3],
                                dtype=features.dtype)
        return features
    normalized_image = normalize_features(image,
                                          efnet_builder.MEAN_RGB,
                                          efnet_builder.STDDEV_RGB)
    logits, endpoints = efnet_builder.build_model(
        normalized_image, model_name='efficientnet-b0', mask=mask,
        training=False)
    pool_inputs = endpoints['masked_features']
    pool_outputs = endpoints['global_pool']
    prediction = tf.argmax(logits, axis=1)
    mask_state = seq_pool.MaskEncoding.encode_net(mask)
    pool_state = seq_pool.PoolEncoding.encode_net(pool_outputs)
    state = tf.concat([mask_state, pool_state], axis=1)
    return state, prediction, pool_inputs


def start_policy(start_action_values):
    max = np.NINF
    max_actions = []
    for a, val in enumerate(start_action_values):
        if val > max:
            max = val
            max_actions = [a]
        elif val == max:
            max_actions.append(a)
    assert len(max_actions)
    rand_action_idx = np.random.randint(0, len(max_actions), dtype=np.int32)
    return max_actions[rand_action_idx]


def policy(action_values, current_mask):
    # Only works with batch_size 1, currently.
    assert action_values.shape[0] == 1
    e = np.random.ranf()
    greedy_rate = 0.95
    def action_available(a):
        return a == seq_pool.STOP_ACTION or not current_mask[0, a]
    action = None
    if e > greedy_rate:
        while not action or action_available(action):
            action = np.random.randint(0, seq_pool.NUM_ACTIONS, dtype=np.int8)
    else:
        # This won't work as we have negative values.
        # allowed_actions = np.ones(NUM_ACTIONS, dtype=np.int8)
        # allowed_actions[0:NUM_PIXELS] = np.invert(current_mask.astype(bool))
        # allowed_action_values = action_values * allowed_actions
        # action = np.argmax(allowed_action_values, axis=1)
        # action = action[0]
        action_values = action_values[0]
        max_val = np.NINF
        actions = []
        for a, val in enumerate(action_values):
            if val > max_val and action_available(a):
                max_val = val
                actions = [a]
            elif val == max_val and action_available(a):
                actions.append(a)
        assert len(actions)
        rand_action_idx = np.random.randint(0, len(actions), dtype=np.int32) # What dtype?
        action = actions[rand_action_idx]
        assert action_available(action)
    return action


saved_start_action_vals = [19.128806994560478, 17.903387234493863, 14.35500794529724, 17.629336648327637, 19.40618354039853, 15.368205760829833, 18.69720492677352, 14.712588586951826, 9.347403166732787, 15.435666945452201, 18.314733724258947, 18.88106108516228, 16.273715040664065, 20.080289216025882, 19.411999505789453, 18.451155059475795, 19.573354523760237, 16.08646622262612, 19.427368030845116, 20.568658012122118, 17.919577490273436, 19.076823942232206, 18.82401307941811, 18.30980455808352, 19.47249394411377, 18.532508752228768, 18.392759097324888, 17.795497133342387, 15.397738006417617, 19.707093322953867, 18.352668620635818, 19.43290009745056, 19.12789500386218, 17.778815246750522, 19.3061370267395, 17.43125315181191, 19.380006889667893, 6.808670120239258, 18.857488811035008, 19.45954630933901, 19.721426123732183, 18.934857486186857, 18.608201661688728, 17.24240814690105, 18.36357099050778, 19.435089573227465, 16.476550419098633, 19.02577347091082, 19.77296218548403, 0]

def sarsa2():
    num_episodes = 10*1000
    batch_size = 1
    # start_state_action_values = [0] * 49 (not 50, as STOP isn't a valid first action).
    start_state_action_values = saved_start_action_vals
    iterator = seq_pool.image_iterator()
    image_input, y = iterator.get_next()
    image_var = tf.Variable(
        tf.zeros((batch_size, *seq_pool.IMAGE_SHAPE), dtype=tf.float32),
        trainable=False)
    y_var = tf.Variable([0]*batch_size, dtype=tf.int32, trainable=False)
    # assign_image = image_var.assign(image_input)
    # assign_y = y_var.assign(y)
    img_placeholder = tf.placeholder(dtype=tf.float32,
                                      shape=(batch_size, *seq_pool.IMAGE_SHAPE),
                                      name='img_placeholder')
    # State encoding
    mask_placeholder = tf.placeholder(dtype=tf.float32,
                                      shape=(batch_size, seq_pool.NUM_PIXELS),
                                      name='mask')
    state, prediction, _ = state_net(img_placeholder, mask_placeholder)
    state_placeholder = tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size,
                                              *seq_pool.ENCODED_STATE_SHAPE))
    # This might be able to be calculated at the same time as state.
    trainable_scope = 'action_value_fn'
    action_vals = action_value_model(scope_name=trainable_scope)(state_placeholder)
    W_variables = tf.trainable_variables(scope=trainable_scope)
    action_placeholder = tf.placeholder(dtype=tf.int32, shape=())
    # Batch broken here.
    chosen_action_val = tf.gather(tf.gather(action_vals, 0), action_placeholder)
    dQdW = tf.gradients(chosen_action_val, W_variables)
    max_norm = 20
    dQdW, _ = tf.clip_by_global_norm(dQdW, clip_norm=max_norm)
    # next_action = policy(action_val, mask_var)
    eval_ckpt_driver = efnet_utils.EvalCkptDriver('efficientnet-b0')
    accuracy = 0
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer()])
        b0_path = '/home/k/Sync/micronet/resources/efficientnet-b0/'
        eval_ckpt_driver.restore_model(sess, b0_path, enable_ema=True,
                                       export_ckpt=None)
        # Selectively initialize only the variables that yet to be initialized.
        # yet_uninitialized = tuple(
        #     tf.get_variable(name)
        #     for name in sess.run(tf.report_uninitialized_variables())
        # )
        # sess.run(tf.initialize_variables(yet_uninitialized))
        for e in range(num_episodes):
            # Initialize image, y and calculate prediction.
            # sess.run([y_var.initializer, image_var.initializer])
            # sess.run([assign_image, assign_y])
            full_mask = np.ones((1, seq_pool.NUM_PIXELS))
            y_eval, image_eval = sess.run([y, image_input])
            best_prediction = sess.run(prediction,
                                       feed_dict={mask_placeholder: full_mask,
                                       img_placeholder: image_eval})
            mask = np.zeros((1, seq_pool.NUM_PIXELS), dtype=np.int8)
            # A = np.random.randint(0, NUM_PIXELS, dtype=np.int8)
            A = start_policy(start_state_action_values)
            finished = False
            first_loop = True
            S = None
            while not finished:
                if A == seq_pool.STOP_ACTION:
                    finished = True
                else:
                    mask[0, A] = 1
                S_next, prediction_val, W_current = \
                    sess.run([state, prediction, W_variables],
                             feed_dict={mask_placeholder: mask,
                                        img_placeholder: image_eval})
                # Observe reward
                if finished:
                    if prediction_val == y_eval:
                        reward = SUCCESS_REWARD
                    elif prediction_val == best_prediction:
                        reward = BEST_EFFORT_REWARD
                    else:
                        reward = 0
                    # Log accuracy
                    sample = int(prediction_val == y_eval)
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
                        # TODO: include reguralization in the graph.
                        weight_decay = 1e-5
                        weight_reg = weight_decay*(2*w_cur_val)
                        w_next_val = w_cur_val + ALPHA * td_error * w_grad_val - weight_reg
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
    ones_mask = np.ones((seq_pool.MASK_WIDTH, seq_pool.MASK_HEIGHT),
                        dtype=np.int8)
    mask_var = tf.Variable(tf.ones([seq_pool.MASK_WIDTH, seq_pool.MASK_HEIGHT]))
    action_var = tf.Variable(0)
    state, masked_predict, _ = state_net(image_input, mask_var)
    action_val = action_value_model(state)()
    next_action = policy_net(action_val, mask_var)
    # We are duplicating the efficientnet network here!
    # m = tf.stack([ones_mask, start_mask], axis=2)
    _, best_predict, _ = state_net(image_input, ones_mask)

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
            assert 0 <= next_action_evaluated < seq_pool.NUM_ACTIONS
            is_terminal = next_action_evaluated == seq_pool.STOP_ACTION
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
    tf.logging.set_verbosity(tf.logging.INFO)
    sarsa2()





