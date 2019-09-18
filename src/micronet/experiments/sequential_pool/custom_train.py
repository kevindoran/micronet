# Original: https://github.com/openai/baselines/blob/master/baselines/deepq/build_graph.py
# Edited in order to provide tuple observation space.

import tensorflow as tf
from baselines.deepq.build_graph import build_act_with_param_noise
import baselines.common.tf_util as U


def build_act(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None):
    """Creates the act function:
    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a tuple of placeholders
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        obs, mask = make_obs_ph("observation")
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
        q_values = q_func(obs.get(), mask, num_actions, scope="q_func")
        # Mask actions.
        # Mask should already be boolean, but if not, the following would work:
        # zero = tf.constant(0, dtype=tf.float32)
        # mask_as_bool = tf.not_equal(mask, zero)
        min_tensor = tf.constant(-100.0, tf.float32,
                                 shape=(q_values.shape[-1],))
        masked_action_vals = tf.compat.v1.where_v2(mask, min_tensor, q_values)
        deterministic_actions = tf.argmax(masked_action_vals, axis=1)
        batch_size = tf.shape(obs.get())[0]
        # 2D rand array. Random action is the argmax of the random array after we
        # mask out the actions that are not allowed.
        random_actions = tf.random_uniform((batch_size, num_actions), minval=0, maxval=num_actions, dtype=tf.int32)
        below_zero = tf.fill([batch_size, num_actions], value=-1)
        random_actions = tf.argmax(tf.compat.v1.where_v2(mask, below_zero, random_actions), axis=1)
        choose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(choose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        _act = U.function(inputs=[obs, mask, stochastic_ph, update_eps_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        def act(ob, mask, stochastic=True, update_eps=-1):
            return _act(ob, mask, stochastic, update_eps)
        return act


def build_train(make_obs_ph, q_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0,
    double_q=True, scope="deepq", reuse=None, param_noise=False, param_noise_filter_func=None):
    """Creates the train function:
    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a tuple of placeholders of input (using name as a name prefix)
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    grad_norm_clipping: float or None
        clip gradient norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    param_noise_filter_func: tf.Variable -> bool
        function that decides whether or not a variable should be perturbed. Only applicable
        if param_noise is True. If set to None, default_param_noise_filter is used by default.
    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """
    if param_noise:
        act_f = build_act_with_param_noise(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse,
            param_noise_filter_func=param_noise_filter_func)
    else:
        act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input, mask_t = make_obs_ph("obs_t")
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input, mask_tp1 = make_obs_ph("obs_tp1")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q network evaluation
        q_t = q_func(obs_t_input.get(), mask_t, num_actions, scope="q_func", reuse=True)  # reuse parameters from act
        min_tensor = tf.constant(-100.0, tf.float32,
                                 shape=(q_t.shape[-1],))
        qt = tf.compat.v1.where_v2(mask_t, min_tensor, q_t)
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")

        # target q network evalution
        q_tp1 = q_func(obs_tp1_input.get(), mask_tp1, num_actions, scope="target_q_func")
        min_tensor = tf.constant(-100.0, tf.float32,
                                 shape=(q_tp1.shape[-1],))
        q_tp1 = tf.compat.v1.where_v2(mask_tp1, min_tensor, q_tp1)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/target_q_func")

        # q scores for actions which we know were selected in the given state.
        #q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)
        temp = tf.math.multiply(q_t, tf.one_hot(act_t_ph, num_actions), name='reduce_sum_input')
        q_t_selected = tf.reduce_sum(temp, 1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_using_online_net = q_func(obs_tp1_input.get(), mask_tp1, num_actions, scope="q_func", reuse=True)
            min_tensor = tf.constant(-100.0, tf.float32,
                                     shape=(q_tp1_using_online_net.shape[-1],))
            q_tp1_using_online_net = tf.compat.v1.where_v2(mask_tp1, min_tensor,
                                                       q_tp1_using_online_net)
            q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        with tf.variable_scope('huber_loss'):
            errors = U.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors, name='weighted_error')

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
            optimize_expr = optimizer.apply_gradients(gradients)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                mask_t,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                mask_tp1,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=td_error,
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

        return act_f, train, update_target, {'q_values': q_values}