import tensorflow as tf
import gym
import micronet.experiments.sequential_pool.openai_env as seq_pool_env
import micronet.experiments.sequential_pool.sequential_pool as seq_pool
import micronet.experiments.sequential_pool.custom_train as custom_train
import baselines.common.tf_util as U
import baselines.common.input
import itertools
import numpy as np

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule


def multi_binary_observation_placeholder(ob_space, batch_size=None, name='Ob'):
    assert isinstance(ob_space, gym.spaces.MultiBinary)

    return tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=bool,
                          name=name)

def create_obs_and_mask_placeholder(ob_space, batch_size=None, name='Ob'):
    obs = ob_space[0]
    input = ObservationInput(obs)
    mask_obs = ob_space[1]
    mask_placeholder = tf.placeholder(shape=(batch_size,)+mask_obs.shape,
                                      dtype=bool, name=name + '_mask')
    return input, mask_placeholder


def action_value_net():#encoded_state):
    # TODO: do we need a stop gradient on 'state'?
    inputs = tf.keras.Input(shape=(None, *seq_pool.ENCODED_STATE_SHAPE))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    # TODO: use the layer_norm?
    # x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    # x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
    x = tf.keras.layers.Dense(seq_pool.NUM_ACTIONS, activation=None)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def model(observation, mask, num_actions, scope, reuse=False):
    # TODO fix masking (49 vs 50, action vs pool mask).
    pool_outputs = observation
    with tf.variable_scope(scope):
        mask_state = seq_pool.MaskEncoding.encode_net(mask[:, :-1])
        pool_state = seq_pool.PoolEncoding.encode_net(pool_outputs)
        state = tf.concat([mask_state, pool_state], axis=1)
        av = action_value_net()(state)
        return av

# Original: https://github.com/openai/baselines/blob/master/baselines/deepq/experiments/custom_cartpole.py
# Updated to use custom build_train().
if __name__ == '__main__':
    with U.make_session(num_cpu=2) as sess:
        # Create the environment
        env = seq_pool_env.PoolEnv(sess)
        def obs_mask_pair(name):
            return create_obs_and_mask_placeholder(env.observation_space,
                                                   name=name)
        # Create all the functions necessary to train the model
        act, train, update_target, debug = custom_train.build_train(
            make_obs_ph=obs_mask_pair,
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        env.load_weights()
        update_target()

        episode_rewards = [0.0]
        obs = env.reset()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                # Show off the result
                env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward",
                                      round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
