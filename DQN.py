import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters for DQN
GAMMA = 0.99  # discount factor for target Q
INITIAL_EPSILON = 1  # starting value of epsilon for epsilon-greedy policy
FINAL_EPSILON = 0.05  # final value of epsilon
DECAY_STEP = 500000  # number of step between INITIAL_EPSILON and FINAL_EPSILON
REPLAY_SIZE = 100000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch
TORQUE = 10000  # target network was updated every TORQUE steps
LEARNING_RATE = 0.00025
NUMBER_OF_NEURONS = 50
DELTA = 0.01  # a small postitive constant that prevent priority become 0
ALPHA = 0.6  # control the difference between high and low error


class DQN():
    # DQN Agent
    def __init__(self, env):
        # init experience replay
        if Prioritized:
            self.replay_memory = Prioritized_memory()
        else:
            self.replay_memory = deque()  # list-like container
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON  # epsilon will keep decreasing
        # self.epsilons = np.linspace(INITIAL_EPSILON, FINAL_EPSILON, DECAY_STEP)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_Q_network()
        self.create_target_network()
        self.create_training_method()
        self.saver = tf.train.Saver()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

    def create_Q_network(self):
        # network weights
        with tf.variable_scope("online_Q"):
            W1 = self.weight_variable([self.state_dim, NUMBER_OF_NEURONS])
            b1 = self.bias_variable([NUMBER_OF_NEURONS])
            W2 = self.weight_variable([NUMBER_OF_NEURONS, NUMBER_OF_NEURONS])
            b2 = self.bias_variable([NUMBER_OF_NEURONS])
            W3 = self.weight_variable([NUMBER_OF_NEURONS, self.action_dim])
            b3 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        # hidden layers
        h_layer1 = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        h_layer2 = tf.nn.relu(tf.matmul(h_layer1, W2) + b2)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer2, W3) + b3

    def create_target_network(self):
        # network weights
        with tf.variable_scope("target_Q"):
            W1 = self.weight_variable([self.state_dim, NUMBER_OF_NEURONS])
            b1 = self.bias_variable([NUMBER_OF_NEURONS])
            W2 = self.weight_variable([NUMBER_OF_NEURONS, NUMBER_OF_NEURONS])
            b2 = self.bias_variable([NUMBER_OF_NEURONS])
            W3 = self.weight_variable([NUMBER_OF_NEURONS, self.action_dim])
            b3 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input_2 = tf.placeholder("float", [None, self.state_dim])
        # hidden layers
        h_layer1 = tf.nn.relu(tf.matmul(self.state_input_2, W1) + b1)
        h_layer2 = tf.nn.relu(tf.matmul(h_layer1, W2) + b2)
        # Q Value layer
        self.target_Q_value = tf.matmul(h_layer2, W3) + b3

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])  # one hot presentation
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(
            tf.mul(self.Q_value, self.action_input), reduction_indices=1)  # reduce_sum to turn into scalar
        err = self.y_input - Q_action
        if ErrorClipping:
            #  sqrt(1+a^2)-1
            self.cost = tf.reduce_mean(tf.sqrt(1 + tf.square(err)) - 1)
        else:
            self.cost = tf.reduce_mean(tf.square(err))
        # self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=0.95).minimize(self.cost)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done, train=True):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        if Prioritized:
            online_Q_value_next_state = self.Q_value.eval(
                feed_dict={self.state_input: [next_state]})
            target_Q_value_next_state = self.target_Q_value.eval(
                feed_dict={self.state_input_2: [next_state]})
            online_Q_value_current_state = self.Q_value.eval(
                feed_dict={self.state_input: [state]})
            estimate_value = online_Q_value_current_state[0][action]
            if DoubleDQN:
                target_value = reward + GAMMA * target_Q_value_next_state[0][np.argmax(online_Q_value_next_state[0])]
            else:
                target_value = reward + GAMMA * np.max(target_Q_value_next_state[0])
            error = abs(target_value - estimate_value)
            self.replay_memory.add(error, (state, one_hot_action, reward, next_state, done))
        else:
            self.replay_memory.append(
                (state, one_hot_action, reward, next_state, done))
            if len(self.replay_memory) > REPLAY_SIZE:
                self.replay_memory.popleft()
        if train:
            # self.epsilon = self.epsilons[min(self.time_step, DECAY_STEP - 1)]
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        if Prioritized:
            minibatch = self.replay_memory.sample()
        else:
            minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        if DoubleDQN:
            Q_value_batch = self.Q_value.eval(
                feed_dict={self.state_input: next_state_batch})  # required for Double DQN
        target_Q_value_batch = self.target_Q_value.eval(
            feed_dict={self.state_input_2: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                if DoubleDQN:
                    y_batch.append(reward_batch[i] +
                                   GAMMA * target_Q_value_batch[i][np.argmax(Q_value_batch[i])])  # Double DQN
                else:
                    y_batch.append(reward_batch[i] +
                                   GAMMA * np.max(target_Q_value_batch[i]))  # Nature DQN
        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def egreedy_action(self, state):
        if self.time_step % TORQUE == 0 and self.time_step != 0:
            self.update_target_network()
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            Q_value = self.Q_value.eval(feed_dict={
                self.state_input: [state]
            })[0]
            return np.argmax(Q_value)

    def update_target_network(self):
        online_params = [t for t in tf.trainable_variables() if t.name.startswith("online_Q")]
        online_params = sorted(online_params, key=lambda v: v.name)
        target_params = [t for t in tf.trainable_variables() if t.name.startswith("target_Q")]
        target_params = sorted(target_params, key=lambda v: v.name)
        update_operations = []
        for online_value, target_value in zip(online_params, target_params):
            operation = target_value.assign(online_value)
            update_operations.append(operation)
        self.session.run(update_operations)
        print("target network updated!")

    def best_action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)


class BinarySumTree():
    # data structure for replay memory
    write = 0

    def __init__(self):
        self.capacity = REPLAY_SIZE
        self.tree = np.zeros(2 * REPLAY_SIZE - 1)
        self.data = np.zeros(REPLAY_SIZE, dtype=object)

    def _propagate(self, index, change):
        parent = (index - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, index, s):
        left_child = 2 * index + 1
        right_child = left_child + 1

        if left_child >= len(self.tree):
            return index

        if s <= self.tree[left_child]:
            return self._retrieve(left_child, s)
        else:
            return self._retrieve(right_child, s - self.tree[left_child])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        index = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(index, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, index, priority):
        change = priority - self.tree[index]

        self.tree[index] = priority
        self._propagate(index, change)

    def get(self, s):
        index = self._retrieve(0, s)
        dataIndex = index - self.capacity + 1

        return (index, self.tree[index], self.data[dataIndex])


class Prioritized_memory():

    def __init__(self):
        self.tree = BinarySumTree()

    def _getPriority(self, error):
        #  prevent priority getting zero
        return (error + DELTA) ** ALPHA

    def add(self, error, sample):
        priority = self._getPriority(error)
        self.tree.add(priority, sample)

    def sample(self):
        batch = []
        segment = self.tree.total() / BATCH_SIZE

        for i in range(BATCH_SIZE):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (index, priority, data) = self.tree.get(s)
            # batch.append((index, data))
            batch.append(data)

        return batch

    def update(self, index, error):
        priority = self._getPriority(error)
        self.tree.update(index, priority)


# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'Acrobot-v1'
POSTFIX = "_DD"
MONITOR_RECORD_SEPERATION = 1000
EPISODE = 17001  # Episode for training
FIXEPISODE = 10000  # episode to start fixing epsilon at FINAL_EPSILON
TEST = 14000  # episode to start testing
DoubleDQN = True  # True for DDQN, False for target DQN
Prioritized = False  # True for using prioritized experience replay, False otherwise
ErrorClipping = False  # True for using error clipping (Huber loss)


def main():
    if DoubleDQN:
        print('Double DQN is turned on')
    if Prioritized:
        print('Prioritized experience replay is turned on')
    if ErrorClipping:
        print('ErrorClipping is turned on')
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    print('State dimension: ', env.observation_space)
    print('Action dimensin: ', env.action_space)
    print('Sample State: ', env.observation_space.sample())
    print('Sample Action: ', env.action_space.sample())
    agent = DQN(env)
    fillMemoryAction = 0
    while fillMemoryAction < REPLAY_SIZE:
        #  fill up the replay memory using random action before training
        state = env.reset()
        while True:
            action = agent.egreedy_action(state)
            fillMemoryAction += 1
            next_state, reward, done, _ = env.step(action)
            agent.perceive(state, action, reward, next_state, done, train=False)
            if fillMemoryAction % 1000 == 1:
                print('Filling up replay memory:', fillMemoryAction, '/', REPLAY_SIZE)
            state = next_state
            if done:
                break
    # env.monitor.start('./record/' + ENV_NAME + POSTFIX, force=True,
    #                   video_callable=lambda x: x % MONITOR_RECORD_SEPERATION == 0)
    print('---------START TRAINING----------')
    for episode in range(EPISODE):
        print('episode: ', episode)
        # initialize task
        total_reward = 0
        state = env.reset()
        # Train
        while True:
            action = agent.egreedy_action(state)  # e-greedy action for train
            if episode >= TEST:
                action = agent.best_action(state)
            print(action)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.perceive(state, action, reward, next_state, done, train=True)
            state = next_state
            if done:
                print('score: ', total_reward)
                if episode <= FIXEPISODE:
                    agent.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / FIXEPISODE
                    print('epsilon: ', agent.epsilon)
                break
    env.monitor.close()

    # save the trained model
    save_path = agent.saver.save(agent.session, 'trained_model/' + ENV_NAME + POSTFIX)
    print('Model saved in file: %s' % save_path)


if __name__ == '__main__':
    main()
