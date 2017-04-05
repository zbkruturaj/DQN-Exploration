import numpy as np
import tensorflow as tf
import gym

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

env = gym.make('MountainCar-v0')
env.reset()

alpha = 3e-1 # learning rate
gamma = 0.99 # discount factor
batch_size = 5

d_X = env.observation_space.shape[0]
d_H = 8    # number of hidden units
d_A = env.action_space.n

tf.reset_default_graph()

obs = tf.placeholder(tf.float32, [None,d_X], name="X")
W1 = tf.get_variable("Wxh", shape=[d_X, d_H], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
hid = tf.nn.relu(tf.matmul(obs, W1))
W2 = tf.get_variable("Wha", shape=[d_H, d_A], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
# print tf.shape(W1)
# print tf.shape(W2)
prob = tf.reshape(tf.nn.softmax(tf.matmul(hid, W2)),[-1])
chosen_action = tf.argmax(prob,0)
# print chosen_action
# print prob
chosen_prob = tf.slice(prob, [tf.cast(chosen_action,tf.int32)], [1])

trainable_vars = tf.trainable_variables()

reward_ph = tf.placeholder(tf.float32, [None,1], name="reward_ph")

loss = -tf.reduce_mean(tf.log(chosen_prob)*reward_ph)
grads = tf.gradients(loss, trainable_vars)

optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
W1_gradient = tf.placeholder(tf.float32, shape=[d_X, d_H], name="W1_gradient")
W2_gradient = tf.placeholder(tf.float32,shape=[d_H, d_A],  name="W2_gradient")
gradients = [W1_gradient, W2_gradient]
updateGrads = optimizer.apply_gradients(zip(gradients,trainable_vars))


xs,drs = [],[]
#xs are the states
#drs are the discounted rewards
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.global_variables_initializer()
print "starting"
with tf.Session() as sess:
        rendering = False
        sess.run(init)
        observation = env.reset()
        gradBuffer = sess.run(trainable_vars)
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad*0

        while episode_number <= total_episodes:
            if reward_sum/batch_size > 100 or rendering == True :
                env.render()
                rendering = False
            X = np.reshape(observation,[1,d_X])
            prob, action = sess.run([chosen_prob,chosen_action],feed_dict={obs:X})
            # print prob
            # print action
            xs.append(X)
            observation, reward, done, info = env.step(action)
            # print reward
            reward_sum += reward
            drs.append(reward)
            if done:
                # print str(episode_number) + " : " + str(reward_sum)
                episode_number += 1
                epx = np.vstack(xs)
                epr = np.vstack(drs)
                xs,drs = [],[]
                discounted_r = discount_rewards(epr)
                discounted_r -= np.mean(discounted_r)
                discounted_r /= np.std(discounted_r)
                tGrad = sess.run(grads, feed_dict={obs: epx, reward_ph: discounted_r})
                for ix,grad in enumerate(tGrad):
                    gradBuffer[ix] += grad
                if episode_number%batch_size == 0:
                    sess.run(updateGrads,feed_dict={W1_gradient: gradBuffer[0],W2_gradient:gradBuffer[1]})
                    for ix, grad in enumerate(tGrad):
                        gradBuffer[ix] = grad*0

                    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                    print '%d : Average reward for episode %f.  Total average reward %f.' % (episode_number,reward_sum/batch_size, running_reward/batch_size)
                    if reward_sum/batch_size > 200:
                        print "Task solved in",episode_number,'episodes!'
                        break
                    reward_sum = 0
                observation = env.reset()
print episode_number,'Episodes completed.'
