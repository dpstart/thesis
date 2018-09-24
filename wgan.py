batch_size = 32
X_dim = 784
z_dim = 10
h_dim = 128


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

#Placeholder for the image input.
X = tf.placeholder(tf.float32, shape=[None, X_dim])

#Disciminator parameters
D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

#Input noise
z = tf.placeholder(tf.float32, shape=[None, z_dim])

# Generator parameters 
G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

# We use this to sample the Gaussian input variables for the generator.
def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# The generator networks is the same as in standard GANs.
def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob

# Disciminator network. 
# We have removed the sigmoid output activation since we don't want to output a probability anymore.
def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out


G_sample = generator(z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

#Loss functions
D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)


# In the original paper, RMSProp is suggested as optimization algorithm.
D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(-D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(G_loss, var_list=theta_G))

# We will use this to clip the weights at every iteration of the training loop.
clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for it in range(1000000):

    # 5 iterations of D optimization, 1 iteration of G optimization.
    for _ in range(5):
        X_batch, _ = mnist.train.next_batch(batch_size)

        # Optimize D
        _, D_loss_curr, _ = sess.run(
            [D_solver, D_loss, clip_D],
            feed_dict={X: X_batch, z: sample_z(batch_size, z_dim)}
        )
    # Optimize G  
    _, G_loss_curr = sess.run(
      [G_solver, G_loss],
      feed_dict={z: sample_z(batch_size, z_dim)}
    )
