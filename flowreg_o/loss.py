import tensorflow as tf

alphaval = 0.2

def photometric_loss(true, pred):
    size = tf.cast(tf.size(pred), tf.float32)
    alpha = alphaval
    beta = 1.0
    diff = tf.abs(true-pred)
    dist = tf.reduce_sum(diff, axis=3, keep_dims=True)
    loss = charbonnier(dist, alpha, beta, 0.001)
    photo_loss = tf.reduce_sum(loss)/size

    corr_loss = correlation_loss(true, pred)
    lam_bda = 0.5
    return (lam_bda*photo_loss) + corr_loss
    # return lam_bda*photo_loss


def smoothness_loss(zflow, flow):
    size = tf.cast(tf.size(flow), tf.float32)
    x, y = tf.unstack(flow, axis=3)
    x = tf.expand_dims(x, axis=3)
    y = tf.expand_dims(y, axis=3)

    u = tf.constant([[0., 0., 0.],
                     [0., 1., -1.],
                     [0., 0., 0.]])
    u = tf.expand_dims(u, axis=2)
    u = tf.expand_dims(u, axis=3)
    v = tf.constant([[0., 0., 0.],
                     [0., 1., 0.],
                     [0., -1., 0.]])
    v = tf.expand_dims(v, axis=2)
    v = tf.expand_dims(v, axis=3)

    u_diff = tf.nn.conv2d(x, u, strides=[1, 1, 1, 1], padding='SAME')
    v_diff = tf.nn.conv2d(y, v, strides=[1, 1, 1, 1], padding='SAME')
    all_diff = tf.concat([u_diff, v_diff], axis=3)
    dists = tf.reduce_sum(tf.abs(all_diff), axis=3, keep_dims=True)
    alpha = alphaval
    beta = 1.0
    lam_bda = 1.
    loss = charbonnier(dists, alpha, beta, 0.001)

    return lam_bda*(tf.reduce_sum(loss)/size)

def charbonnier(x, alpha, beta, epsilon):
    x = x*beta
    out = tf.pow((tf.square(x)+tf.square(epsilon)), alpha)
    return out

def correlation_loss(true, pred):
    true = tf.cast(true, tf.float32)
    pred = tf.cast(pred, tf.float32)

    mux = tf.reduce_mean(true)
    muy = tf.reduce_mean(pred)
    n = tf.cast(tf.size(true), tf.float32)

    varx = tf.reduce_sum(tf.square(true - mux))/n
    vary = tf.reduce_sum(tf.square(pred - muy))/n

    corr = 1/n * tf.reduce_sum((true - mux) * (pred - muy)) / tf.math.sqrt(varx * vary)

    return 1-corr
