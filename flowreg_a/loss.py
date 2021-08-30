import tensorflow as tf

def correlation(true, pred):
    true = tf.cast(true, tf.float32)
    pred = tf.cast(pred, tf.float32)

    mux = tf.reduce_mean(true)
    muy = tf.reduce_mean(pred)
    n = tf.cast(tf.size(true), tf.float32)

    varx = tf.reduce_sum(tf.square(true - mux))/n
    vary = tf.reduce_sum(tf.square(pred - muy))/n

    corr = 1/n * tf.reduce_sum((true - mux) * (pred - muy)) / tf.math.sqrt(varx * vary)

    return 1-corr