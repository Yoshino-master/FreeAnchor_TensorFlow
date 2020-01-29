import tensorflow as tf
from tensorflow.core.framework import summary_pb2

class AvarageMeter():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0.0
        self.n = 0.0
        self.sum = 0.0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.n += n
        
    def get_avg(self):
        return self.sum / self.n if self.n != 0 else 0.0

def config_learning_rate(args, global_step):
    if args.lr_type == 'exponential':
        decay_steps_exp = 10000,
        decay_factors_exp = 0.96
        return tf.train.exponential_decay(args.lr_init, global_step, decay_steps_exp, decay_factors_exp)
    elif args.lr_type == 'cosine_decay':
        return tf.train.cosine_decay(args.lr_init, global_step, args.trainset_num * args.epochs // args.batch_size, 0.0001)
    elif args.lr_type == 'cosine_decay_restart':
        warm_up_epochs = 2
        return tf.train.cosine_decay_restarts(args.lr_init, global_step, warm_up_epochs * args.trainset_num // args.batch_size,
                                              t_mul=2.0, m_mul=1.0)
    else:
        raise ValueError('Unsupported learning rate type!')

def config_optimizer(optimizer_name, learning_rate, decay=0.9, momentum=0.9):
    if optimizer_name == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum)
    elif optimizer_name == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Unsupported optimizer type!')

def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])