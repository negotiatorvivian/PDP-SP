#! -*- coding: utf-8 -*-

# import tensorflow as tf
import torch

'''
inputs是一个形如(batch_size, seq_len, word_size)的张量；
函数返回一个形如(batch_size, seq_len, position_size)的位置张量。
'''
# def Position_Embedding(inputs, position_size):
#     batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
#     position_j = 1. / tf.pow(10000., 2 * tf.range(position_size / 2, dtype=tf.float32) / position_size)
#     position_j = tf.expand_dims(position_j, 0)
#     position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
#     position_i = tf.expand_dims(position_i, 1)
#     position_ij = tf.matmul(position_i, position_j)
#     position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
#     position_embedding = tf.expand_dims(position_ij, 0) + tf.zeros((batch_size, seq_len, position_size))
#     return position_embedding


def position_embedding(inputs, position_size):
    batch_size, seq_len = inputs.shape[0], inputs.shape[1]
    position_j = 1. / torch.pow(10000., 2 * torch.arange(position_size / 2, dtype=torch.float32) / position_size)
    position_j = position_j.expand([1, position_j.shape[0]])
    position_i = torch.arange(seq_len.to(torch.float32), dtype=torch.float32)
    position_i = position_i.expand([position_i.shape[0], 1])
    position_ij = torch.mul(position_i, position_j)
    position_ij = torch.cat((torch.cos(position_ij), torch.sin(position_ij)), 1)
    position_embedding = position_ij.expand([1] + list(position_ij.shape)) + torch.zeros((batch_size, seq_len,
                                                                                          position_size))
    return position_embedding


'''
inputs是一个二阶以上的张量，代表输入序列，比如形如(batch_size, seq_len, input_size)的张量；
seq_len是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；
mode分为mul和add，mul是指把多出部分全部置零，一般用于全连接层之前；
add是指把多出部分全部减去一个大的常数，一般用于softmax之前。
'''
# def Mask(inputs, seq_len, mode='mul'):
#     if seq_len == None:
#         return inputs
#     else:
#         mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
#         for _ in range(len(inputs.shape)-2):
#             mask = tf.expand_dims(mask, 2)
#         if mode == 'mul':
#             return inputs * mask
#         if mode == 'add':
#             return inputs - (1 - mask) * 1e12


def mask(inputs, seq_len, mode='mul'):
    if seq_len is None:
        return inputs
    else:
        mask = sequence_mask(seq_len).to(dtype=torch.float32)
        for _ in range(len(inputs.shape)-2):
            mask = torch.unsqueeze(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask


'''
普通的全连接
inputs是一个二阶或二阶以上的张量，即形如(batch_size,...,input_size)。
只对最后一个维度做矩阵乘法，即输出一个形如(batch_size,...,ouput_size)的张量。
'''
# def Dense(inputs, ouput_size, bias=True, seq_len=None):
#     input_size = int(inputs.shape[-1])
#     W = tf.Variable(tf.random_uniform([input_size, ouput_size], -0.05, 0.05))
#     if bias:
#         b = tf.Variable(tf.random_uniform([ouput_size], -0.05, 0.05))
#     else:
#         b = 0
#     outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b
#     outputs = tf.reshape(outputs, tf.concat([tf.shape(inputs)[:-1], [ouput_size]], 0))
#
#     if seq_len != None:
#         outputs = Mask(outputs, seq_len, 'mul')
#     return outputs


def dense(inputs, output_size, bias=True, seq_len=None):
    input_size = int(inputs.shape[-1])
    W = torch.tensor(input_size, output_size).uniform_(-0.05, 0.05)
    if bias:
        b = torch.tensor(output_size).uniform_(-0.05, 0.05)
    else:
        b = 0
    outputs = inputs.reshape(-1, input_size).mm(W) + b
    outputs = outputs.reshape(list(inputs.shape[:-1]) + [output_size], 0)
    if seq_len is not None:
        outputs = mask(outputs, seq_len)
    return outputs


'''
Multi-Head Attention的实现
'''
# def Attention(Q, K, V, nb_head, size_per_head, Q_len=None, V_len=None):
#     #对Q、K、V分别作线性映射
#     Q = Dense(Q, nb_head * size_per_head, False)
#     Q = tf.reshape(Q, (-1, tf.shape(Q)[1], nb_head, size_per_head))
#     Q = tf.transpose(Q, [0, 2, 1, 3])
#     K = Dense(K, nb_head * size_per_head, False)
#     K = tf.reshape(K, (-1, tf.shape(K)[1], nb_head, size_per_head))
#     K = tf.transpose(K, [0, 2, 1, 3])
#     V = Dense(V, nb_head * size_per_head, False)
#     V = tf.reshape(V, (-1, tf.shape(V)[1], nb_head, size_per_head))
#     V = tf.transpose(V, [0, 2, 1, 3])
#     #计算内积，然后mask，然后softmax
#     A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))
#     A = tf.transpose(A, [0, 3, 2, 1])
#     A = Mask(A, V_len, mode='add')
#     A = tf.transpose(A, [0, 3, 2, 1])
#     A = tf.nn.softmax(A)
#     #输出并mask
#     output = tf.matmul(A, V)
#     output = tf.transpose(output, [0, 2, 1, 3])
#     output = tf.reshape(output, (-1, tf.shape(output)[1], nb_head * size_per_head))
#     output = Mask(output, Q_len, 'mul')
#     return output


def attention(X, nb_head, size_per_head, Q_len=None, V_len=None):
    X = dense(X, nb_head * size_per_head, False)
    X = X.reshape(-1, X.shape[1], nb_head, size_per_head)
    X = X.permute(0, 2, 1, 3)
    A = X.mm(X.T) / torch.sqrt(float(size_per_head))
    A = A.permute(0, 3, 2, 1)
    A = mask(A, V_len, mode='add')
    A = A.permute(0, 3, 2, 1)
    A = torch.nn.softmax(A)
    output = A.mm(X)
    output = output.permute(0, 2, 1, 3)
    output = output.reshape(-1, output.shape[1], nb_head * size_per_head)
    output = mask(output, Q_len, 'mul')
    return output

