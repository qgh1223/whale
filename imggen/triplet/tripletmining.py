import numpy as np
import tensorflow as tf

def cosine_distance(features):
    distance=1-tf.matmul(features,features,transpose_b=True)
    return distance

def semi_hard_mining(distance,n_class_per_iter,n_img_per_class,thresold):
    N=n_class_per_iter*n_img_per_class
    N_pair=n_img_per_class*n_img_per_class

    pre_idx=np.arange(N)
    arch_indexes=np.repeat(pre_idx,n_img_per_class)
    pos_indexes=np.repeat(pre_idx.reshape(n_class_per_iter,n_img_per_class),n_img_per_class,axis=0).reshape(-1)
    pos_pair_indexes=np.stack([arch_indexes,pos_indexes],1)

    arch_indexes=tf.constant(arch_indexes)
    pos_indexes=tf.constant(pos_indexes)
    pos_pair_indexes=tf.constant(pos_pair_indexes)

    with tf.control_dependencies([tf.assert_equal(tf.shape(distance)[0],N)]):
        pos_distance=tf.gather_nd(distance,pos_pair_indexes)
    neg_distance=tf.gather(distance,arch_indexes)

    neg_pos_mask=np.ones(shape=[N*n_img_per_class,N])
    for i in range(n_class_per_iter):
        neg_pos_mask[i*N_pair:(i+1)*N_pair,i*n_img_per_class:(i+1)*n_img_per_class]=0
    neg_pos_mask=tf.constant(neg_pos_mask)
    candiate_mask=(neg_distance-tf.expand_dims(pos_distance)-1)<thresold
    candiate_mask=tf.logical_and(candiate_mask,neg_pos_mask)
    deletion_mask=tf.reduce_any(candiate_mask,axis=1)
    arch_indexes=tf.boolean_mask(arch_indexes,deletion_mask)
    pos_indexes=tf.boolean_mask(pos_indexes,deletion_mask)
    candiate_mask=tf.boolean_mask(candiate_mask,deletion_mask)
    n_candidate_per_archor=tf.reduce_sum(tf.to_int32(candiate_mask),axis=1)
    sampler=tf.distributions.Uniform(0.,tf.to_float(n_candidate_per_archor)-1e-3)
    sample_idx=tf.to_int32(tf.floor(tf.reshape(sampler.sample(1),[-1])))
    start_idx=tf.cumsum(n_candidate_per_archor,exclusive=True)
    sample_idx=start_idx+sample_idx
    candiate_indexes=tf.where(candiate_mask)
    neg_indexes=tf.gather(candiate_indexes,sample_idx)[:,1]
    return (tf.stop_gradient(arch_indexes),
            tf.stop_gradient(pos_indexes),
            tf.stop_gradient(neg_indexes))