import keras.backend as K

def triplet_loss(y_true,y_pred,alpha=0.4):
    totallen=y_pred.shape.as_list()[-1]
    anchor=y_pred[:,0:int(totallen*1/3)]
    positive=y_pred[:,int(totallen*1/3):int(totallen*2/3)]
    negative=y_pred[:,int(totallen*2/3):totallen]
    pos_dist=K.sum(K.square(anchor-positive))
    neg_dist=K.sum(K.square(anchor-negative))
    basic_loss=pos_dist-neg_dist+alpha
    loss=K.maximum(basic_loss,0.0)
    return loss
