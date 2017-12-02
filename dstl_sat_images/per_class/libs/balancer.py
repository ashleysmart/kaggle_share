import numpy as np

def rebalance(data,classes,samples,ratio):
    # index the classes
    true_idx  = classes == 1.0
    false_idx = true_idx == False
    true_idx  = np.where(true_idx)[0]
    false_idx = np.where(false_idx)[0]

    np.random.shuffle(true_idx)
    np.random.shuffle(false_idx)

    # sample sizes
    true_target  = int(ratio*samples)
    false_target = samples - true_target
    
    # select true set
    if true_idx.shape[0] < true_target:
        # we are short in true classes
        repeats  = true_target / true_idx.shape[0]
        leftover = true_target % true_idx.shape[0]

        true_idx = np.concatenate((np.repeat(true_idx, repeats), true_idx[:leftover]), axis=0)
    else:
        true_idx = true_idx[:true_target]

    # select false set
    if false_idx.shape[0] < false_target:
        # we are short in true classes
        repeats  = false_target / false_idx.shape[0]
        leftover = false_target % false_idx.shape[0]

        false_idx = np.concatenate((np.repeat(false_idx, repeats), false_idx[:leftover]), axis=0)
    else:
        false_idx = false_idx[:false_target]
                                  
    idx_all = np.concatenate((false_idx, true_idx), axis=0)
    np.random.shuffle(idx_all)
    
    #return true_idx, false_idx, idx_all
    return data[idx_all], classes[idx_all]



#def min2_balance(train_patches,train_roi):      
#    true_idx  = np.argmax(train_roi == 1.0)
#    false_idx = np.argmax(train_roi == 0.0)
#    if (true_idx == 0):
#        if false_idx != 1:
#            #swap values
#            train_patches[1], train_patches[false_idx]  = train_patches[false_idx].copy(), train_patches[1].copy()
#            train_roi[1],     train_roi[false_idx]      = train_roi[false_idx].copy(),     train_roi[1].copy()
#    elif (false_idx == 0):
#        if true_idx != 1:
#            #swap values
#            train_patches[1], train_patches[true_idx]  = train_patches[true_idx].copy(), train_patches[1].copy()
#            train_roi[1],     train_roi[true_idx]      = train_roi[true_idx].copy(),     train_roi[1].copy()
