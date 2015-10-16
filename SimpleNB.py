import numpy as np
from collections import Counter

def naive_bayes(X,Y, gamma=1, verbose=False):
    unique_labels = np.unique(Y) 

    data_array = np.array(X) # the data in array format


    n = data_array.shape[0] # number of observations we have
    m = data_array.shape[1] # number of features
    k = len(unique_labels) # number of classes
    t = 2 #assuming x_i are boolean variables

    labels_array = np.array(Y)
    
    theta = (gamma/float(gamma*t)) * np.ones((m,t,k))
    pie = np.zeros(k) # filling in default values as the smoothening factor
    
    unique_feature_labels = [None]*m # array to hold the unique values of each feature

    for label_index, label in enumerate(unique_labels):
        matching_indicies = labels_array == label
        pie[label_index] = (np.sum(matching_indicies)+gamma)/(float(n)+gamma*k)
        if verbose:
            print('k=',label_index)
            print('y_k=',label)
            print('pie[k]=',pie[label_index])
        # obtain all the data that matches our labels.
        data_for_label = data_array[matching_indicies]

        # transpose the matrix because we want to now calculate the counts of the features.
        feature_observations = data_for_label.T

        for feature_index, feature_observation in enumerate(feature_observations):

            unique_feature_labels[feature_index] = list(sorted(np.unique(feature_observation)))

            for count in Counter(feature_observation).items():
                t_instance, N_condition = count[0], count[1] # value, frequency 

                theta[feature_index, t_instance, label_index] = (N_condition + gamma)/float(np.sum(matching_indicies) + gamma * t)
                if verbose:
                    print('theta(j=',feature_index,', t=',t_instance,'k=',label_index,') = ',theta[feature_index, t_instance, label_index])
            #endfor
        #endfor
    #endfor
    
    def predictor(X, verbose=False):
        to_return = []
        for to_classify in X:
            if verbose:
                print('to_classify:',to_classify)
            results = []
            for label in unique_labels:
                p = pie[label]
                if verbose:
                    print('P(y=',label,')=',pie[label])
                for j, val in zip(range(m), to_classify):
                    if verbose:
                        print('P(X_',j,'=',val,' | y=',label,')=', theta[j,val,label], '\n')
                    p = p * theta[j,val,label]
                    # print('\n')
                #endfor
                results.append((label, p))
            if verbose:
                print('Ps: ',results)
                print('Prediction:', max(results,key=lambda l: l[1]))
            #endfor
            to_return.append( results )
        #endfor
        return to_return
    return predictor
