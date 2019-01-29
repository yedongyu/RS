"""
SLIM basic implementation. To understand deeply how it works we encourage you to
read "SLIM: Sparse LInear Methods for Top-N Recommender Systems".
"""
from sklearn.linear_model import SGDRegressor
from util import tsv_to_matrix
from metrics import compute_precision
from recommender import slim_recommender
import numpy as np
import operator
from scipy.sparse import lil_matrix


def slim_train(A, l1_reg=0.001, l2_reg=0.0001):
    """
    Computes W matrix of SLIM

    This link is useful to understand the parameters used:

        http://web.stanford.edu/~hastie/glmnet_matlab/intro.html

        Basically, we are using this:

            Sum( yi - B0 - xTB) + ...
        As:
            Sum( aj - 0 - ATwj) + ...

    Remember that we are wanting to learn wj. If you don't undestand this
    mathematical notation, I suggest you to read section III of:

        http://glaros.dtc.umn.edu/gkhome/slim/overview
    """
    alpha = l1_reg+l2_reg
    l1_ratio = l1_reg/alpha

    model = SGDRegressor(
        penalty='elasticnet',
        fit_intercept=False,
        alpha=alpha,
        l1_ratio=l1_ratio,
    )

    # TODO: get dimensions in the right way
    m, n = A.shape

    # Fit each column of W separately
    W = lil_matrix((n, n))

    for j in range(n):
        if j % 50 == 0:
            print('-> %2.2f%%' % ((j/float(n)) * 100))

        aj = A[:, j].copy()
        # We need to remove the column j before training
        A[:, j] = 0

        model.fit(A, aj.toarray().ravel())
        # We need to reinstate the matrix
        A[:, j] = aj

        w = model.coef_

        # Removing negative values because it makes no sense in our approach
        w[w<0] = 0

        for el in w.nonzero()[0]:
            W[(el, j)] = w[el]

    return W


def main(train_file, part_file ,test_file):


         
    AG = tsv_to_matrix(train_file, 942, 1682)
    AP = tsv_to_matrix(part_file, 942, 1682)

         
    W1 = slim_train(AG)
    W2 = slim_train(AP)
    # total_precision = []
    k = 2
    matrix_5 = np.zeros((21, k))
    matrix_10 = np.zeros((21, k))
    matrix_15 = np.zeros((21, k))
    matrix_20 = np.zeros((21, k))


    for i in range(0, 105, 5):
        gu = i / 100
        W = gu * W1 + (1 - gu) * W2
        print("gu: " + str(gu))
        recommendations = slim_recommender(AP, W)
        top5, top10, top15, top20 = compute_precision(recommendations, test_file)
        for j in range(2):
            matrix_5[int(i/5)][j] = top5[j]
            matrix_10[int(i/5)][j] = top10[j]
            matrix_15[int(i/5)][j] = top15[j]
            matrix_20[int(i/5)][j] = top20[j]

    hr_values = []
    hr_values1 = []
    index1, value1 = max(enumerate(matrix_5[:,0]), key=operator.itemgetter(1))
    index2, value2 = max(enumerate(matrix_10[:,0]), key=operator.itemgetter(1))
    index3, value3 = max(enumerate(matrix_15[:,0]), key=operator.itemgetter(1))
    index4, value4 = max(enumerate(matrix_20[:,0]), key=operator.itemgetter(1))
    hr_values.append(index1*0.05)
    hr_values.append(value1)
    hr_values.append(index2*0.05)
    hr_values.append(value2)
    hr_values.append(index3*0.05)
    hr_values.append(value3)
    hr_values.append(index4*0.05)
    hr_values.append(value4)
    hr_values1.append(matrix_5[20][0])
    hr_values1.append(matrix_10[20][0])
    hr_values1.append(matrix_15[20][0])
    hr_values1.append(matrix_20[20][0])

    arhr_values = []
    arhr_values1 = []
    index1, value1 = max(enumerate(matrix_5[:,1]), key=operator.itemgetter(1))
    index2, value2 = max(enumerate(matrix_10[:,1]), key=operator.itemgetter(1))
    index3, value3 = max(enumerate(matrix_15[:,1]), key=operator.itemgetter(1))
    index4, value4 = max(enumerate(matrix_20[:,1]), key=operator.itemgetter(1))

    arhr_values.append(index1*0.05)
    arhr_values.append(value1)
    arhr_values.append(index2*0.05)
    arhr_values.append(value2)
    arhr_values.append(index3*0.05)
    arhr_values.append(value3)
    arhr_values.append(index4*0.05)
    arhr_values.append(value4)
    
    arhr_values1.append(matrix_5[20][1])
    arhr_values1.append(matrix_10[20][1])
    arhr_values1.append(matrix_15[20][1])
    arhr_values1.append(matrix_20[20][1])

    print ('k8 top5: %s' % (matrix_5))
    print ('k8 top10: %s' % (matrix_10))
    print ('k8 top15: %s' % (matrix_15))
    print ('k8 top20: %s' % (matrix_20))

    print ('Max HR: %s' % (hr_values))
    print ('HR at gu = 1: %s' % (hr_values1))
    print ('Max ARHR: %s' % (arhr_values))
    print ('ARHR at gu = 1: %s' % (arhr_values1))
    


if __name__ == '__main__':
    main('./data/train_data.tsv',
         './data/k8/traink8_1.tsv',
         './data/k8/testk8_1.tsv')
    main('./data/train_data.tsv',
         './data/k8/traink8_2.tsv',
         './data/k8/testk8_2.tsv')
    main('./data/train_data.tsv',
         './data/k8/traink8_3.tsv',
         './data/k8/testk8_3.tsv')
    main('./data/train_data.tsv',
         './data/k8/traink8_4.tsv',
         './data/k8/testk8_4.tsv')
    main('./data/train_data.tsv',
         './data/k8/traink8_5.tsv',
         './data/k8/testk8_5.tsv')
    main('./data/train_data.tsv',
         './data/k8/traink8_6.tsv',
         './data/k8/testk8_6.tsv')
    main('./data/train_data.tsv',
         './data/k8/traink8_7.tsv',
         './data/k8/testk8_7.tsv')
    main('./data3/train_data.tsv',
         './data/k8/traink8_8.tsv',
         './data/k8/testk8_8.tsv')
    

    

