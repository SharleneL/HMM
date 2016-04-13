__author__ = 'luoshalin'

import sys
from preprocess import preprocess
from preprocess import analyze
from forwardBackword import gen_matrix
from forwardBackword import forward
from forwardBackword import backward
from forwardBackword import forward_backward


def main(argv):
    # x = sys.argv[2]
    train_file_path = '../../data/train_data'

    # preprocess
    input_str = preprocess(train_file_path)
    # analysis
    # analyze(input_str)

    # forward & backward algorithms
    index_dic, A, B = gen_matrix(input_str)
    # alpha_table = forward(input_str, A, B, index_dic)
    # beta_table = backward(input_str, A, B, index_dic)
    # A, B = forward_backward(alpha_table, beta_table, A, B, index_dic, input_str)
    for i in range(10):
        alpha_table = forward(input_str, A, B, index_dic)
        beta_table = backward(input_str, A, B, index_dic)
        A, B = forward_backward(alpha_table, beta_table, A, B, index_dic, input_str)

    print 'END!'



if __name__ == '__main__':
    main(sys.argv[1:])