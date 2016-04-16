__author__ = 'luoshalin'

import sys
from preprocess import preprocess
from preprocess import analyze
from forwardBackword import gen_matrix
from forwardBackword import forward
from forwardBackword import backward
from forwardBackword import forward_backward
from forwardBackword import plot_pcll
from forwardBackword import vtb_decode
from forwardBackword import ll_decode


def main(argv):
    # x = sys.argv[2]
    # hmm_train_file_path = '../../data/hmm_train_data'
    # hmm_train_file_path = '../../data/hmm_test_data'
    # hmm_train_file_path = '../../data/hmm_train_data_jpn'
    hmm_train_file_path = '../../data/hmm_test_data_jpn'
    vtb_train_file_path = '../../data/viterbi_train_data'
    # F-B stopping criteria
    threshold = 1e-5
    param_log_filepath = 'log1'

    # preprocess
    input_str = preprocess(hmm_train_file_path)
    # do analysis
    analyze(input_str)

    # forward & backward algorithms
    index_dic, A, B = gen_matrix()
    pcll_old = -1000.0
    pcll_new = -1000.0
    pcll_list = []
    itr = 1
    while True:
        alpha_table, pcll_alpha = forward(input_str, A, B, index_dic)
        beta_table, pcll_beta = backward(input_str, A, B, index_dic)
        A, B = forward_backward(alpha_table, beta_table, A, B, index_dic, input_str)

        # update pcll
        pcll_new = pcll_alpha
        pcll_list.append(pcll_new)
        if abs(pcll_new - pcll_old) < threshold:
            break

        print 'ITERATION#' + str(itr) + ': ' + str(pcll_new)

        # update
        pcll_old = pcll_new
        itr += 1

    # plot
    plot_pcll(pcll_list)

    print '\n=========FINAL A & B==========='
    print 'A:'
    print A
    print '\nB:'
    print B
    print '==============================\n'


    # viterbi decoder
    vtb_input_str = preprocess(vtb_train_file_path)
    vtb_hidden_state_list = vtb_decode(vtb_input_str, A, B, index_dic)
    ll_hidden_state_list = ll_decode(vtb_input_str, B, index_dic)
    print ll_hidden_state_list

    print 'END!'


if __name__ == '__main__':
    main(sys.argv[1:])