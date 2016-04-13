__author__ = 'luoshalin'

import numpy as np
from scipy.misc import logsumexp
import math


def gen_matrix(input_str):
    index_dic = gen_index_dic()
    A, B = init_matrix(index_dic)
    return index_dic, A, B


def forward(input_str, A, B, index_dic):
    alpha_table_list = []
    alpha_old = np.array([0.5, 0.5])

    # do calculation
    alpha_old = np.log(alpha_old)  # 1*2 vector; log of original alpha_old
    alpha_table_list.append(alpha_old)
    for t in range(len(input_str)):
        obsv_ch = input_str[t]  # current observed char
        # alpha_new = np.multiply(    np.dot(alpha_old, A),      B[index_dic[obsv_ch]]    )
        right_M = np.log(np.multiply(A, B[index_dic[obsv_ch]]))  # 2*2 matrix; B*A element-wise
        tmp_M = right_M + alpha_old[:, np.newaxis]
        # alpha_new = log_add_matrix_by_col(tmp_M)
        alpha_new = logsumexp(tmp_M, axis=0)  # axis=0: sum each col
        # got the new alpha
        # print 'alpha_new:'
        # print alpha_new
        # break
        alpha_table_list.append(alpha_new)
        alpha_old = alpha_new

    # calculate pcll
    pcll = logsumexp(alpha_table_list[-1]) / len(alpha_table_list)
    print alpha_table_list[-1]
    print pcll
    return np.array(alpha_table_list)


def backward(input_str, A, B, index_dic):
    beta_table_list = []
    beta_old = np.array([0.5, 0.5])

    # do calculation
    beta_old = np.log(beta_old)  # 1*2 vector; log of original beta_old
    beta_table_list.append(beta_old)
    for i in range(len(input_str)-1, -1, -1):
        obsv_ch = input_str[i]
        # beta_new = np.dot(    np.multiply(beta_old, B[index_dic[obsv_ch]]),         A   )
        right_M = np.log(np.multiply(A, B[index_dic[obsv_ch]]))  # 2*2 matrix; B*A element-wise
        tmp_M = right_M + beta_old
        # beta_new = log_add_matrix_by_col(tmp_M)
        beta_new = logsumexp(tmp_M, axis=1)
        # print beta_new
        # got the new beta
        # print beta_new
        beta_table_list.append(beta_new)
        beta_old = beta_new

    # calculate pcll
    print beta_table_list[-1]
    pcll = logsumexp(beta_old) / len(beta_table_list)
    print pcll
    beta_table_list.reverse()
    return np.array(beta_table_list)  # reverse the list


def forward_backward(alpha_table, beta_table, A_org, B_org, index_dic, input_str):  # input: 2 np matrix
    A = np.log(A_org)
    B = np.log(B_org)

    # ===== / E STEP start - generate xi_list / ===== #
    xi_list = []    # list of 2*2 matrix
    # p = np.exp(logsumexp(alpha_table[-1, :]))  # denominator
    p = logsumexp(alpha_table[-1, :])  # denominator
    # print 'alpha_table[-1, :]:'
    # print logsumexp(alpha_table[-1, :])
    for t in range(0, len(alpha_table) - 1):
        alpha_t = np.array(alpha_table[t])[:, np.newaxis]  # transpose to 2*1 vector
        # print alpha_table[t][:, np.newaxis]
        beta_t_plus_one = beta_table[t+1]           # 1*2 vector
        B_t_plus_one = B[index_dic[input_str[t+1]]] # 1*2 vector
        # xi_t = (A * alpha_t * B_t_plus_one * beta_t_plus_one) / p  # a 2*2 matrix
        xi_t = A + alpha_t + B_t_plus_one + beta_t_plus_one - p  # a 2*2 matrix

        xi_list.append(xi_t)
    # get xi_list (len = T-1, each elem is a 2*2 matrix for time t)
    # ===== / E STEP end / ===== #

    # ===== / M STEP start - update A & B / ===== #
    xi_sum = np.zeros(A.shape)
    # for i in range(len(xi_list)):
    #     xi_sum = xi_sum + xi_list[i]

    for i, j in np.ndindex(xi_sum.shape):
        xi_sum[i, j] = logsumexp([xi_list[x][i, j] for x in range(len(xi_list))])

    # update A
    # old non log sum:
    # new_A_denom = np.sum(xi_sum, axis=1)[:, np.newaxis]
    # new log sum:
    new_A_denom = np.zeros((xi_sum.shape[0], 1))
    for i in range(new_A_denom.shape[0]):
        tmp_l = []
        for m in range(len(xi_list)):
            tmp_l.append(logsumexp([xi_list[m][i, n] for n in range(xi_sum.shape[1])]))  # row sum
        new_A_denom[i, 0] = logsumexp(tmp_l)
    new_A = xi_sum - new_A_denom

    # update B
    new_B = np.zeros(B.shape)  # 27*2 matrix
    # new_B_denom = np.sum(xi_sum, axis=0)     # 1*2 vector
    new_B_denom = np.zeros((1, xi_sum.shape[0]))
    for i in range(new_B_denom.shape[1]):
        tmp_l = []
        for m in range(len(xi_list)):
            tmp_l.append(logsumexp([xi_list[m][i, n] for n in range(xi_sum.shape[0])]))  # col sum
        new_B_denom[0, i] = logsumexp(tmp_l)


    obsv_xlst_dic = dict()      # <obsv, current_obsv_xi_list>
    for i in range(len(input_str)-1):
        next_obsv = input_str[i+1]
        if next_obsv not in obsv_xlst_dic:
            xlst = []
            xlst.append(xi_list[i])
            obsv_xlst_dic[next_obsv] = xlst
        else:
            obsv_xlst_dic[next_obsv].append(xi_list[i])

    for obsv, xlst in obsv_xlst_dic.iteritems():
        xlst_sum = np.zeros((1, xlst[0].shape[1]))  # 1*2 matrix
        # for i in range(len(xlst)):
        #     cur_x = xlst[i]
        #     xlst_sum = xlst_sum + np.sum(cur_x, axis=0)   # 1*2 matrix, col sum
        for i in range(xlst_sum.shape[1]):
            tmp_l = []
            for m in range(len(xlst)):
                tmp_l.append(logsumexp([xlst[m][i, n] for n in range(xi_sum.shape[0])]))  # col sum
            xlst_sum[0, i] = logsumexp(tmp_l)

        new_B[index_dic[obsv]] = xlst_sum - new_B_denom

    # get new_A and new_B, 1 iter ends
    A = np.exp(new_A)
    # print A
    B = np.exp(new_B)
    return A, B


# HELPER FUNCTIONS
# function to index the characters: A:0, B:1, ..., Z:25, ' ':26
def gen_index_dic():
    dic = dict()
    for i in range(26):
        dic[chr(65 + i)] = i
    dic[' '] = 26
    return dic


# function to generate Transition Matrix A & Emission Matrix B
# C-0; V-1
def init_matrix(index_dic):
    # initialize A
    # A = np.array([[0.41, 0.27], [0.27, 0.05]])
    A = np.array([[0.24765875, 0.75234125], [0.95411204, 0.04588796]])

    # initialize B
    B = np.zeros((27, 2))

    # B[index_dic[' '], 0] = 0.269384956219
    # B[index_dic['T'], 0] = 0.098586348771
    # B[index_dic['N'], 0] = 0.0815486865703
    # B[index_dic['H'], 0] = 0.0791222702817
    # B[index_dic['S'], 0] = 0.0766431058128
    # B[index_dic['R'], 0] = 0.0742694377044
    # B[index_dic['D'], 0] = 0.0534339065302
    # B[index_dic['L'], 0] = 0.0461019094841
    # B[index_dic['M'], 0] = 0.0336533389598
    # B[index_dic['F'], 0] = 0.0305939445089
    # B[index_dic['W'], 0] = 0.0282202764005
    # B[index_dic['C'], 0] = 0.0275872982382
    # B[index_dic['Y'], 0] = 0.0251081337694
    # B[index_dic['G'], 0] = 0.0186201076063
    # B[index_dic['B'], 0] = 0.0170904103808
    # B[index_dic['P'], 0] = 0.0164046840384
    # B[index_dic['V'], 0] = 0.0149277349931
    # B[index_dic['K'], 0] = 0.00469458803671
    # B[index_dic['Q'], 0] = 0.00168794176601
    # B[index_dic['J'], 0] = 0.00116045996413
    # B[index_dic['X'], 0] = 0.00110771178394
    # B[index_dic['Z'], 0] = 5.27481801878e-05
    # B[index_dic['E'], 1] = 0.336270708495
    # B[index_dic['O'], 1] = 0.207848666432
    # B[index_dic['A'], 1] = 0.20091646105
    # B[index_dic['I'], 1] = 0.183762190107
    # B[index_dic['U'], 1] = 0.0712019739161

    c_plist = [5.72482639e-02,   6.76097579e-02,   5.32669429e-02,   2.92552278e-02,
               1.94083383e-02,   2.27931545e-02,   5.51975355e-02,   4.52549686e-02,
               3.37799449e-02,   2.98281194e-03,   1.84010429e-03,   5.99670829e-02,
               2.21161701e-02,   6.39406023e-02,   3.90074565e-02,   5.76843345e-02,
               3.62624878e-02,   6.25240503e-02,   1.04404729e-02,   5.33771198e-02,
               2.29924701e-02,   9.17564722e-05,   3.69603060e-02,   5.22156435e-02,
               3.65602181e-02,   3.41946055e-02,   2.30281727e-02]
    v_plist = [4.69000607e-02,   3.63109776e-02,   5.00662394e-02,   5.34804147e-02,
               3.07480701e-02,   3.67968195e-02,   5.73452960e-02,   1.01652530e-02,
               1.31955364e-02,   7.34181292e-02,   5.99057561e-02,   3.70296828e-02,
               4.54309822e-02,   4.47673511e-02,   4.73604908e-02,   6.74726205e-02,
               4.11496913e-02,   3.19366497e-03,   6.35058469e-02,   1.18046495e-02,
               1.92433268e-02,   3.95879605e-02,   2.35828607e-02,   1.01672814e-02,
               2.05918877e-02,   1.83547085e-02,   3.84244412e-02]
    B[index_dic['A'], 0] = c_plist[0]
    B[index_dic['B'], 0] = c_plist[1]
    B[index_dic['C'], 0] = c_plist[2]
    B[index_dic['D'], 0] = c_plist[3]
    B[index_dic['E'], 0] = c_plist[4]
    B[index_dic['F'], 0] = c_plist[5]
    B[index_dic['G'], 0] = c_plist[6]
    B[index_dic['H'], 0] = c_plist[7]
    B[index_dic['I'], 0] = c_plist[8]
    B[index_dic['J'], 0] = c_plist[9]
    B[index_dic['K'], 0] = c_plist[10]
    B[index_dic['L'], 0] = c_plist[11]
    B[index_dic['M'], 0] = c_plist[12]
    B[index_dic['N'], 0] = c_plist[13]
    B[index_dic['O'], 0] = c_plist[14]
    B[index_dic['P'], 0] = c_plist[15]
    B[index_dic['Q'], 0] = c_plist[16]
    B[index_dic['R'], 0] = c_plist[17]
    B[index_dic['S'], 0] = c_plist[18]
    B[index_dic['T'], 0] = c_plist[19]
    B[index_dic['U'], 0] = c_plist[20]
    B[index_dic['V'], 0] = c_plist[21]
    B[index_dic['W'], 0] = c_plist[22]
    B[index_dic['X'], 0] = c_plist[23]
    B[index_dic['Y'], 0] = c_plist[24]
    B[index_dic['Z'], 0] = c_plist[25]
    B[index_dic[' '], 0] = c_plist[26]
    B[index_dic['A'], 1] = v_plist[0]
    B[index_dic['B'], 1] = v_plist[1]
    B[index_dic['C'], 1] = v_plist[2]
    B[index_dic['D'], 1] = v_plist[3]
    B[index_dic['E'], 1] = v_plist[4]
    B[index_dic['F'], 1] = v_plist[5]
    B[index_dic['G'], 1] = v_plist[6]
    B[index_dic['H'], 1] = v_plist[7]
    B[index_dic['I'], 1] = v_plist[8]
    B[index_dic['J'], 1] = v_plist[9]
    B[index_dic['K'], 1] = v_plist[10]
    B[index_dic['L'], 1] = v_plist[11]
    B[index_dic['M'], 1] = v_plist[12]
    B[index_dic['N'], 1] = v_plist[13]
    B[index_dic['O'], 1] = v_plist[14]
    B[index_dic['P'], 1] = v_plist[15]
    B[index_dic['Q'], 1] = v_plist[16]
    B[index_dic['R'], 1] = v_plist[17]
    B[index_dic['S'], 1] = v_plist[18]
    B[index_dic['T'], 1] = v_plist[19]
    B[index_dic['U'], 1] = v_plist[20]
    B[index_dic['V'], 1] = v_plist[21]
    B[index_dic['W'], 1] = v_plist[22]
    B[index_dic['X'], 1] = v_plist[23]
    B[index_dic['Y'], 1] = v_plist[24]
    B[index_dic['Z'], 1] = v_plist[25]
    B[index_dic[' '], 1] = v_plist[26]


    return A, B


# # log add each column
# def log_add_matrix_by_col(M):
#     # elem1 = log_add(M[0, 0], M[1, 0])
#     # elem2 = log_add(M[0, 1], M[1, 1])
#     # return np.array([elem1, elem2])
#     res = logsumexp(M, axis=0)
#     return res



# calculate the log add for 2 numbers
# def log_add(left, right):
#     if right < left:
#         return left + np.log1p(np.exp(right - left))
#     elif right > left:
#         return right + np.log1p(np.exp(left - right))
#     else:
#         return left + math.log(2)