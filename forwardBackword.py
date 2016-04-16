__author__ = 'luoshalin'

import numpy as np
from scipy.misc import logsumexp
import math
import matplotlib.pyplot as plt
import pylab as pl


def gen_matrix():
    index_dic = gen_index_dic()
    A, B = init_matrix(index_dic)
    return index_dic, A, B


def forward(input_str, A, B, index_dic):
    alpha_table_list = []
    alpha_old = np.array([0.5, 0.5])

    # do calculation
    alpha_old = np.log(alpha_old)           # 1*2 vector; log of original alpha_old
    alpha_table_list.append(alpha_old)
    for t in range(len(input_str)):
        obsv_ch = input_str[t]              # current observed char
        right_M = np.log(A * B[index_dic[obsv_ch]])     # 2*2 matrix; B*A element-wise
        tmp_M = right_M + alpha_old[:, np.newaxis]
        alpha_new = logsumexp(tmp_M, axis=0)            # axis=0: sum each col
        # ASSERT: got the new alpha
        alpha_table_list.append(alpha_new)
        alpha_old = alpha_new

    # calculate pcll
    pcll = logsumexp(alpha_table_list[-1]) / (len(alpha_table_list)-1)
    return np.array(alpha_table_list), pcll


def backward(input_str, A, B, index_dic):
    beta_table_list = []
    beta_old = np.array([1.0, 1.0])
    alpha_init = np.array([0.5, 0.5])

    # do calculation
    beta_old = np.log(beta_old)         # 1*2 vector; log of original beta_old
    beta_table_list.append(beta_old)
    for i in range(len(input_str)-1, -1, -1):
        obsv_ch = input_str[i]
        right_M = np.log(np.multiply(A, B[index_dic[obsv_ch]]))  # 2*2 matrix; B*A element-wise
        tmp_M = right_M + beta_old
        beta_new = logsumexp(tmp_M, axis=1)
        beta_table_list.append(beta_new)
        beta_old = beta_new

    # calculate pcll
    pcll = logsumexp(beta_table_list[-1] + np.log(alpha_init)) / (len(beta_table_list)-1)
    beta_table_list.reverse()
    return np.array(beta_table_list), pcll  # reverse the list


def forward_backward(alpha_table, beta_table, A_org, B_org, index_dic, input_str):  # input: 2 np matrix
    A = np.log(A_org)
    B = np.log(B_org)

    # ===== / E STEP start - generate xi_list / ===== #
    xi_list = []    # list of 2*2 matrix
    p = logsumexp(alpha_table[-1, :])  # denominator
    for t in range(0, len(input_str)):
        obsv_ch = input_str[t]
        alpha_t = np.array(alpha_table[t])[:, np.newaxis]   # transpose to 2*1 vector
        beta_t_plus_one = beta_table[t+1]                   # 1*2 vector
        B_t_plus_one = B[index_dic[obsv_ch]]                # 1*2 vector
        xi_t = A + alpha_t + B_t_plus_one + beta_t_plus_one - p  # a 2*2 matrix

        xi_list.append(xi_t)

    # ASSERT: got xi_list (len = T-1, each elem is a 2*2 matrix for time t)
    # ===== / E STEP end / ===== #

    # ===== / M STEP start - update A & B / ===== #
    xi_sum = np.zeros(A.shape)

    for i, j in np.ndindex(xi_sum.shape):
        xi_sum[i, j] = logsumexp([xi_list[x][i, j] for x in range(len(xi_list))])

    # update A
    row_cnt = xi_sum.shape[0]
    col_cnt = xi_sum.shape[1]
    new_A_denom = np.zeros((row_cnt, 1))
    for i in range(new_A_denom.shape[0]):
        tmp_l = []
        for m in range(len(xi_list)):
            tmp_l.append(logsumexp([xi_list[m][i, n] for n in range(col_cnt)]))  # row sum
        new_A_denom[i, 0] = logsumexp(tmp_l)
    new_A = xi_sum - new_A_denom

    # update B
    new_B = np.zeros(B.shape)  # 27*2 matrix
    new_B_denom = np.zeros((1, xi_sum.shape[0]))
    for i in range(new_B_denom.shape[1]):
        tmp_l = []
        for m in range(len(xi_list)):
            tmp_l.append(logsumexp([xi_list[m][n, i] for n in range(xi_sum.shape[0])]))  # col sum
        new_B_denom[0, i] = logsumexp(tmp_l)

    obsv_xlst_dic = dict()      # <obsv, current_obsv_xi_list>
    for i in range(len(input_str)):
        next_obsv = input_str[i]
        if next_obsv not in obsv_xlst_dic:
            xlst = []
            xlst.append(xi_list[i])
            obsv_xlst_dic[next_obsv] = xlst
        else:
            obsv_xlst_dic[next_obsv].append(xi_list[i])

    for obsv, xlst in obsv_xlst_dic.iteritems():
        xlst_sum = np.zeros((1, xlst[0].shape[1]))  # 1*2 matrix
        for i in range(xlst_sum.shape[1]):
            tmp_l = []
            for m in range(len(xlst)):
                tmp_l.append(logsumexp([xlst[m][n, i] for n in range(xi_sum.shape[0])]))  # col sum
            xlst_sum[0, i] = logsumexp(tmp_l)

        new_B[index_dic[obsv]] = xlst_sum - new_B_denom

    # ASSERT: got new_A and new_B, 1 iter ends
    A = np.exp(new_A)
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
    A = np.array([[0.60, 0.40], [0.89, 0.11]])  #
    # A = np.load("ttta.npy")

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

    # cz's init p lists
    # B = np.load("tttb_shift.npy").T
    # c_plist = B[:, 0].tolist()
    # v_plist = B[:, 1].tolist()

    # Natural p lists w/ pseudocounts
    # c_plist = [0.2690930159064574, 5.267038870746866e-05, 0.017117876329927315, 0.02759928368271358, 0.05340777414937322, 5.267038870746866e-05, 0.030601495839039292, 0.018645317602443905, 0.07905825344991046, 5.267038870746866e-05, 0.0012114189402717792, 0.00474033498367218, 0.04608659011903508, 0.033656378384072476, 0.08148109133045402, 5.267038870746866e-05, 0.016433161276730224, 0.0017381228273464658, 0.07421257768882335, 0.07658274518065944, 0.09844095649425892, 5.267038870746866e-05, 0.0149583903929211, 0.028231328347203204, 0.0011587485515643105, 0.02512377541346255, 0.00010534077741493732]
    # v_plist = [0.00011712344811431249, 0.20039821972358865, 0.00011712344811431249, 0.00011712344811431249, 0.00011712344811431249, 0.33532443195127665, 0.00011712344811431249, 0.00011712344811431249, 0.00011712344811431249, 0.18329819629889904, 0.00011712344811431249, 0.00011712344811431249, 0.00011712344811431249, 0.00011712344811431249, 0.00011712344811431249, 0.2073085031623331, 0.00011712344811431249, 0.00011712344811431249, 0.00011712344811431249, 0.00011712344811431249, 0.00011712344811431249, 0.07109393300538767, 0.00011712344811431249, 0.00011712344811431249, 0.00011712344811431249, 0.00011712344811431249, 0.00011712344811431249]

    # random init
    # A
    # alist1 = np.random.dirichlet(np.ones(2), size=1)[0]
    # alist2 = np.random.dirichlet(np.ones(2), size=1)[0]
    # A = np.array([alist1, alist2])
    # B
    # c_plist = np.random.dirichlet(np.ones(27), size=1)[0]
    # v_plist = np.random.dirichlet(np.ones(27), size=1)[0]

    # best pc-ll init - Q8 & Q9
    # A = np.array([[0.07471292, 0.92528708], [0.45377186, 0.54622814]])
    # c_plist = [2.42995968e-04,   2.39561379e-02,   3.86445056e-02,   7.48992104e-02,
    #            1.19827010e-06,   4.28844444e-02,   2.43660067e-02,   1.01347900e-01,
    #            2.12711086e-10,   1.62665134e-03,   6.58053955e-03,   5.28622002e-02,
    #            4.71728889e-02,   1.14309018e-01,   8.39301924e-15,   2.13094211e-02,
    #            2.36603832e-03,   1.04105683e-01,   1.07432927e-01,   1.37425390e-01,
    #            1.23598443e-03,   2.09246513e-02,   3.95572031e-02,   1.48198365e-03,
    #            3.51930823e-02,   7.39386972e-05,   3.41560233e-12]
    # v_plist = [1.22386446e-01,   4.77556284e-16,   2.46660729e-05,   6.69143886e-07,
    #            2.05229533e-01,   1.61571568e-18,   1.68204842e-03,   9.27182934e-03,
    #            1.12152623e-01,   2.12232237e-32,   4.37985457e-09,   1.14055540e-02,
    #            3.78263149e-20,   2.01778559e-07,   1.26852935e-01,   1.63468164e-03,
    #            9.71521365e-23,   2.47718013e-09,   6.94921483e-10,   7.42933037e-04,
    #            4.22568466e-02,   9.37368279e-35,   1.19095014e-31,   6.85959322e-05,
    #            1.68525270e-06,   2.30633767e-58,   3.66288745e-01]


    # best pc-ll init - Q14
    A = np.array([[0.10284645, 0.89715355], [0.63859265, 0.36140735]])
    c_plist = [2.78752401e-01,   1.24166202e-18,   2.44387237e-03,   2.41589415e-24,
               1.39376200e-01,   5.90119590e-25,   4.90652895e-27,   5.03046270e-14,
               2.25329557e-01,   1.75119914e-37,   4.84838269e-34,   2.71948586e-03,
               2.94218087e-29,   7.41735247e-28,   2.08595451e-01,   3.08386213e-15,
               1.00000000e+00,   7.59286387e-59,   6.99839271e-03,   2.56550625e-13,
               1.35369663e-01,   7.50254693e-85,   5.75820645e-21,   3.40981529e-04,
               7.39851018e-05,   1.33249330e-58,   9.41542497e-09]
    v_plist = [1.05046820e-13,   1.74138235e-02,   6.57304018e-03,   3.56771018e-02,
               1.73866643e-16,   4.67200143e-03,   2.18431236e-02,   4.65379883e-02,
               4.21611783e-05,   7.88779462e-03,   9.57456917e-02,   4.85998242e-03,
               6.05539926e-02,   9.61097437e-02,   5.07217366e-12,   3.76187128e-03,
               1.00000000e+00,   6.34057337e-02,   6.27930930e-02,   9.86581081e-02,
               3.02633617e-09,   2.48768907e-03,   2.83960606e-02,   2.09822636e-40,
               2.59770617e-02,   8.00914531e-03,   3.08594789e-01]

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


    print '\n=========INIT A & B==========='
    print 'A:'
    print A
    print '\nB c_plist:'
    print c_plist
    print 'B v_plist:'
    print v_plist
    print '==============================\n'

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


# function to plot the pcll list
def plot_pcll(pcll_list):
    x = range(len(pcll_list))
    pl.plot(x, pcll_list)
    pl.show()


def vtb_decode(input_str, A, B, index_dic):
    vtb_table_list = []
    vtb_old = np.array([0.5, 0.5])

    # do calculation
    vtb_old = np.log(vtb_old)           # 1*2 vector
    # vtb_table_list.append(vtb_old.tolist())
    for t in range(len(input_str)):
        obsv_ch = input_str[t]              # current observed char
        right_M = np.log(A * B[index_dic[obsv_ch]])     # 2*2 matrix; B*A element-wise
        tmp_M = right_M + vtb_old[:, np.newaxis]
        # vtb_new = np.log(np.amax(tmp_M, axis=0))
        vtb_new = np.amax(tmp_M, axis=0)
        # ASSERT: got the new vtb
        vtb_table_list.append(vtb_new.tolist())
        # hidden_state_list.append(tmp_M.argmax(axis=0))
        vtb_old = vtb_new
    hidden_state_list = np.array(vtb_table_list).argmax(axis=1)
    # np.array(vtb_table_list)

    # ASSERT: got vtb_table_list &  hidden_state_list
    return hidden_state_list


def ll_decode(input_str, B, index_dic):
    hidden_state_list = []
    for t in range(len(input_str)):
        obsv_ch = input_str[t]
        if B[index_dic[obsv_ch]][0] > B[index_dic[obsv_ch]][1]:
            hidden_state_list.append(0)
        else:
            hidden_state_list.append(1)
    return hidden_state_list