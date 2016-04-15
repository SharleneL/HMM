__author__ = 'luoshalin'

import re
import operator
import numpy as np

# making all letters uppercase
# replacing all non-alphabetic characters (numbers, punctuation, and whitespace, including line breaks) with spaces, and then collapsing sequences of spaces to be one space (this should take 2-3 lines of python code).
# You will be treating this dataset as a stream of characters (that's why we don't want any line breaks).
def preprocess(fpath):
    s = ''

    f = open(fpath)
    lines = f.readlines()
    for line in lines:
        line = re.sub('[^a-zA-Z]+', ' ', line).strip().upper()
        if line != '':
            s = s + line + ' '
    f.close()
    return s


def analyze(s):
    vowels = 'AEIOU'
    cc_cnt = 0
    vv_cnt = 0
    cv_cnt = 0
    vc_cnt = 0

    v_dic = dict()
    c_dic = dict()

    corpus = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '
    for i in range(len(corpus)):
        ch = corpus[i]
        v_dic[ch] = 1
        c_dic[ch] = 1

    v_total = len(v_dic)
    c_total = len(c_dic)

    length = len(s)
    if length < 2:
        return

    if s[0] in vowels:
        v_total += 1
        v_dic[s[0]] = 1
    else:
        c_total += 1
        c_dic[s[0]] = 1

    for i in range(1, length):
        last = s[i-1]
        cur = s[i]
        if cur in vowels:
            v_total += 1
            # update 1
            if cur in v_dic:
                v_dic[cur] += 1
            else:
                v_dic[cur] = 1
            # update 2
            if last in vowels:  # vv
                vv_cnt += 1
            else:               # cv
                cv_cnt += 1
        else:
            c_total += 1
            # update 1
            if cur in c_dic:
                c_dic[cur] += 1
            else:
                c_dic[cur] = 1
            # update 2
            if last in vowels:  # vc
                vc_cnt += 1
            else:               # cc
                cc_cnt += 1

    print 'P(C->C) = ' + str(float(cc_cnt) / (c_total - 27))
    print 'P(C->V) = ' + str(float(cv_cnt) / (c_total - 27))
    print 'P(V->V) = ' + str(float(vv_cnt) / (v_total - 27))
    print 'P(V->C) = ' + str(float(vc_cnt) / (v_total - 27))

    sorted_c_list = reversed(sorted(c_dic.items(), key=operator.itemgetter(1)))
    sorted_v_list = reversed(sorted(v_dic.items(), key=operator.itemgetter(1)))

    # sorted_c_list = sorted(c_dic.items(), key=operator.itemgetter(0))
    # sorted_v_list = sorted(v_dic.items(), key=operator.itemgetter(0))

    cp_list = []
    vp_list = []

    print '\nConsonant Distribution:'
    for (key, value) in sorted_c_list:
        cp = float(value) / c_total
        print 'P(' + key + '): ' + str(cp)
        cp_list.append(cp)

    print '\nVowel Distribution:'
    for (key, value) in sorted_v_list:
        vp = float(value) / v_total
        print 'P(' + key + '): ' + str(vp)
        vp_list.append(vp)

    print sum(cp_list)
    print sum(vp_list)

    print '\n'
    print cp_list
    print vp_list
