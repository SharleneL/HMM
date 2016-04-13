__author__ = 'luoshalin'

import re
import operator

# making all letters uppercase
# replacing all non-alphabetic characters (numbers, punctuation, and whitespace, including line breaks) with spaces, and then collapsing sequences of spaces to be one space (this should take 2-3 lines of python code).
# You will be treating this dataset as a stream of characters (that's why we don't want any line breaks).
def preprocess(fpath):
    s = ''
    with open(fpath) as f:
        line = f.readline()
        while line != '':
            # process
            line = re.sub('[^a-zA-Z]+', ' ', line).strip().upper()
            if line != '':
                s = s + line + ' '
            line = f.readline()
    return s


def analyze(s):
    vowels = 'AEIOU'
    cc_cnt = 0
    vv_cnt = 0
    cv_cnt = 0
    vc_cnt = 0

    v_dic = dict()
    c_dic = dict()
    v_total = 0
    c_total = 0


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

    print 'P(C->C) = ' + str(float(cc_cnt) / (length - 1))
    print 'P(V->V) = ' + str(float(vv_cnt) / (length - 1))
    print 'P(C->V) = ' + str(float(cv_cnt) / (length - 1))
    print 'P(V->C) = ' + str(float(vc_cnt) / (length - 1))

    sorted_c_list = reversed(sorted(c_dic.items(), key=operator.itemgetter(1)))
    sorted_v_list = reversed(sorted(v_dic.items(), key=operator.itemgetter(1)))
    print '\nConsonant Distribution:'
    for (key, value) in sorted_c_list:
        print 'P(' + key + '): ' + str(float(value) / c_total)

    print '\nVowel Distribution:'
    for (key, value) in sorted_v_list:
        print 'P(' + key + '): ' + str(float(value) / v_total)