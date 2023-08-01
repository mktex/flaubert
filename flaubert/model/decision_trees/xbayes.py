
N = 147
x0 = 32  # Test == True
x1 = 15  # Test == True, K == True
x4 = 8  # Test == False, K == True
p0 = p1 = x2 = p2 = x3 = p3 = p4 = x5 = p5 = 0


def set_data(xN, test_true, test_true_k_true, test_false_k_true):
    global N, x0, p0, x1, p1, x2, p2, x3, p3, x4, p4, x5, p5
    N = xN
    x0 = test_true
    x1 = test_true_k_true
    x4 = test_false_k_true

    # x0
    p0 = round(x0 / N, 2)
    # x1
    p1 = round(x1 / x0, 2)
    x2 = x0 - x1  # Test == True, K == False
    p2 = round(x2 / x0, 2)

    x3 = N - x0  # Test == False
    p3 = round(x3 / N, 2)
    # x4
    p4 = round(x4 / x3, 2)
    x5 = x3 - x4  # Test == False, K == False
    p5 = round(x5 / x3, 2)


def stats_basic():
    global N, x0, p0, x1, p1, x2, p2, x3, p3, x4, p4, x5, p5
    print("*******************************************************")
    print("P(Test == True):                ", p0)
    print("P(K == True  | Test == True):   ", p1)
    print("P(K == False | Test == True):   ", p2)
    print("P(Test == False):               ", p3)
    print("P(K == True  | Test == False):  ", p4)
    print("P(K == False | Test == False):  ", p5)


def confusion_matrix():
    print("*******************************************************")
    print("P(K==True | Test == True)", p1)
    print("P(T == True)", p0)
    print("P(K==True | Test == False)", p4)
    print("P(T==False)", p3)
    t1 = (p1 * p0) / (p1 * p0 + p4 * p3)
    print("P(Test == True | K == True)", t1)

    print()
    print("P(K==True | Test == False)", p4)
    print("P(T == False)", p3)
    print("P(K==True | Test == True)", p1)
    print("P(T==True)", p0)
    t2 = (p4 * p3) / (p4 * p3 + p1 * p0)
    print("P(Test == False | K == True)", t2)

    print()
    print("P(K==False | Test == True)", p2)
    print("P(T == True)", p0)
    print("P(K==False | Test == False)", p5)
    print("P(T==False)", p3)
    t3 = (p2 * p0) / (p2 * p0 + p5 * p3)
    print("P(Test == True | K == False)", t3)

    print()
    print("P(K==False | Test == False)", p5)
    print("P(T == False)", p3)
    print("P(K==False | Test == True)", p2)
    print("P(T==True)", p0)
    t4 = p5 * p3 / (p5 * p3 + p2 * p0)
    print("P(Test == False | K == False)", t4)

    print()
    t1 *= 100
    t2 *= 100
    t3 *= 100
    t4 *= 100
    print("**************** Confusion Matrix *****************")
    print("                     | K == True)     | K == False)\n")
    print("(Test == True  |     " + "%.2f" % (t1) + "%            " + "%.2f" % (t3) + "%\n")
    print("(Test == False |     " + "%.2f" % (t2) + "%            " + "%.2f" % (t4) + "%\n")
    print("***************************************************")


# set_data(N, x0, x1, x4)

"""
if True:
    N = 200
    x0 = 150 # Test == True
    x1 = 120 # Test == True, K == True => sollte so hoch wie möglich sein
    x4 = 10 # Test == False, K == True => sollte so niedrig wie möglich sein
    xBayes.set_data(N, x0, x1, x4)
    xBayes.confusion_matrix()
    xBayes.stats_basic()

"""
