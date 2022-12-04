import numpy as np

attack = "EA_untargeted.npy"

attackReport = np.load(f"{attack}")

FP_EA_Unt = [38,38,29,34,33,26,25,26,39,36]


c1 = attackReport[0:100, :].copy()
c2 = attackReport[100:200, :].copy()
c3 = attackReport[200:300, :].copy()
c4 = attackReport[300:400, :].copy()
c5 = attackReport[400:500, :].copy()
c6 = attackReport[500:600, :].copy()
c7 = attackReport[600:700, :].copy()
c8 = attackReport[700:800, :].copy()
c9 = attackReport[800:900, :].copy()
c10 = attackReport[900:1000, :].copy()

t = 3
Rth = 51

def countAdvers(model):
    n = 0
    for i in model[:,0]:
        if i != 5:
            n +=1
    return n

def metrics(arr, t, Rth):
    ''' it returns number of adversarial images, Detection rate for Rth DR,
    number of True Positives, and number of False Negatives.'''
    c_clean = np.where(arr == 5, 0, arr)
    c_clean = c_clean[:,0:t].sum(axis=1)
    c_clean = c_clean*100/t
    TP = np.count_nonzero(c_clean >= Rth)
    DR = TP * 100 / countAdvers(arr)
    FN = countAdvers(arr) - TP
    return countAdvers(arr),DR, TP, FN

# advers, DR, TP, FN = metrics(c1, t, Rth)


advers, DR, TP, FN = metrics(c1, t, Rth)
precision = TP/(TP+FP_EA_Unt[0])
recall = TP/(TP+FN)
F1 = 2*(recall*precision)/(recall+precision)
print("%.3f, %.3f, %.3f" %(precision, recall, F1))