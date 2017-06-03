from joblib import Parallel, delayed



l = [1,2,3,4,5,6,7]


def task(i):
    l[i] = l[i]+100
    return l[i]

bla = Parallel(n_jobs=4)(delayed(task)(i) for i in range(4))

print(l)

print(bla)