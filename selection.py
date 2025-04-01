from __future__ import division
import numpy as np
class Selection(object):

    def RouletteSelection(self, _a, k):
        a = np.asarray(_a)
        idx = np.argsort(a)
        idx = idx[::-1]
        sort_a = a[idx]
        sum_a = np.sum(a).astype(np.float)
        selected_index = []
        for i in range(k):
            u = np.random.rand()*sum_a
            sum_ = 0
            for i in range(sort_a.shape[0]):
                sum_ +=sort_a[i]
                if sum_ > u:
                    selected_index.append(idx[i])
                    break
        return selected_index


    def binary_tournament_selection(self, _a, k):
        selected_index = []
        i = 0
        while i < k:
            [idx1, idx2] = np.random.choice(np.arange(len(_a)), size=2, replace=False)
            if _a[idx1] >= _a[idx2] and idx1 not in selected_index:
                selected_index.append(idx1)
                i += 1
            elif _a[idx1] < _a[idx2] and idx2 not in selected_index:
                selected_index.append(idx2)
                i += 1

        return selected_index

if __name__ == '__main__':
    s = Selection()
    # a = [1, 3, 2, 1, 4, 4, 5]
    a = np.arange(1,41)
    # selected_index = s.RouletteSelection(a, k=20)
    selected_index = s.binary_tournament_selection(a, k=20)
    print(len(selected_index))
    new_a =[a[i] for i in selected_index]
    print(list(np.asarray(a)[selected_index]))
    print(new_a)


