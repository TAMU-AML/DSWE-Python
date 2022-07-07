# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def matchcov(ref, obj, thres, circ_pos, flag):

    row_ref = ref.shape[0]
    row_obj = obj.shape[0]
    ncols = ref.shape[1]

    match = np.zeros(row_ref)

    index = np.arange(1, row_obj + 1, 1)

    for i in range(row_ref):

        ref_i = ref[i]
        score = np.abs(obj - ref_i)

        if flag:

            circref = np.array([ref_i[circ_pos]])[0]
            circdata = obj[:, circ_pos].reshape(-1, len(circ_pos))

            cols = circdata.shape[1]

            for j in range(cols):
                id_dcor = np.argwhere((circdata[:, j] - circref[j]) > 180)
                score_vec = score[:, j]
                circdata_vec = circdata[:, j]
                score_vec[id_dcor] = (
                    360 - (circdata_vec[id_dcor] - circref[j])) / (circref[j] + 1e-4)
                score[:, j] = score_vec

        decision = np.zeros(obj.shape)

        for k in range(decision.shape[1]):

            des = 1 * (score[:, k] < thres[k])
            decision[:, k] = des

        id_sum = decision.sum(axis=1)
        id_index = np.where(id_sum == obj.shape[1])[0]

        id_num = len(id_index)

        if id_num > 0:

            score_adjusted = score / thres[None, :]
            max_score = score_adjusted[id_index, :].max(axis=1)
            id_min = id_index[np.argmin(max_score)]
            match[i] = index[id_min]
            unmatched_id = np.where(index != match[i])

            obj = obj[unmatched_id[0], :]
            index = index[unmatched_id[0]]

    return match


def matching(Xlist, ylist, weight, circ_pos):

    flag = False
    if circ_pos:
        flag = True

    refid = len(Xlist) - 1

    ref = Xlist[refid]
    test_id = list(range(refid))
    ratio = np.std(ref.astype(float), axis=0)
    thresh = ratio * weight

    match_id = [0] * len(test_id)
    for idx in test_id:
        match_id[idx] = matchcov(
            ref, Xlist[test_id[idx]], thresh, circ_pos, flag)

    matched_X = [[]] * len(Xlist)

    ref_id = match_id[0] > 0

    if len(Xlist) >= 3:
        for i in range(1, len(Xlist) - 1):
            ref_id = ref_id & (match_id[i] > 0)

    ref_ID = np.where(ref_id)[0]
    matched_X[refid] = Xlist[refid][ref_ID, :]

    if ylist:
        matched_y = [[]] * len(Xlist)
        matched_y[refid] = ylist[refid][ref_ID, :]

    for j in range(len(Xlist) - 1):
        idx = match_id[j][ref_ID].astype('int32') - 1
        matched_X[test_id[j]] = Xlist[test_id[j]][idx, :]

        if ylist:
            matched_y[test_id[j]] = ylist[test_id[j]][idx, :]

    if ylist:
        return matched_X, matched_y
    else:
        return matched_X


def min_max(data):
    if isinstance(data, list):
        data = np.concatenate((data[0], data[1]))

    return {'min': np.min(data, axis=0), 'max': np.max(data, axis=0)}
