import numpy as np
def preprocess_two_weights(M, K, weight_num, BM, BY, bm, by, weight, final_weight):
    weight = np.reshape(weight, (weight_num // 2, 2))
    hi_weight = np.multiply(np.split(weight, 2, axis=1)[0], 3)
    lo_weight = np.split(weight, 2, axis=1)[1]

    weight = np.reshape((hi_weight + lo_weight), weight_num // 2)

    # row-major index
    weight = weight + 4
    weight = np.reshape(weight, (M, K // 2)).astype(np.uint8)

    outer_BM_weights = np.split(weight, (M // BM), axis=0)
    for outer_BM_weight in outer_BM_weights:
        # split in col with size of by (32index * 3 == 96nums)
        outer_BY_weights = np.split(outer_BM_weight, (K // BY), axis=1)
        for outer_BY_weight in outer_BY_weights:
            # split in row with size of bm (32)
            inner_bm_weights = np.split(outer_BY_weight, (BM // bm), axis=0)
            for inner_bm_weight in inner_bm_weights:
                # split in col with size of by (2index * 2 == 4nums)
                inner_by_weights = np.split(inner_bm_weight, (BY // by), axis=1)
                for inner_by_weight in inner_by_weights:
                    func_weights = np.split(inner_by_weight, 2, axis=1)

                    left_weight = func_weights[0]
                    left_sub_weights = np.split(left_weight, 4, axis=0)
                    new_left_weight = np.reshape(
                                        np.concatenate([left_sub_weights[0], left_sub_weights[2], 
                                        left_sub_weights[1], left_sub_weights[3]], axis=0, dtype=np.uint8),
                                        (bm))

                    right_weight = func_weights[1]
                    right_sub_weights = np.split(right_weight, 4, axis=0)
                    new_right_weight = np.reshape(
                                        np.concatenate([right_sub_weights[0], right_sub_weights[2], 
                                        right_sub_weights[1], right_sub_weights[3]], axis=0, dtype=np.uint8),
                                        (bm))
                    hi_weight = new_left_weight.astype(np.uint8) << 4
                    lo_weight = new_right_weight
                    func_weight = hi_weight + lo_weight
                    func_weight = np.reshape(func_weight, bm * by // 4)
                    final_weight.append(func_weight)

def preprocess_three_weights(M, K, weight_num, BM, BY, bm, by, weight, final_weight):
    weight = np.reshape(weight, (weight_num // 3, 3))
    split_weights = np.split(weight, 3, axis=1)
    first_weight = np.multiply(split_weights[0], 9)
    second_weight = np.multiply(split_weights[1], 3)
    third_weight = split_weights[2]

    weight = np.reshape((first_weight + second_weight + third_weight), weight_num // 3)
    sign_weight = np.sign(weight) + 2
    sign_weight = np.where(sign_weight > 1, 0, sign_weight)
    weight = np.abs(weight)

    # row-major index
    weight = np.reshape(weight, (M, K // 3)).astype(np.uint8)
    sign_weight = np.reshape(sign_weight, (M, K // 3)).astype(np.uint8)
    # print(weight)

    # split in row with size of BM (160)
    outer_BM_weights = np.split(weight, (M // BM), axis=0)
    for outer_BM_weight in outer_BM_weights:
        # split in col with size of by (32index * 3 == 96nums)
        outer_BY_weights = np.split(outer_BM_weight, (K // BY), axis=1)
        for outer_BY_weight in outer_BY_weights:
            # split in row with size of bm (32)
            inner_bm_weights = np.split(outer_BY_weight, (BM // bm), axis=0)
            for inner_bm_weight in inner_bm_weights:
                # split in col with size of by (2index * 3 == 6nums)
                inner_by_weights = np.split(inner_bm_weight, (BY // by), axis=1)
                for inner_by_weight in inner_by_weights:
                    func_weights = np.split(inner_by_weight, 2, axis=1)

                    left_weight = func_weights[0]
                    left_sub_weights = np.split(left_weight, 4, axis=0)
                    new_left_weight = np.reshape(
                                        np.concatenate([left_sub_weights[0], left_sub_weights[2], 
                                        left_sub_weights[1], left_sub_weights[3]], axis=0, dtype=np.uint8),
                                        (bm))

                    right_weight = func_weights[1]
                    right_sub_weights = np.split(right_weight, 4, axis=0)
                    new_right_weight = np.reshape(
                                        np.concatenate([right_sub_weights[0], right_sub_weights[2], 
                                        right_sub_weights[1], right_sub_weights[3]], axis=0, dtype=np.uint8),
                                        (bm))
                    hi_weight = new_left_weight.astype(np.uint8) << 4
                    lo_weight = new_right_weight
                    # func_weights = np.split(inner_by_weight, 2, axis=0)
                    # top_weight = np.reshape(func_weights[0], (bm3))
                    # low_weight = np.reshape(func_weights[1], (bm3))
                    # hi_weight = top_weight.astype(np.uint8) << 4
                    # lo_weight = low_weight
                    func_weight = hi_weight + lo_weight
                    func_weight = np.reshape(func_weight, bm * by // 6)
                    final_weight.append(func_weight)

    sign_weight_list = []
    sign_outer_BM_weights = np.split(sign_weight, (M // BM), axis=0)
    for sign_outer_BM_weight in sign_outer_BM_weights:
        # split in col with size of by (32index * 3 == 96nums)
        sign_outer_BY_weights = np.split(sign_outer_BM_weight, (K // BY), axis=1)
        for sign_outer_BY_weight in sign_outer_BY_weights:
            # split in row with size of bm (32)
            sign_inner_bm_weights = np.split(sign_outer_BY_weight, (BM // bm), axis=0)
            for sign_inner_bm_weight in sign_inner_bm_weights:
                # split in col with size of by (4index * 3 == 12nums)
                sign_inner_by_weights = np.split(sign_inner_bm_weight, (BY // (by * 4)), axis=1)
                for sign_inner_by_weight in sign_inner_by_weights:
                    func_weight = np.split(sign_inner_by_weight, 8, axis=1)
                    combine_weight = np.zeros((16, 1), dtype=np.uint16)
                    for i in range(len(func_weight)):
                        min_weight = np.split(func_weight[i], 2)
                        min_top_weight = min_weight[0].astype(np.uint16) << 15 - (2 * i)
                        min_bot_weight = min_weight[1].astype(np.uint16) << 15 - (2 * i + 1)
                        combine_weight += min_top_weight
                        combine_weight += min_bot_weight
                    combine_weight = combine_weight.view(np.uint8)
                    # combine_weight = combine_weight[:, [1, 0]]
                    combine_weight = np.reshape(combine_weight, bm)
                    sign_weight_list.append(combine_weight)
    final_weight.extend(sign_weight_list)
    final_weight.extend(sign_weight_list)

M = 4096
K = 8192

w_scale = 0.22
w = np.array(np.random.randint(-1, 2, M * K)).astype(np.float32) * w_scale

w = w.reshape(M, K)
a = np.ones((1, M), dtype=np.float32)
c = np.matmul(a, w)

wdtype = w.dtype
origin_weight = w.astype(np.float32)
weight = origin_weight
scale  = np.max(np.abs(weight))
weight = np.where(np.abs(weight) < 1e-6, 0, weight).astype(wdtype)
weight = np.sign(weight)
weight_num = np.prod(weight.shape)

# for three num 6 bit ->

# outer loop
BM3 = 128
BY3 = 96

# inner loop (32row 32num/16index)
bm3 = 32
by3 = 6

# for two num 4 bit ->

# outer loop
BM2 = 128
BY2 = 32

# inner loop (32row 32num/16index)
bm2 = 32
by2 = 4

if (weight.shape[1] % BY3 != 0):
    slice_k_idx = weight.shape[1] - weight.shape[1] % BY3
    slice_weights = np.split(weight, [slice_k_idx], axis=1)
    three_weight = slice_weights[0]
    two_weight = slice_weights[1]
else:
    three_weight = weight

final_weight = []

preprocess_three_weights(three_weight.shape[0],
                         three_weight.shape[1],
                         three_weight.shape[0] * three_weight.shape[1],
                         BM3,
                         BY3,
                         bm3,
                         by3,
                         three_weight,
                         final_weight)

if (weight.shape[1] % BY3 != 0):
    preprocess_two_weights(  two_weight.shape[0],
                         two_weight.shape[1],
                         two_weight.shape[0] * two_weight.shape[1],
                         BM2,
                         BY2,
                         bm2,
                         by2,
                         two_weight,
                         final_weight)

fout = open("origin_3200_3200_lut3.weight", "wb")
tfout = open("trans_3200_3200_lut3.weight", "wb")

origin_weight.tofile(fout)
np.array(final_weight).astype(np.uint8).tofile(tfout)

fout.close()
tfout.close()