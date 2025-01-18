import os

datas = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ELC', 'Exchange', 'Weather2023', 'Traffic']
gpu_list = [0,2,3,4,5,6,7]
input_len = 96*7
# pred_lens = [96, 192, 24, 48]
pred_lens = [288, 384, 480, 576]
# ks = [100, 200, 500]
ks = [100]
ls = [0.0125]
# ls = [0.0025]
# datas = ['ELC', 'Traffic']
stds = [0.03]
# ls = [0.0125, 0.01, 0.015, 0.0175, 0.02, 0.0075]
# stds = [0.02, 0.03, 0.05]

# data = 'ETTh1'
# pred_len = 96
gpu_index = 0
# index = 0
# 0-5
print(len(datas) * len(pred_lens) * len(ks) * len(stds) * len(ls))

for data in datas:
    for pred_len in pred_lens:
        for std in stds:
            for k in ks:
                for l in ls:
                    script = f'nohup python -u run.py \
                        --data {data} \
                        --pred_len {pred_len} \
                        --gpu {gpu_list[gpu_index]} \
                        --k {k} \
                        --std {std} \
                        --l {l} > logs/{data}_{pred_len}.log 2>&1 &'
                    os.system(script)
                    gpu_index += 1
                    gpu_index %= len(gpu_list)
