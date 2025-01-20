datas = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ELC', 'Exchange', 'Weather2023', 'Traffic']
models = ['timer-UTSD', 'timer-LOTSA', 'chronos-bolt-tiny', 'chronos-bolt-base', 'timer-base-84M']
pred_len = [24, 48, 96, 192, 336, 720]
stds = [0, 0.01, 0.02, 0.05, 0.08]
ls = [0, 0.01, 0.02, 0.05, 0.08]
import os
import sys
data_index = int(sys.argv[1])
gpu = 4 + data_index // 4
data = datas[data_index]
for model in models:
    for p_l in pred_len:
        for std in stds:
            for l in ls:
                if std == 0 and l == 0:
                    continue
                script = f'python -u run_v2.py \
                        --data {data} \
                        --model {model} \
                        --pred_len {p_l} \
                        --gpu {gpu} \
                        --std {std} \
                        --l {l}'
                os.system(script)