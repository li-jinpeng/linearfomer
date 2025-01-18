import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
import random
import numpy as np
import torch
from data_provider.data_loader import data_provider
from models import Timer

parser = argparse.ArgumentParser(description='transformation matrix for time series forecasting')

parser.add_argument('--ltsm', type=str, default='Timer', help='model name, options: [Timer]')
parser.add_argument('--ckpt', type=str, default='model_files/Timer_forecast_1.0.ckpt')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--data', type=str, default='Exchange', help='dataset type')
# 可选Data： ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ELC', 'Exchange', 'Weather2023', 'Traffic']
parser.add_argument('--features', type=str, default='S')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size of train input data')
parser.add_argument('--gpu', type=int, default=7, help='gpu')
parser.add_argument('--input_len', type=int, default=96*7, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--k', type=int, default=99, help='Number of replicas for Ensemble forecasting')
parser.add_argument('--l', type=float, default=0.0125, help='The range of values of the transformation matrix')
parser.add_argument('--std', type=float, default=0.03, help='White noise variance')

args = parser.parse_args()
print('=================')
print(args)
print('=================')

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

device = torch.device(f'cuda:{args.gpu}')

ltsm = Timer.Model(args.ckpt).to(device)
ltsm.eval()

print('LTSM Model parameters: ', sum(param.numel() for param in ltsm.parameters()))

input_len = args.input_len
pred_len = args.pred_len

data_set, data_loader = data_provider(args.batch_size, args.data, 'all', input_len, pred_len)

def MSE(pred, true):
    return torch.mean((pred-true)**2)

zero_shot_mse = torch.tensor(0.0, dtype=float).to(device) 
transformation_matrix_mse = torch.tensor(0.0, dtype=float).to(device) 
noise_injection_mse = torch.tensor(0.0, dtype=float).to(device)
samples_num = torch.tensor(len(data_set), dtype=int).to(device)

k = args.k
# 生成k个随机变换矩阵
w_list = []
for i in range(k):
    w = torch.rand((input_len, input_len)).float().to(device) # 值的范围：[0, 1] 
    w = w - 0.5 # 值的范围：[-0.5, 0.5]
    w = w / (0.5 / args.l) # 值的范围：[-args.l, args.l]
    w = w + torch.eye(input_len,dtype=torch.float32,device=device) # 加上对角阵
    w_list.append(w)
# 生成k个随机白噪音
noise_list = []  
for i in range(k):
    m = torch.randn(input_len).float().to(device) # std = 1 噪音标准差
    m /= (1 / args.std) # std = args.std
    noise_list.append(m)

with torch.no_grad():
    for i, (batch_x, batch_y) in enumerate(data_loader):
        # k+1个预测结果,k+1是因为包含不做处理本身
        transformation_matrix_predictions = torch.zeros((k + 1, batch_x.shape[0], pred_len)).to(device)
        noise_injection_predictions = torch.zeros((k + 1, batch_x.shape[0], pred_len)).to(device)
        batch_y = torch.squeeze(batch_y).float().to(device)
        batch_x = batch_x.float().to(device)
        for j in range(k + 1):
            if j == 0:
                # j=0 不做处理直接预测
                y = ltsm(batch_x)[:, -pred_len:, :]
                y = torch.squeeze(y)
                transformation_matrix_predictions[j] = y
                noise_injection_predictions[j] = y
                zero_shot_mse_item = MSE(y, batch_y)
            else:
                # 白噪音
                # 将噪音处理为和batch_x形状一致
                noise = torch.unsqueeze(noise_list[j - 1], 0)
                noise = noise.repeat(batch_x.shape[0], 1)
                noise = torch.unsqueeze(m, -1)
                # 得到注入噪音的x
                noise_x = batch_x + noise
                noise_y = ltsm(noise_x)[:, -pred_len:, :]
                noise_y = torch.squeeze(noise_y)
                noise_injection_predictions[j] = noise_y 
                
                # 线性变换
                w = w_list[j - 1]
                transformation_x = batch_x.permute(0, 2, 1) 
                transformation_x = torch.matmul(transformation_x, w)
                transformation_x = transformation_x.permute(0, 2, 1)
                transformation_y = ltsm(transformation_x)[:, -pred_len:, :]
                transformation_y = torch.cat((transformation_x[:, -(input_len - pred_len):, :], transformation_y), dim=1)
                transformation_y = transformation_y.permute(0, 2, 1)
                # 逆矩阵
                u = torch.linalg.inv(w)
                transformation_y = torch.matmul(transformation_y, u)
                transformation_y = transformation_y.permute(0, 2, 1)
                transformation_y = transformation_y[:, -pred_len:, :]
                transformation_y = torch.squeeze(transformation_y)
                transformation_matrix_predictions[j] = transformation_y
        
        # 取中位数
        mid_transformation_y = transformation_matrix_predictions.quantile(q=0.5,dim=0)
        mid_noise_y = noise_injection_predictions.quantile(q=0.5,dim=0)
        
        # 计算MSE
        transformation_matrix_mse_item = MSE(mid_transformation_y, batch_y)
        noise_injection_mse_item = MSE(mid_noise_y, batch_y)

        # 汇总MSE
        zero_shot_mse += (zero_shot_mse_item * batch_x.shape[0] / samples_num)
        transformation_matrix_mse += (transformation_matrix_mse_item * batch_x.shape[0] / samples_num)
        noise_injection_mse += (noise_injection_mse_item * batch_x.shape[0] / samples_num)
 
        # 输出
        print(i, len(data_loader), zero_shot_mse_item, noise_injection_mse_item, transformation_matrix_mse_item)
        torch.cuda.empty_cache()

print(f'data: {args.data} | pred_len: {args.pred_len} k: {args.k} l: {args.l} std: {args.std} | zero-shot: {zero_shot_mse} | noise-mse: {noise_injection_mse} | matrix-mse: {transformation_matrix_mse}')
