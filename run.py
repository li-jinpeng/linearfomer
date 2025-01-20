import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
import random
import numpy as np
import torch
import json

parser = argparse.ArgumentParser(description='transformation matrix for time series forecasting')

parser.add_argument('--model', type=str, default='chronos-bolt-base', help='model name')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
# 可选Data： ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ELC', 'Exchange', 'Weather2023', 'Traffic']
parser.add_argument('--features', type=str, default='S')
parser.add_argument('--gpu', type=int, default=4, help='gpu')
parser.add_argument('--input_len', type=int, default=96*15, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--k', type=int, default=100, help='Number of replicas for Ensemble forecasting')
parser.add_argument('--l', type=float, default=0.01, help='The range of values of the transformation matrix')
parser.add_argument('--b', type=float, default=0.05, help='White noise variance')

args = parser.parse_args()
print('=================')
print(args)
print('=================')

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

device = torch.device(f'cuda:{args.gpu}')

input_len = args.input_len
pred_len = args.pred_len

from _model import Model
model = Model(args.model, device)
from _dataset import Dataset
data_set = Dataset(args.data, args.input_len, args.pred_len)

def MSE(pred, true):
    return torch.mean((pred-true)**2)

zero_shot_mse = torch.tensor(0.0, dtype=float).to(device) 
mid_mse = torch.tensor(0.0, dtype=float).to(device) 
avg_mse = torch.tensor(0.0, dtype=float).to(device)
top_mse = torch.tensor(0.0, dtype=float).to(device)
samples_num = torch.tensor(len(data_set), dtype=int).to(device)

k = args.k
b = args.b
# 生成k个随机变换矩阵
# k个矩阵
w_list = torch.zeros((k, input_len, input_len), dtype=torch.float32).to(device)
# k个逆矩阵
u_list = torch.zeros((k, input_len, input_len), dtype=torch.float32).to(device)
# k个bias
b_list = torch.zeros((k, input_len), dtype=torch.float32).to(device)
for i in range(k):
    if i == 0:
        w = torch.eye(input_len,dtype=torch.float32,device=device)
        bias = torch.zeros(input_len,dtype=torch.float32,device=device)
    else:
        if args.l != 0:
            w = torch.rand((input_len, input_len)).to(torch.float32).to(device) # 值的范围：[0, 1] 
            w = w - 0.5 # 值的范围：[-0.5, 0.5]
            w = w / (0.5 / args.l) # 值的范围：[-args.l, args.l]
            w = w + torch.eye(input_len,dtype=torch.float32,device=device) # 加上对角阵
        else:
            w = torch.eye(input_len,dtype=torch.float32,device=device)
        if args.b != 0:
            bias = torch.rand(input_len).to(torch.float32).to(device)
            bias = bias - 0.5
            bias = bias / (0.5 / b)
        else:
            bias = torch.zeros(input_len,dtype=torch.float32,device=device)
    w_list[i,:,:] = w
    u = torch.linalg.inv(w)
    u_list[i,:,:] = u
    u_list[i,:,:] = w
    
    b_list[i] = bias


with torch.no_grad():
    for i in range(0, len(data_set)):
        batch_x, batch_y = data_set[i]
        batch_y = torch.tensor(batch_y).to(torch.float32).to(device)
        batch_x = torch.tensor(batch_x).to(torch.float32).to(device)
        
        # 数据变换 x' = wx + b
        batch_x = torch.unsqueeze(batch_x, 0)
        batch_x = batch_x.repeat(k, 1)
        batch_x = torch.unsqueeze(batch_x, -1)
        batch_x = torch.matmul(w_list, batch_x)
        batch_x = torch.squeeze(batch_x, -1)
        batch_x = batch_x + b_list
        # 预测
        preds = model(batch_x, pred_len)
        preds = preds.to(device)
        # 逆变换 y = w^-1(y' - b)
        preds = torch.cat((batch_x[:, -(input_len - pred_len):], preds), dim=1)
        batch_x = batch_x - b_list
        preds = torch.unsqueeze(preds, -1)
        preds = torch.matmul(w_list, preds)
        preds = torch.squeeze(preds, -1)
        preds = preds[:, -pred_len:]
        
        # 取中位数 
        mid_pred = preds.quantile(q=0.5,dim=0)
        # 取均值
        avg_pred = torch.mean(preds, dim=0)
        # 取距离其他所有点最近的点
        top_pred = torch.zeros(pred_len, dtype=torch.float32, device=device)
        tmp_pred = torch.zeros((k, pred_len), dtype=torch.float32, device=device)
        for j in range(k):
            tmp = preds[j, :]
            tmp = torch.unsqueeze(tmp, 0)
            tmp = tmp.repeat(k, 1)
            tmp = tmp - preds
            tmp = tmp ** 2
            tmp = torch.sum(tmp, dim=0)
            tmp_pred[j] = tmp
        tmp_pred = torch.argmin(tmp_pred, dim=0)
        for p in range(pred_len):
            top_pred[p] = preds[tmp_pred[p], p]      
        
        # 计算MSE
        mid_mse_item = MSE(mid_pred, batch_y)
        avg_mse_item = MSE(avg_pred, batch_y)
        top_mse_item = MSE(top_pred, batch_y)
        zero_shot_mse_item = MSE(preds[0], batch_y)
        
        # 汇总MSE
        zero_shot_mse += (zero_shot_mse_item / samples_num)
        mid_mse += (mid_mse_item / samples_num)
        avg_mse += (avg_mse_item / samples_num)
        top_mse += (top_mse_item / samples_num)
 
        torch.cuda.empty_cache()

result = json.load(open(f'results/{args.data}.json', 'r'))
if args.model not in result:
    result[args.model] = {}
if str(args.pred_len) not in result[args.model]:
    result[args.model][str(args.pred_len)] = []
item = {
    'w': args.l,
    'b': args.b,
    'zero-shot': zero_shot_mse.item(),
    'mid': mid_mse.item(),
    'avg': avg_mse.item(),
    'distance': top_mse.item()
}
result[args.model][str(args.pred_len)].append(item)
json.dump(result, open(f'results/{args.data}.json', 'w'), indent=4)
print(f'data: {args.data} | pred_len: {args.pred_len} l: {args.l} b: {args.b} | zero-shot: {zero_shot_mse} | mid-mse: {mid_mse} | avg-mse: {avg_mse} | distance-mse: {top_mse}')
