import torch

from transformers import AutoModelForCausalLM
from chronos import BaseChronosPipeline, ChronosPipeline
from models import Timer

all_model = ['timer-UTSD', 'timer-LOTSA', 'chronos-bolt-tiny', 'chronos-bolt-base', 'timer-base-84M']
model_path_dict = {
    'timer-UTSD': 'model_files/Timer_UTSD.ckpt',
    'timer-LOTSA': 'model_files/Timer_LOTSA.ckpt',
    'chronos-bolt-tiny': 'model_files/chronos-bolt-tiny',
    'chronos-bolt-base': 'model_files/chronos-bolt-base',
    'timer-base-84M': 'model_files/timer-base-84m'
}

class Model:
    def __init__(self, model_name, device):
        
        self.model_name = model_name
        if model_name == 'timer-base-84M':
            self.model = AutoModelForCausalLM.from_pretrained(model_path_dict['timer-base-84M'], trust_remote_code=True).to(device)
        elif model_name == 'chronos-bolt-tiny':
            self.pipeline = BaseChronosPipeline.from_pretrained(
                model_path_dict['chronos-bolt-tiny'],
                device_map=device,
                torch_dtype=torch.float32,
            )
            self.pipeline.model.eval()
        elif model_name == 'chronos-bolt-base':
            self.pipeline = BaseChronosPipeline.from_pretrained(
                model_path_dict['chronos-bolt-base'],
                device_map=device,
                torch_dtype=torch.float32,
            )
            self.pipeline.model.eval()
        elif model_name == 'chronos-t5-small':
            self.pipeline = ChronosPipeline.from_pretrained(
                model_path_dict['chronos-t5-small'],
                device_map=device,
                torch_dtype=torch.float32,
            )
            self.pipeline.model.eval()
        else:
            self.model = Timer.Model(model_path_dict[model_name]).to(device)
            self.model.eval()
            
    def __call__(self, data, pred_len):
        if self.model_name == 'timer-base-84M':
            with torch.no_grad():
                preds = self.model.generate(data, max_new_tokens=pred_len)
        elif 'chronos' in self.model_name:
            with torch.no_grad():
                _, preds = self.pipeline.predict_quantiles(
                    context=data,
                    prediction_length=pred_len,
                    limit_prediction_length=False,
                )
        else:
            data = torch.unsqueeze(data, -1)
            preds = self.model(data)[:, -pred_len:, :]
            preds = torch.squeeze(preds, -1)
        return preds
