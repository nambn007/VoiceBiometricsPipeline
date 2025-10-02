import torch 
from speechbrain.lobes.features import Fbank

tensor_4096 = torch.full((1,4096), 0.5)
print(tensor_4096)

fbank = Fbank(context=False, sample_rate=16000, n_mels=80)
output = fbank(tensor_4096)

print(output.shape)
print(output[0][1])