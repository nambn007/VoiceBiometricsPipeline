import torch 
import math 

from speechbrain.processing.features import Filterbank

compute_fbanks = Filterbank()

# inputs = torch.randn([10, 101, 201])
# inputs = torch.zeros(10, 101, 201)
inputs = torch.ones(10, 101, 201)
features = compute_fbanks(inputs)

print(features.shape)
print(features)