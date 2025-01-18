import torch

state_dict = torch.load("C:\\Users\\vaibhav\\Desktop\\VIJAY_SLOWBOT\\ext_data\\models\\vijay2a_MAE_small_model.pth", map_location=torch.device('cuda')) 

for key in state_dict.keys():
    print(key) 