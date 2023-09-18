import torch
from model.faster_rcnn_da import DANN


# configurations
class Config:
    def __init__(self):
        self.load_path = False
    
    def get_opt(self):
        return self 

opt = Config().get_opt()

fake_input = torch.rand((2,3,256,256),dtype=torch.float32)

net = DANN()

print(net(fake_input).shape)

