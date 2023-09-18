import torch


class Config:
    def __init__(self):
        
        # sigma for l1_smooth_loss
        self.rpn_sigma = 3.
        self.roi_sigma = 1.

        # visualization
        self.env = 'faster-rcnn'  # visdom env
        self.port = 8097
        self.plot_every = 40  # vis every N iter

        """ training epoches """
        # self.epoch = 100 
        self.iter = 1000
        self.interval = 1


        """ learning rate """
        self.lr = 1e-5

        """ GPU or CPU ? """
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


        """ pretrain ? """
        self.load_path = None


        """ dataloader """
        self.num_workers = 4
        self.test_num_workers = 4

    def get_opt(self):
        return self