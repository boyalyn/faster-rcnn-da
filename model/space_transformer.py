import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.set_default_datatype(torch.float32)


class Deformer(nn.Module):
    def __init__(self):
        super(Deformer, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=4),
            # nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        
        return x


if __name__ == "__main__":
    model = Deformer()
    input_tensor = torch.rand(1,1,7,7)
    print(model(input_tensor).shape)