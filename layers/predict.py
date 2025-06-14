import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs = inputs.to(self.args.device)

        inputs = inputs.view(-1, 1024)
        outputs = self.fc(inputs)

        return outputs

