import torch
import torch.nn as nn

# 设计神经网络
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # 独特结构：共 17 层（算激活/池化/全连接/卷积/reshape）
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        # 全连接层
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu7 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu8 = nn.ReLU()
        self.fc4 = nn.Linear(128, 10)

        # 用于特征提取的钩子
        self.conv_features = {}
        self.fc_features = {}

        self.conv2.register_forward_hook(self.get_conv_features('conv2'))
        self.fc2.register_forward_hook(self.get_fc_features('fc2'))
        self.fc4.register_forward_hook(self.get_fc_features('fc4'))

        # 统计层数
        self.layer_count = 17

    def get_conv_features(self, name):
        def hook(model, input, output):
            self.conv_features[name] = output.detach()
        return hook

    def get_fc_features(self, name):
        def hook(model, input, output):
            self.fc_features[name] = output.detach()
        return hook

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.relu3(self.bn2(self.conv3(x)))
        x = self.relu4(self.conv4(x))
        x = self.pool2(x)
        x = self.relu5(self.dropout(self.conv5(x)))
        x = x.view(-1, 256 * 7 * 7)
        x = self.relu6(self.fc1(x))
        x = self.relu7(self.fc2(x))
        x = self.relu8(self.fc3(x))
        x = self.fc4(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)