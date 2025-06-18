import torch.nn as nn

# 设计神经网络
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # 卷积层1：输入通道1，输出通道16，卷积核3x3，步长1，padding 1
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        # 批归一化，作用于16通道
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        # 卷积层2：输入16，输出32
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.relu2 = nn.ReLU()
        # 最大池化层1：2x2窗口，步长2
        self.pool1 = nn.MaxPool2d(2, 2)
        # 卷积层3：输入32，输出64
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        # 批归一化，作用于64通道
        self.bn2 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        # 卷积层4：输入64，输出128
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu4 = nn.ReLU()
        # 最大池化层2：2x2窗口，步长2
        self.pool2 = nn.MaxPool2d(2, 2)
        # 卷积层5：输入128，输出256
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.relu5 = nn.ReLU()
        # Dropout层，概率0.25
        self.dropout = nn.Dropout(0.25)

        # 全连接层1，输入256*7*7，输出512
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.relu6 = nn.ReLU()
        # 全连接层2，输入512，输出256
        self.fc2 = nn.Linear(512, 256)
        self.relu7 = nn.ReLU()
        # 全连接层3，输入256，输出128
        self.fc3 = nn.Linear(256, 128)
        self.relu8 = nn.ReLU()
        # 全连接层4，输入128，输出10（分类数）
        self.fc4 = nn.Linear(128, 10)

        # 用于特征提取的钩子，便于中间特征可视化/分析
        self.conv_features = {}
        self.fc_features = {}
        self.conv2.register_forward_hook(self.get_conv_features('conv2'))
        self.fc2.register_forward_hook(self.get_fc_features('fc2'))
        self.fc4.register_forward_hook(self.get_fc_features('fc4'))

        # 统计网络层总数，方便后续引用
        self.layer_count = 17

    def get_conv_features(self, name):
        # 卷积层特征钩子函数
        def hook(model, input, output):
            self.conv_features[name] = output.detach()
        return hook

    def get_fc_features(self, name):
        # 全连接层特征钩子函数
        def hook(model, input, output):
            self.fc_features[name] = output.detach()
        return hook

    def forward(self, x):
        # 前向传播过程
        x = self.relu1(self.bn1(self.conv1(x)))     # conv1 -> bn1 -> relu1
        x = self.relu2(self.conv2(x))               # conv2 -> relu2
        x = self.pool1(x)                           # pool1
        x = self.relu3(self.bn2(self.conv3(x)) )    # conv3 -> bn2 -> relu3
        x = self.relu4(self.conv4(x))               # conv4 -> relu4
        x = self.pool2(x)                           # pool2
        x = self.relu5(self.dropout(self.conv5(x))) # conv5 -> dropout -> relu5
        # 展平为一维（适配全连接输入）
        x = x.view(-1, 256 * 7 * 7)
        x = self.relu6(self.fc1(x))                 # fc1 -> relu6
        x = self.relu7(self.fc2(x))                 # fc2 -> relu7
        x = self.relu8(self.fc3(x))                 # fc3 -> relu8
        x = self.fc4(x)                             # fc4
        return x

# 统计模型可训练参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)