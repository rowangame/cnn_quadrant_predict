
# CNN 入门理解篇
# 我们已知四个数据点(1,1)(-1,1)(-1,-1)(1,-1)，这四个点分别对应I~IV象限，如果这时候给我们一个新的坐标点（比如(2,2)），
# 那么它应该属于哪个象限呢？请用神经网络分类处理
# see: https://zhuanlan.zhihu.com/p/65472471
# author: 负熵笔记

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 要保存的模型路径
model_path = "./model/quadrant.pt"

# 构建神经网络模型
class QuadrantClassifier(nn.Module):
    def __init__(self):
        super(QuadrantClassifier, self).__init__()

        # 输入层(2个神经元，对应x和y的值),隐藏层(32个神经元,使用ReLU激活函数）
        self.fc1 = nn.Linear(2, 32)
        # 输入层:隐藏层(32个神经元),输出层:隐藏层(16个神经元,使用ReLU激活函数)
        self.fc2 = nn.Linear(32, 16)
        # 输入层:隐藏层(16个神经元),输出层:(7个神经元,对应第0,1,2,3象限,x-轴,y-轴,原点共7个分类)
        self.fc3 = nn.Linear(16, 7)

        # 解释为何要有隐藏层(参考):
        # 在神经网络中，隐藏层的神经元数量是一个超参数，需要根据具体任务和数据集的复杂性来调整。
        # 选择隐藏层神经元的数量的目标之一是使网络能够捕捉输入数据的复杂特征，以便在训练过程中学习更好的表示。
        #
        # 在这个例子中，选择了隐藏层神经元，主要是基于一些经验和试验。这个数量并不是固定的，可以根据实际情况进行调整。
        # 较多的隐藏层神经元可以使网络具有更大的容量，有助于学习更复杂的模式和特征，但也可能导致过拟合。
        # 较少的隐藏层神经元可能导致网络无法捕捉输入数据中的一些复杂模式。
        #
        # 在实践中，选择隐藏层神经元数量通常是一个需要调整的超参数，可以通过交叉验证等方法来优化。
        # 过程中可以尝试不同的神经元数量，并观察模型在验证集上的性能，从而选择最佳的配置。
        # 需要注意的是，隐藏层神经元的数量并不是越多越好，因为增加网络容量可能会增加训练时间和资源的需求，
        # 同时也可能引入过拟合问题。合适的网络结构和参数调整是深度学习中一个重要的实践性问题

    # 神经网络每次调用时都需要执行的前向传播计算(使用ReLU激活函数)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 查看模型参数数据
def show_mode_params(model, tag):
    show_logs = False
    if not show_logs:
        return
    print(tag)
    for tmpP in model.parameters():
        print(tmpP)

# 检查tx,ty坐标象限分类
def check_quadrant(tx, ty):
    if (tx > 0) and (ty > 0):
        return 0  # 第一象限
    if (tx < 0) and (ty > 0):
        return 1  # 第二象限
    if (tx < 0) and (ty < 0):
        return 2  # 第三象限
    if (tx > 0) and (ty < 0):
        return 3  # 第四象限
    # 浮点数比较
    if (abs(tx) > 0) and (abs(ty) < 0.000001):
        return 4  # x轴
    # 浮点数比较
    if (abs(tx) < 0.000001) and (abs(ty) > 0):
        return 5  # y轴
    return 6  # 原点

# 随机整数
def random_int():
    low = -2147483648
    high = 2147483647
    return np.random.randint(low, high)

# 生成训练数据
def generate_traindata():
    # # 原始训练数据(只处理第0,1,2,3象限内的值)
    # # 输入数据(仅四组值)
    # data = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float32)
    # # 对应的象限标签
    # labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    # return data, labels

    # 新训练数据(第0,1,2,3象限内的值和x轴,y轴,原点的值)
    # 这里取样要根据原点,x轴,y轴,象限内的数据占比(就是每个分类都要有样本数据)
    max_cnt = 350
    data = np.zeros((max_cnt, 2), dtype=np.float32)
    labels = np.zeros(max_cnt, dtype=np.int64)
    for i in range(max_cnt):
        tx = np.random.randint(-5, 6)
        ty = np.random.randint(-5, 6)
        data[i,0:2] = [tx, ty]
        labels[i] = check_quadrant(tx, ty)
    return torch.from_numpy(data), torch.from_numpy(labels)

def train_process():
    data,labels = generate_traindata()
    print("train_process:")
    print("data.shape", data.shape)
    print("labels.shape", labels.shape)

    # 创建模型
    model = QuadrantClassifier()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器(采用Adam算法 see: https://blog.csdn.net/kgzhang/article/details/77479737)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 查看模型参数
    show_mode_params(model, "-"*30 + "first params:")
    # 根据训练模型(初始化层(3个层对象)的数据类型,模型参数有3组:)
    # Layer1: W1(32 x 2)权重参数  + b1(1 x 32)(偏置值,默认存在。若bias=False,则该网络层无偏置,图层不会学习附加偏差)
    # Layer2: W2(16 x 32)权重参数 + b2(1 x 16)
    # Layer3: W3(7 x 16)权重参数  + b3(1 x 7)

    # 将数据转换为 DataLoader
    dataset = TensorDataset(data, labels)
    # shuffle为True 每次训练时,数据加载器会扰乱数据的先后顺序
    # batch_size: 数据加载器每一次要读取多少个原数据(用于训练)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 训练模型(次数)
    epochs = 1000
    for epoch in range(epochs):
        # 查看模型参数(训练前目的:神经元网络在反向传播后,模型参数的调优情况)
        show_mode_params(model, "-" * 30 + f"epoch={epoch}-> start:")
        for inputs, targets in dataloader:
            # optimizer.zero_grad()函数会遍历模型的所有参数，通过p.grad.detach_()方法截断反向传播的梯度流，
            # 再通过p.grad.zero_()函数将每个参数的梯度值设为0，即上一次的梯度记录被清空。
            optimizer.zero_grad()
            # 模型输出
            outputs = model(inputs)
            # 交叉熵损失
            loss = criterion(outputs, targets)
            # 反向传播求梯度
            loss.backward()
            # 更新所有参数
            optimizer.step()
        # 查看横型参数(一次训练后,loss值应该越来越小,才符合模型训练的结果)
        show_mode_params(model, "-" * 30 + f"epoch={epoch}-> end:")

    # Pytorch如何保存训练好的模型
    # see: https://blog.csdn.net/comli_cn/article/details/107516740
    # (1）只保存模型参数字典（推荐）
    torch.save(model.state_dict(), model_path)
    # (2) 保存整个模型
    # torch.save(model, model_path)

# 加载模型
def loadModel():
    # 读取(模型参数字典)
    qmodel = QuadrantClassifier()
    qmodel.load_state_dict(torch.load(model_path))

    # 读取(整个模型)
    #qmodel = torch.load(model_path)
    return qmodel

# 测试模型预测的正确率
def test_prediction():
    infos = ["I", "II", "III", "IV", "X-axis", "Y-axis", "Origin"]
    model = loadModel()
    low = -2147483648
    high = 2147483647
    total = 0
    sucCnt = 0
    for p in range(1000):
        total += 1
        tx = np.random.randint(low, high)
        ty = np.random.randint(low, high)
        # 预测新的坐标点
        new_point = torch.tensor([[tx, ty]], dtype=torch.float32)

        # with torch.no_grad的作用:
        # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False
        #
        # see: https://blog.csdn.net/sazass/article/details/116668755
        # 在pytorch中，tensor有一个requires_grad参数，如果设置为True，则反向传播时，该tensor就会自动求导。
        # tensor的requires_grad的属性默认为False,若一个节点（叶子变量：自己创建的tensor）requires_grad被设置为True，
        # 那么所有依赖它的节点requires_grad都为True（即使其他相依赖的tensor的requires_grad = False）
        with torch.no_grad():
            # model.eval函数理解:
            # 不启用BatchNormalization和Dropout，保证BN和dropout不发生变化，
            # pytorch框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值，
            # 不然的话，一旦test的batch_size过小，很容易就会被BN层影响结果。
            # see: https://blog.csdn.net/sazass/article/details/116616664
            model.eval()

            # torch.argmax函数理解:
            # see: https://blog.csdn.net/weixin_42494287/article/details/92797061
            # 使用model(new_point)预测结果类型,然后通过torch.argmax得到分类最接近的值
            prediction = torch.argmax(model(new_point), dim=1).item()

        # 真实结果
        realValue = check_quadrant(tx, ty)
        """
        模型泛化(参考):
        在机器学习中，模型的泛化（generalization）指的是模型在未见过的新数据上的表现能力。
        一个好的泛化模型能够很好地适应训练集之外的数据，而不仅仅是在训练集上表现良好。
        模型泛化的目标是学到数据的潜在规律，而不仅仅是记住训练数据的具体样本。
        模型泛化的重要性在于，我们通常关注的是模型在现实场景中的应用，而不仅仅是在训练数据上的性能。
        如果一个模型只是在训练数据上表现良好但在新数据上表现糟糕，那么它对于实际问题的解决是不可靠的。
        有几个因素影响模型的泛化能力：
        模型的复杂度： 如果模型过于复杂，它可能会在训练数据上过拟合，即学到了训练数据中的噪声和细节，而不是真正的模式。
        这会导致在新数据上表现不佳。因此，一个良好的泛化模型需要适度的复杂度，足以捕捉数据的规律，但不至于过分记忆训练数据。
        数据的质量和多样性： 充足、高质量、多样性的训练数据有助于模型更好地学到数据的潜在规律，从而提高泛化能力。
        正则化： 正则化技术用于约束模型参数的大小，防止模型过度拟合训练数据。常见的正则化方法包括L1正则化和L2正则化。
        避免过拟合： 通过在训练过程中使用验证集进行监控，并在模型在验证集上表现达到最优时停止训练，可以避免模型过拟合。
        在卷积神经网络（CNN）中，以上的概念同样适用。通过合适的模型结构、正则化技术和充足的数据，可以提高CNN在新数据上的泛化性能。
        """
        # 验证状态,成功状态就累加(评估模型的泛化能力)
        success = prediction == realValue
        if success:
            sucCnt += 1
        print(f'{tx,ty} is {infos[realValue]}, predict is {infos[prediction]}, state={success}')
    print(f'success ratio:{1.0 * sucCnt / total}')

# 训练模型
train_process()

# 模型预测测试
# test_prediction()