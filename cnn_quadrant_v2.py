"""
CNN 参数调优
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# 数据生成器
class DataGenerator(object):
    def __init__(self):
        super(DataGenerator, self).__init__()
        self.trainData = []
        self.trainLabels = []
        self.valData = []
        self.valLabels = []

    # 检查tx,ty坐标象限分类
    def check_quadrant(self, tx, ty):
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
    def random_int(self, low, high):
        return np.random.randint(low, high)

    # 随机生成训练和验证数据
    # train(low, high, count)
    # val(low, high, count)
    def generateData(self, train: tuple[int, int, int], val: tuple[int, int, int]):
        self.trainData.clear()
        self.trainLabels.clear()
        self.valData.clear()
        self.valLabels.clear()
        low = train[0]
        high = train[1]
        count = train[2]
        for i in range(count):
            tx = self.random_int(low, high)
            ty = self.random_int(low, high)
            tl = self.check_quadrant(tx, ty)
            self.trainData.append([tx, ty])
            self.trainLabels.append(tl)

        low = val[0]
        high = val[1]
        count = val[2]
        for i in range(count):
            tx = self.random_int(low, high)
            ty = self.random_int(low, high)
            tl = self.check_quadrant(tx, ty)
            self.valData.append([tx, ty])
            self.valLabels.append(tl)

    # 转化为Numpy数据格式
    def toNumpyObj(self):
        return np.array(self.trainData, dtype=np.float32), np.array(self.trainLabels, dtype=np.int64), \
               np.array(self.valData, dtype=np.float32), np.array(self.valLabels, dtype=np.int64)

    # 转化为张量对象
    def toTensorObj(self):
        td, tl, vd, vl = self.toNumpyObj()
        return torch.from_numpy(td), torch.from_numpy(tl), torch.from_numpy(vd), torch.from_numpy(vl)


# 构建神经网络模型
class QuadrantClassifier(nn.Module):
    def __init__(self):
        super(QuadrantClassifier, self).__init__()
        # 输入层,中间层,输出层列表
        self.mLayers = []

    # 设置输入层,中间层,输出层(用于层数量的多少对结果性能的比较)
    # 输入层,中间层,输出层特别数要一一对应起来(否则会出错)
    # 至少要有两层(一个输入层,一个输出层)
    def setLayers(self, layers: list[tuple[int, int]]):
        self.mLayers.clear()
        layerCnt = 0
        for tmpLL in layers:
            layerCnt += 1
            tmpLayer = nn.Linear(tmpLL[0], tmpLL[1])
            # 这里需要动态设置性值(否则创建优化器时,会报模型的参数为空,不知是何原因)
            setattr(self, "layer_%d" % layerCnt, tmpLayer)
            self.mLayers.append(tmpLayer)

    # 神经网络每次调用时都需要执行的前向传播计算(使用ReLU激活函数)
    def forward(self, x):
        layerSize = len(self.mLayers)

        # 输入层和中间层都采用ReLU激活函数
        for tl in range(layerSize - 1):
            x = torch.relu(self.mLayers[tl](x))
        # 最后一层为输出层(输出层不使用激活函数)
        x = self.mLayers[layerSize - 1](x)

        return x


# 模型处理类
class ModelProcessor(object):
    def __init__(self):
        super(ModelProcessor, self).__init__()
        # 训练数据和验证数据
        self.mTrainData = None
        self.mTrainLabels = None
        self.mValData = None
        self.mValLabels= None

        # 模型对象
        self.mQmodel = None
        # 输入层,中间层,输出层列表
        self.mLayers = None

        # 损失函数
        self.mCriterion = None
        # 优化器
        self.mOptimizer = None
        # 数据加载器
        self.mDataloader = None
        # 学习率
        self.mLr = 0.001
        # 训练次数
        self.mEpochs = 100

        # 模型保存的路径
        self.mQmodelPath = "./model/quadrant_v2.pt"
        # 模型的准确率
        self.mSucRatio = 0.0
        # 训练所花费的总时间(s)
        self.mTrainTime = 0.0

    def setTrainValData(self, trainData, trainLabels, valData, valLabels):
        self.mTrainData = trainData
        self.mTrainLabels = trainLabels
        self.mValData = valData
        self.mValLabels = valLabels

    # 设置输入层,中间层,输出层(用于层数量的多少对结果性能的比较)
    # 输入层,中间层,输出层特别数要一一对应起来(否则会出错)
    # 至少要有两层(一个输入层,一个输出层)
    def setLayers(self, layers: list[tuple[int, int]]):
        self.mLayers = layers

    def setEpochs(self, epochs: int):
        self.mEpochs = epochs

    def setLr(self, lr: float = 0.001):
        self.mLr = lr

    def setQmodelSavePath(self, path):
        self.mQmodelPath = path

    def setTrainTime(self, sparedTime):
        self.mTrainTime = sparedTime

    # 查看模型参数数据
    def show_mode_params(self, model, tag):
        show_logs = False
        if not show_logs:
            return
        print(tag)
        for tmpP in model.parameters():
            print(tmpP)

    # 执行训练逻辑
    def doTrainProcess(self):
        print("doTrainProcess:")
        print("trainData.shape", self.mTrainData.shape)
        print("trainLabels.shape", self.mTrainLabels.shape)

        # 创建模型
        self.mQmodel = QuadrantClassifier()
        # 设置层(设置输入层,中间层,输出层)
        self.mQmodel.setLayers(self.mLayers)

        # 损失函数
        self.mCriterion = nn.CrossEntropyLoss()
        # 优化器(采用Adam算法)
        self.mOptimizer = optim.Adam(self.mQmodel.parameters(), lr=self.mLr)
        # 查看模型参数
        self.show_mode_params(self.mQmodel, "-"*30 + "first params:")

        # 将数据转换为 DataLoader
        dataset = TensorDataset(self.mTrainData, self.mTrainLabels)
        # shuffle为True 每次训练时,数据加载器会扰乱数据的先后顺序
        # batch_size: 数据加载器每一次要读取多少个原数据(用于训练)
        self.mDataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # 训练模型
        for epoch in range(self.mEpochs):
            # 查看模型参数(训练前目的:神经元网络在反向传播后,模型参数的调优情况)
            self.show_mode_params(self.mQmodel, "-" * 30 + f"epoch={epoch}-> start:")
            for inputs, targets in self.mDataloader:
                # optimizer.zero_grad()函数会遍历模型的所有参数，通过p.grad.detach_()方法截断反向传播的梯度流，
                # 再通过p.grad.zero_()函数将每个参数的梯度值设为0，即上一次的梯度记录被清空。
                self.mOptimizer.zero_grad()
                # 模型输出
                outputs = self.mQmodel(inputs)
                # 交叉熵损失
                loss = self.mCriterion(outputs, targets)
                # 反向传播求梯度
                loss.backward()
                # 更新所有参数
                self.mOptimizer.step()
            # 查看横型参数(一次训练后,loss值应该越来越小,才符合模型训练的结果)
            self.show_mode_params(self.mQmodel, "-" * 30 + f"epoch={epoch}-> end:")

        # 保存模型
        torch.save(self.mQmodel.state_dict(), self.mQmodelPath)

    # 验证模型
    def doValidateProcess(self):
        print("doValidateProcess:")
        quadrant_type = ["I", "II", "III", "IV", "X-axis", "Y-axis", "Origin"]

        # 将数据转换为DataLoader
        dataset = TensorDataset(self.mValData, self.mValLabels)
        dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)
        # 验证每一个新的象素点(用数据加载器加载每一个要验证的像素点)
        total = 0
        sucCnt = 0
        for tmpNewPoint, tmpRealLabel in dataLoader:
            total += 1
            with torch.no_grad():
                self.mQmodel.eval()
                prediction = torch.argmax(self.mQmodel(tmpNewPoint), dim=1).item()
            # 验证状态,成功状态就累加(评估模型的泛化能力)
            realValue = tmpRealLabel.item()
            success = prediction == realValue
            if success:
                sucCnt += 1
            # 将Tensor转化为numpy
            tmpP = tmpNewPoint.numpy()
            tx = tmpP[0,0]
            ty = tmpP[0, 1]
            # print(f'{tx, ty} is {quadrant_type[realValue]}, predict is {quadrant_type[prediction]}, state={success}')
        self.mSucRatio = 1.0 * sucCnt / total
        print(f'success ratio:{self.mSucRatio}')

# 基本测试
def doCommonTest():
    # 数据生成器
    dataGen = DataGenerator()
    trainDataDesc = (-5, 6, 300)
    valDataDesc = (-5, 6, 500)
    dataGen.generateData(trainDataDesc, valDataDesc)
    td, tl, vd, vl = dataGen.toTensorObj()
    print(td.shape, tl.shape)
    print(vd.shape, vl.shape)

    # 模型处理类
    modelProc = ModelProcessor()
    modelProc.setTrainValData(td, tl, vd, vl)
    modelProc.setLayers([(2, 32), (32, 16), (16, 7)])
    modelProc.setEpochs(5)
    # 执行训练
    modelProc.doTrainProcess()
    # 验证模型
    modelProc.doValidateProcess()


# 训练数据不变,调节epochs时,分析模型泛化能力
def doAnalyzeByEpochs():
    # 数据生成器
    dataGen = DataGenerator()
    trainDataDesc = (-5, 6, 300)
    valDataDesc = (-2147483648, 2147483647, 500)
    layers = [(2, 32), (32, 16), (16, 7)]
    dataGen.generateData(trainDataDesc, valDataDesc)

    # 只分析epochs数据的影响,其它所有参数不变
    lstProcs = []
    model_cnt = 20
    epochs = []
    for i in range(model_cnt):
        epochs.append((i + 1) * 50)
    for i in range(model_cnt):
        print(f"test->{i}:")
        # Tensor数据对象(所有对象的训练数据和验证数据相同)
        # 重新得到Tensor对象(防止数据被不同模型修改）
        td, tl, vd, vl = dataGen.toTensorObj()

        # 模型处理类
        modelProc = ModelProcessor()
        lstProcs.append(modelProc)
        modelProc.setTrainValData(td, tl, vd, vl)
        modelProc.setLayers(layers)
        modelProc.setEpochs(epochs[i])
        modelProc.setQmodelSavePath("./model/quadrant_v2_%d.pt" % (epochs[i]))
        # 执行训练(并记录时长)
        lasttime = time.time()
        modelProc.doTrainProcess()
        curtime = time.time()
        modelProc.setTrainTime(curtime - lasttime)

        # 验证模型
        modelProc.doValidateProcess()

    # 数据可视化显示
    sucRatios = []
    trainTimes = []
    for tmpModel in lstProcs:
        sucRatios.append(tmpModel.mSucRatio)
        trainTimes.append(tmpModel.mTrainTime)
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(1,2,1)
    # 绘制点线图
    plt.plot(epochs, sucRatios, marker='o', linestyle='-', color='blue')
    plt.xlabel('训练次数')
    plt.ylabel('预测准确率')
    plt.title('次数与性能关系')
    plt.subplot(1,2,2)
    plt.plot(epochs, trainTimes, marker='o', linestyle='-', color='red')
    plt.xlabel('训练次数')
    plt.ylabel('耗时(s)')
    plt.title('次数与时长关系')
    plt.savefig('./model/model_epochs_ratio.png')
    plt.show()

# 改变训练数据(分布范围),其它数据不变,分析模型泛化能力
def doAnalyzeByTrainData():
    model_cnt = 20
    valDataDesc = (-2147483648, 2147483647, 500)
    layers = [(2, 32), (32, 16), (16, 7)]
    epochs = 300
    # 将本样本数据的范围扩大,其它数据不变
    lstProcs = []
    highs = []
    for i in range(model_cnt):
        print(f"test->{i}:")
        dataGen = DataGenerator()
        # 范围扩大后(原点和x,y轴上的数据随机生成的概率就急速变小(需要注意))
        low = -5 - 100 * i
        high = 6 + 100 * i
        trainDataDesc = (low, high, 300)
        highs.append(high - low)
        dataGen.generateData(trainDataDesc, valDataDesc)
        td, tl, vd, vl = dataGen.toTensorObj()

        # 模型处理类
        modelProc = ModelProcessor()
        lstProcs.append(modelProc)
        modelProc.setTrainValData(td, tl, vd, vl)
        modelProc.setLayers(layers)
        modelProc.setEpochs(epochs)
        modelProc.setQmodelSavePath("./model/quadrant_v2_td%d.pt" % (i))
        # 执行训练(并记录时长)
        lasttime = time.time()
        modelProc.doTrainProcess()
        curtime = time.time()
        modelProc.setTrainTime(curtime - lasttime)

        # 验证模型
        modelProc.doValidateProcess()

    # 数据可视化显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    sucRatios = []
    for tmpModel in lstProcs:
        sucRatios.append(tmpModel.mSucRatio)
    # 绘制点线图
    plt.plot(highs, sucRatios, marker='o', linestyle='-', color='blue')
    plt.xlabel('样本的宽度')
    plt.ylabel('预测准确率')
    plt.title('样本范围与性能关系')
    plt.savefig('./model/model_traindata_ratio.png')
    plt.show()

# 改变学习率分析模型,其它数据不变,分析模型泛化能力
def doAnalyzeByLr():
    # 数据生成器
    dataGen = DataGenerator()
    trainDataDesc = (-10, 11, 350)
    valDataDesc = (-2147483648, 2147483647, 500)
    layers = [(2, 32), (32, 16), (16, 7)]
    dataGen.generateData(trainDataDesc, valDataDesc)

    lstProcs = []
    epochs = 500
    model_cnt = 20
    lrs = np.linspace(0.0005, 0.001, 20)
    # print(lrs)
    for i in range(model_cnt):
        print(f"test->{i}:")
        # Tensor数据对象(所有对象的训练数据和验证数据相同)
        # 重新得到Tensor对象(防止数据被不同模型修改）
        td, tl, vd, vl = dataGen.toTensorObj()

        # 模型处理类
        modelProc = ModelProcessor()
        lstProcs.append(modelProc)
        modelProc.setTrainValData(td, tl, vd, vl)
        modelProc.setLayers(layers)
        modelProc.setEpochs(epochs)
        modelProc.setLr(lrs[i])
        modelProc.setQmodelSavePath("./model/quadrant_v2_lr_%.05f.pt" % (lrs[i]))

        # 执行训练(并记录时长)
        lasttime = time.time()
        modelProc.doTrainProcess()
        curtime = time.time()
        modelProc.setTrainTime(curtime - lasttime)

        # 验证模型
        modelProc.doValidateProcess()

    # 分析各模型
    sucRatios = []
    trainTimes = []
    for tmpModel in lstProcs:
        sucRatios.append(tmpModel.mSucRatio)
        trainTimes.append(tmpModel.mTrainTime)
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.subplot(1, 2, 1)
    # 绘制点线图
    plt.plot(lrs.tolist(), sucRatios, marker='o', linestyle='-', color='blue')
    plt.xlabel('学习率')
    plt.ylabel('预测准确率')
    plt.title('学习率与准确率关系')
    plt.subplot(1, 2, 2)
    plt.plot(lrs.tolist(), trainTimes, marker='o', linestyle='-', color='red')
    plt.xlabel('学习率')
    plt.ylabel('耗时(s)')
    plt.title('学习率与时长关系')
    plt.savefig('./model/model_lr_ratio.png')
    plt.show()

if __name__ == "__main__":
    # doCommonTest()
    # doAnalyzeByEpochs()
    # doAnalyzeByTrainData()
    doAnalyzeByLr()