"""
CNN CUDA(统一计算设备架构,是由NVIDIA推出的通用并行计算架构)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

# CUDA环境是否存在(如果不存在就使用CPU环境)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # 特殊值数据
    def random_special_point(self, index, count, low, high):
        perCnt = count // 3
        if index < perCnt:
            tx = 0
            ty = 0
            tl = self.check_quadrant(tx, ty)
        elif index < perCnt * 2:
            tx = 0
            ty = self.random_int(low, high)
            tl = self.check_quadrant(tx, ty)
        else:
            tx = self.random_int(low, high)
            ty = 0
            tl = self.check_quadrant(tx, ty)
        return tx, ty, tl

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
        specialCnt = count // 5
        for i in range(count):
            # 特殊坐标点(原点,x轴,y轴样本数据[随机数范围变大后,落在原点,x轴,y轴的概率极小])
            if i < specialCnt:
                tx, ty, tl = self.random_special_point(i, specialCnt, low, high)
            # 其它数据
            else:
                tx = self.random_int(low, high)
                ty = self.random_int(low, high)
                tl = self.check_quadrant(tx, ty)
            self.trainData.append([tx, ty])
            self.trainLabels.append(tl)

        low = val[0]
        high = val[1]
        count = val[2]
        specialCnt = count // 5
        for i in range(count):
            if i < specialCnt:
                tx, ty, tl = self.random_special_point(i, specialCnt, low, high)
            else:
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
        # # 输入层,中间层,输出层列表
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
            # 迁移到GPU上
            tmpLayer.to(device)
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
        self.mQmodelPath = "./model/quadrant_v3.pt"
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
        # 将模型移到GPU上
        self.mQmodel.to(device)

        # 设置层(设置输入层,中间层,输出层)
        self.mQmodel.setLayers(self.mLayers)

        # 损失函数
        self.mCriterion = nn.CrossEntropyLoss()
        self.mCriterion.to(device)

        # 优化器(采用Adam算法)
        self.mOptimizer = optim.Adam(self.mQmodel.parameters(), lr=self.mLr)

        # 将数据转换为 DataLoader
        dataset = TensorDataset(self.mTrainData, self.mTrainLabels)
        # shuffle为True 每次训练时,数据加载器会扰乱数据的先后顺序
        # batch_size: 数据加载器每一次要读取多少个原数据(用于训练)
        self.mDataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # 训练模型(添加训练进度显示)
        for epoch in tqdm(range(self.mEpochs), ascii=True):
            for inputs, targets in self.mDataloader:
                # 将输入数据和标签移到 GPU 上
                # data, targets = Variable(inputs).to(device), Variable(targets.long()).to(device)
                data, targets = inputs.to(device), targets.long().to(device)

                # optimizer.zero_grad()函数会遍历模型的所有参数，通过p.grad.detach_()方法截断反向传播的梯度流，
                # 再通过p.grad.zero_()函数将每个参数的梯度值设为0，即上一次的梯度记录被清空。
                self.mOptimizer.zero_grad()
                # 模型输出
                outputs = self.mQmodel(data)
                # 交叉熵损失
                loss = self.mCriterion(outputs, targets)
                # 反向传播求梯度
                loss.backward()
                # 更新所有参数
                self.mOptimizer.step()
            print("epoch=%d, train_loss:%f" % (epoch, loss.cpu().item()))

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
        for tmpNp, tmpRl in dataLoader:
            tmpNewPoint = tmpNp.to(device)
            tmpRealLabel = tmpRl

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
            tmpP = tmpNewPoint.cpu().numpy()
            tx = tmpP[0, 0]
            ty = tmpP[0, 1]
            print(f'{tx, ty} is {quadrant_type[realValue]}, predict is {quadrant_type[prediction]}, state={success}')
        self.mSucRatio = 1.0 * sucCnt / total
        print(f'success ratio:{self.mSucRatio}')

    # 加载模型
    def loadModel(self, model_path):
        self.mQmodel = QuadrantClassifier()
        """
        # 提示:
        # 这里要动态设置层数据,从源码分析得出:
        # 加载本地模型文件时,得到的所有层数据并赋值给模型对象时,没有对应的属性值就会报:
        # Unexpected key(s) in state_dict: "xxx.weight", "xxx.bias" 错误
        # 由 self.mQmodel.load_state_dict(torch.load(model_path),strict=False) 加载不会报错
        # 但本地模型对象的层根本没初始化,运行时还是会报异常(原因:由动态设置模型的层数据导致的) 
        # 因此不建议用动态方式设置模型的层关系
        """
        # 分析有序字典数据,来动态设置层属性(输入层,中间层,输出层)
        orderedDict = torch.load(model_path)
        lstLayers = []
        for key, value in orderedDict.items():
            # print(key, type(key), type(value), value.shape)
            if key.endswith(".weight"):
                if type(value) == torch.Tensor:
                    lstLayers.append((value.shape[1], value.shape[0]))
        # print(lstLayers)
        self.setLayers(lstLayers)
        self.mQmodel.setLayers(lstLayers)
        # 加载数据到模型
        self.mQmodel.load_state_dict(orderedDict)
        self.mQmodel.to(device)

# 基本测试
def doCommonTest():
    # 数据生成器
    dataGen = DataGenerator()
    trainDataDesc = (-100, 101, 500)
    valDataDesc = (-2147483648, 2147483647, 1000)
    dataGen.generateData(trainDataDesc, valDataDesc)
    td, tl, vd, vl = dataGen.toTensorObj()
    print(td.shape, tl.shape)
    print(vd.shape, vl.shape)

    # 模型处理类
    modelProc = ModelProcessor()
    modelProc.setTrainValData(td, tl, vd, vl)
    modelProc.setLayers([(2, 32), (32, 16), (16, 7)])
    modelProc.setEpochs(500)
    modelProc.setLr(0.0005)
    # 执行训练
    modelProc.doTrainProcess()
    # 验证模型
    modelProc.doValidateProcess()

# 测试本地模型
def testLocalModel():
    model_path = "./model/quadrant_v3.pt"
    dataGen = DataGenerator()
    trainDataDesc = (-32768, 32767, 500)
    # valDataDesc = (-32768, 32767, 200)
    valDataDesc = (-2147483648, 2147483647, 1000)
    dataGen.generateData(trainDataDesc, valDataDesc)
    td, tl, vd, vl = dataGen.toTensorObj()

    modelProc = ModelProcessor()
    modelProc.setTrainValData(td, tl, vd, vl)
    modelProc.loadModel(model_path)
    modelProc.doValidateProcess()

if __name__ == "__main__":
    doCommonTest()
    # testLocalModel()