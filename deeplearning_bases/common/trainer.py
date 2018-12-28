# coding: utf-8
from deeplearning_bases.common.optimizer import *

#训练函数Trainer.
#1.支持多种权重更新方法,自定义每次迭代的数据集batch大小
#2.自定义是否打印每次迭代的准确率以及测试准确率的数据集大小evalute
#3.打印最终模型的准确率
class Trainer:
    def __init__(self, network, x_train, t_train, x_test, t_test, epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr': 0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network

        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        #小批量的训练集的size
        self.batch_size = mini_batch_size
        #每个纪元迭代的样本数
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'adagrad': AdaGrad, 'adam': Adam}
        # 展开参数字典的value ，https://docs.python.org/2.7/tutorial/controlflow.html#unpacking-argument-lists
        #momentum和adagrad和adam有默认参数
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        #每个epoch迭代的样本数=训练集数量/小批量数量（输入）
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        #最大迭代数 = 每一epoch迭代的batch的大小*epoch的数量
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        #随机抽取batch_size的训练集样本
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        #计算梯度，并更新参数，计算loss
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        loss = self.network.loss(x_batch, t_batch)
        #一次迭代的loss
        self.train_loss_list.append(loss)
        #如果verbose为True，打印每次迭代的lloss   会冗长
        if self.verbose: print("train loss:" + str(loss))
        #当前迭代数可以整除一个epoch的所需的迭代数时，epoch+1，并且打印出该epoch的模型准确率
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            #根据evalute_sample_num_per_epoch给出该epoch评估的样本量，如果不给出默认全部
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ====")
        #执行完一次迭代，+1
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()
        #迭代完成后，获取模型的全部测试集的准确率
        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))







