import sys,os
sys.path.append(os.pardir)
from collections import OrderedDict
from common.layers import *
import pickle



#与VGG相似的卷积神经网络



class DeepConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1}, #变化滤波器的数量
                 conv_param_2={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},#步幅为1，请注意
                 conv_param_3={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_4={'filter_num': 32, 'filter_size': 3, 'pad': 2, 'stride': 1},
                 conv_param_5={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_6={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 hidden_size=50, output_size=10):

        #上一层的节点的数量，第一个有点特殊？？
        pre_node_nums = np.array(
            [1 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3, 64 * 3 * 3, 64 * 4 * 4, hidden_size])
        #Relu激活函数的权重系数
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLUを使う場合に推奨される初期値

        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate(
                [conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            #此处变化的主要是filter_num，通道始终为1
            self.params['W' + str(idx + 1)] = weight_init_scales[idx] * \
                                              np.random.randn(conv_param['filter_num'],pre_channel_num,
                                                              conv_param['filter_size'],conv_param['filter_size'])
            #b=过滤器的数量,每个过滤器上共享一个b值
            self.params['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        #Affine--全连接--的上一层节点数为64*4*4
        self.params['W7'] = weight_init_scales[6] * np.random.randn(64 * 4 * 4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = weight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)


        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param_1['stride'], conv_param_1['pad'])

        self.layers['Relu1'] = Relu()
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], conv_param_2['stride'],
                                           conv_param_2['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'], conv_param_3['stride'],
                                           conv_param_3['pad'])

        self.layers['Relu3'] = Relu()
        self.layers['Conv4'] = Convolution(self.params['W4'], self.params['b4'], conv_param_4['stride'],
                                           conv_param_4['pad'])
        self.layers['Relu4'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv5'] = Convolution(self.params['W5'], self.params['b5'], conv_param_5['stride'],
                                           conv_param_5['pad'])

        self.layers['Relu5'] = Relu()
        self.layers['Conv6'] = Convolution(self.params['W6'], self.params['b6'], conv_param_6['stride'],
                                           conv_param_6['pad'])
        self.layers['Relu6'] = Relu()
        self.layers['Pool3'] = Pooling(pool_h=2, pool_w=2, stride=2)


        self.layers['Affine1'] = Affine(self.params['W7'], self.params['b7'])
        self.layers['Relu7'] = Relu()
        #Dropout
        self.layers['Dropout1'] = Dropout(0.5)
        self.layers['Affine2'] = Affine(self.params['W8'], self.params['b8'])
        self.layers['Dropout2'] = Dropout(0.5)

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers.values():
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t):
        #此处train_flg为True，预测模型时，需要乘一定比例
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim !=1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
        grads['W4'], grads['b4'] = self.layers['Conv4'].dW, self.layers['Conv4'].db
        grads['W5'], grads['b5'] = self.layers['Conv5'].dW, self.layers['Conv5'].db
        grads['W6'], grads['b6'] = self.layers['Conv6'].dW, self.layers['Conv6'].db
        grads['W7'], grads['b7'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W8'], grads['b8'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
    #单次存储params的参数，查看参数变化
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
    #load存储的params参数
    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i + 1)]
            self.layers[layer_idx].b = self.params['b' + str(i + 1)]



# class DeepConvNet:
#     """認識率99%以上の高精度なConvNet
#     ネットワーク構成は下記の通り
#         conv - relu - conv- relu - pool -
#         conv - relu - conv- relu - pool -
#         conv - relu - conv- relu - pool -
#         affine - relu - dropout - affine - dropout - softmax
#     """
#
#     def __init__(self, input_dim=(1, 28, 28),
#                  conv_param_1={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1}, #变化滤波器的数量
#                  conv_param_2={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},#步幅为1，请注意
#                  conv_param_3={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
#                  conv_param_4={'filter_num': 32, 'filter_size': 3, 'pad': 2, 'stride': 1},
#                  conv_param_5={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
#                  conv_param_6={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
#                  hidden_size=50, output_size=10):
#         # 重みの初期化===========
#         # 各層のニューロンひとつあたりが、前層のニューロンといくつのつながりがあるか（TODO:自動で計算する）
#         #上一层的节点的数量，第一个有点特殊？？
#         pre_node_nums = np.array(
#             [1 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3, 64 * 3 * 3, 64 * 4 * 4, hidden_size])
#         #Relu激活函数的权重系数
#         weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLUを使う場合に推奨される初期値
#
#         self.params = {}
#         pre_channel_num = input_dim[0]
#         for idx, conv_param in enumerate(
#                 [conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
#             #此处变化的主要是filter_num，通道始终为1
#             self.params['W' + str(idx + 1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'],
#                                                                                         pre_channel_num,
#                                                                                         conv_param['filter_size'],
#                                                                                        conv_param['filter_size'])
#             #b=过滤器的数量
#             self.params['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])
#             pre_channel_num = conv_param['filter_num']
#         #Affine-全连接的上一层节点数为64*4*4
#         self.params['W7'] = weight_init_scales[6] * np.random.randn(64 * 4 * 4, hidden_size)
#         self.params['b7'] = np.zeros(hidden_size)
#         self.params['W8'] = weight_init_scales[7] * np.random.randn(hidden_size, output_size)
#         self.params['b8'] = np.zeros(output_size)
#
#         # レイヤの生成===========
#         self.layers = []
#         self.layers.append(Convolution(self.params['W1'], self.params['b1'],
#                                        conv_param_1['stride'], conv_param_1['pad']))
#         #使用relu 加速收敛、计算快、避免梯度减小过快
#         #append添加可以保证顺序
#         self.layers.append(Relu())
#         self.layers.append(Convolution(self.params['W2'], self.params['b2'],
#                                        conv_param_2['stride'], conv_param_2['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
#         self.layers.append(Convolution(self.params['W3'], self.params['b3'],
#                                        conv_param_3['stride'], conv_param_3['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Convolution(self.params['W4'], self.params['b4'],
#                                        conv_param_4['stride'], conv_param_4['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
#         self.layers.append(Convolution(self.params['W5'], self.params['b5'],
#                                        conv_param_5['stride'], conv_param_5['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Convolution(self.params['W6'], self.params['b6'],
#                                        conv_param_6['stride'], conv_param_6['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
#         self.layers.append(Affine(self.params['W7'], self.params['b7']))
#         self.layers.append(Relu())
#         self.layers.append(Dropout(0.5))
#         self.layers.append(Affine(self.params['W8'], self.params['b8']))
#         self.layers.append(Dropout(0.5))
#
#         self.last_layer = SoftmaxWithLoss()
#
#     def predict(self, x, train_flg=False):
#         for layer in self.layers:
#             if isinstance(layer, Dropout):
#                 x = layer.forward(x, train_flg)
#                 #print(x.shape)
#             else:
#                 x = layer.forward(x)
#                 #print(x.shape)
#         return x
#
#     def loss(self, x, t):
#         y = self.predict(x, train_flg=True)
#         return self.last_layer.forward(y, t)
#
#     def accuracy(self, x, t, batch_size=100):
#         if t.ndim != 1: t = np.argmax(t, axis=1)
#
#         acc = 0.0
#
#         for i in range(int(x.shape[0] / batch_size)):
#             tx = x[i * batch_size:(i + 1) * batch_size]
#             tt = t[i * batch_size:(i + 1) * batch_size]
#             y = self.predict(tx, train_flg=False)
#             y = np.argmax(y, axis=1)
#             acc += np.sum(y == tt)
#
#         return acc / x.shape[0]
#
#     def gradient(self, x, t):
#         # forward
#         self.loss(x, t)
#
#         # backward
#         dout = 1
#         dout = self.last_layer.backward(dout)
#
#         tmp_layers = self.layers.copy()
#         tmp_layers.reverse()
#         for layer in tmp_layers:
#             dout = layer.backward(dout)
#
#         # 設定
#         grads = {}
#         for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
#             grads['W' + str(i + 1)] = self.layers[layer_idx].dW
#             grads['b' + str(i + 1)] = self.layers[layer_idx].db
#
#         return grads

    # def save_params(self, file_name="params.pkl"):
    #     params = {}
    #     for key, val in self.params.items():
    #         params[key] = val
    #     with open(file_name, 'wb') as f:
    #         pickle.dump(params, f)
    #
    # def load_params(self, file_name="params.pkl"):
    #     with open(file_name, 'rb') as f:
    #         params = pickle.load(f)
    #     for key, val in params.items():
    #         self.params[key] = val
    #
    #     for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
    #         self.layers[layer_idx].W = self.params['W' + str(i + 1)]
    #         self.layers[layer_idx].b = self.params['b' + str(i + 1)]

