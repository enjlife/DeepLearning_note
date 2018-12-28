# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from dataset.mnist import load_mnist
from deeplearning_bases.ch08.deep_convert import DeepConvNet
from deeplearning_bases.common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
x_train = x_train[:100]
t_train = t_train[:100]
network = DeepConvNet()
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=2, mini_batch_size=10,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=5)
trainer.train()

# # パラメータの保存
# network.save_params("deep_convnet_params.pkl")
# print("Saved Network Parameters!")