import numpy as np
import torch
import torch.nn as nn
from tprism.op.base import BaseOperator
from typing import List

class Reindex(BaseOperator):
    def __init__(self, parameters: List[str]):
        index = parameters[0].strip("[]").split(",")
        self.out = index
        pass

    def call(self, x):
        return x

    def get_output_template(self, input_template):
        if len(input_template) > 0 and input_template[0] == "b":
            return ["b"] + self.out
        else:
            return self.out


class Sigmoid(BaseOperator):
    def __init__(self, parameters: List[str]) -> None:
        pass

    def call(self, x):
        return torch.sigmoid(x)

    def get_output_template(self, input_template: List[str]) -> List[str]:
        return input_template


class Relu(BaseOperator):
    def __init__(self, parameters: List[str]):
        pass

    def call(self, x):
        return torch.relu(x)

    def get_output_template(self, input_template):
        return input_template


class Softmax(BaseOperator):
    def __init__(self, parameters: List[str]):
        pass

    def call(self, x):
        return torch.softmax(x,dim=-1)

    def get_output_template(self, input_template):
        return input_template

class Min1(BaseOperator):
    def __init__(self, parameters: List[str]):
        pass

    def call(self, x):
        return torch.clamp(x, 0.0, 1.0)

    def get_output_template(self, input_template):
        return input_template


class Mnistaddnet(BaseOperator):
    def __init__(self, parameters: List[str]):
        self.nn = SingleDigitMNISTAdditionNN()
        self.save = SaveOutput(self.nn, self.nn.encoder[0])

    def call(self, x):
        return self.nn.forward(x)

    def get_output_template(self, input_template):
        return ['j']


class SingleDigitMNISTAdditionNN(nn.Module):
    def __init__(self):
        super(SingleDigitMNISTAdditionNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # 1 28 28 -> 6 24 24
            nn.MaxPool2d(2, 2),     # 6 24 24 -> 6 12 12
            nn.ReLU(True),       # save memory
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            # nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True)        # save memory
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            # nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10),  # 10 for digit classification
            nn.Softmax(1)
        )

    def forward(self, x):
        # (batch_size * 2, 1, 28, 28): receives a pair of images for each instance in a batch
        x = x.view(-1, 1, 28, 28)
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)  # should be (batch_size * 2, 10)
        x = x.view(-1)
        return x


# モデルの指定されたレイヤーの出力と勾配を保存するクラス
class SaveOutput:
    def __init__(self, model, target_layer):  # 引数：モデル, 対象のレイヤー
        self.model = model
        self.layer_output = []
        self.layer_grad = []

        # 特徴マップを取るためのregister_forward_hookを設定
        self.feature_handle = target_layer.register_forward_hook(self.feature)
        # 勾配を取るためのregister_forward_hookを設定
        self.grad_handle = target_layer.register_forward_hook(self.gradient)

    # self.feature_handleの定義時に呼び出されるメソッド
    ## モデルの指定されたレイヤーの出力（特徴マップ）を保存する
    def feature(self, model, input, output):
        activation = output
        self.layer_output.append(activation.to("cpu").detach())

    # self.grad_handleの定義時に呼び出されるメソッド
    ## モデルの指定されたレイヤーの勾配を保存する
    ## 勾配が存在しない場合や勾配が必要ない場合は処理をスキップ
    def gradient(self, model, input, output):
        # 勾配が無いとき
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return  # ここでメソッド終了

        # 勾配を取得
        def _hook(grad):
            # gradが定義されていないが、勾配が計算されると各テンソルのgrad属性に保存されるっぽい（詳細未確認）
            self.layer_grad.append(grad.to("cpu").detach())

        # PyTorchのregister_hookメソッド（https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html）
        output.register_hook(_hook)

        # メモリの解放を行うメソッド、フックを解除してメモリを解放する

    def release(self):
        self.feature_handle.remove()
        self.grad_handle.remove()