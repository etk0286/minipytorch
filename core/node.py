import numpy as np

class Node:
    def __init__(self, name=""):
        self.name = name
        self.value = None
        self.grad = 0.0           # 儲存反向傳播時的梯度
        
        self.indegree = 0
        self.inputs = []
        self.outgoing_nodes = []
        
    def forward(self):
        raise NotImplementedError("必須實作 forward() 方法")

    def backward(self):
        """
        反向傳播邏輯。
        基礎節點不實作，交給子類別根據微積分法則來定義。
        """
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.name}': val={self.value}, grad={self.grad}>"


class InputNode(Node):
    def __init__(self, name, value):
        super().__init__(name)
        # 確保初始值是 Numpy 陣列或純量
        self.value = value if isinstance(value, np.ndarray) else np.array(value)
        self.indegree = 0

    def forward(self):
        pass

    def backward(self):
        # 輸入節點是神經網路的起點，不需要再往回傳遞梯度了
        pass


class AddNode(Node):
    def __init__(self, name="Add"):
        super().__init__(name)

    def forward(self):
        if len(self.inputs) > 0:
            # 使用 sum 可以完美支援 Numpy 陣列的逐項相加 (矩陣加法)
            self.value = sum([node.value for node in self.inputs])

    def backward(self):
        """加法節點的微積分法則：將梯度 1:1 分發給所有輸入"""
        for node in self.inputs:
            node.grad += self.grad * 1.0


class MultiplyNode(Node):
    def __init__(self, name="Multiply"):
        super().__init__(name)

    def forward(self):
        if len(self.inputs) == 2:
            self.value = self.inputs[0].value * self.inputs[1].value
        else:
            raise ValueError("MultiplyNode 目前只支援 2 個輸入")

    def backward(self):
        """純量乘法節點微積分法則"""
        if len(self.inputs) == 2:
            x_node = self.inputs[0]
            y_node = self.inputs[1]

            x_node.grad += self.grad * y_node.value
            y_node.grad += self.grad * x_node.value


# ==========================================
# 以下為 [進階升級]：支援矩陣運算與非線性的核心節點
# ==========================================

class MatMulNode(Node):
    """
    支援二維陣列 (矩陣) 內積的計算節點
    負責執行 Y = X @ W
    """
    def __init__(self, name="MatMul"):
        super().__init__(name)

    def forward(self):
        if len(self.inputs) != 2:
            raise ValueError("MatMulNode 必須有 2 個輸入")
        
        # 確保進來的資料是 Numpy 陣列
        x_val = np.array(self.inputs[0].value)
        w_val = np.array(self.inputs[1].value)
        
        # 執行矩陣內積
        self.value = np.dot(x_val, w_val)

    def backward(self):
        """
        反向傳播：利用轉置矩陣 (Transpose) 計算梯度
        這展現了優異的資料結構記憶體佈局應用
        """
        x_node = self.inputs[0]
        w_node = self.inputs[1]

        # dL/dX = dL/dY @ W^T
        x_node.grad += np.dot(self.grad, np.array(w_node.value).T)
        
        # dL/dW = X^T @ dL/dY
        w_node.grad += np.dot(np.array(x_node.value).T, self.grad)


class ReLUNode(Node):
    """
    ReLU 激勵函數節點 (Rectified Linear Unit)
    """
    def __init__(self, name="ReLU"):
        super().__init__(name)

    def forward(self):
        # 將小於 0 的數值變成 0，大於 0 的維持不變
        x_val = np.array(self.inputs[0].value)
        self.value = np.maximum(0, x_val)

    def backward(self):
        # x > 0 時斜率為 1，否則為 0
        x_val = np.array(self.inputs[0].value)
        local_grad = (x_val > 0).astype(float)
        
        # 將梯度乘上 local_grad 後傳遞回去
        self.inputs[0].grad += self.grad * local_grad


class MSELossNode(Node):
    """
    均方誤差損失函數 (Mean Squared Error)
    全面升級為支援矩陣誤差的平均值計算
    """
    def __init__(self, name="MSE_Loss"):
        super().__init__(name)

    def forward(self):
        pred = np.array(self.inputs[0].value)
        target = np.array(self.inputs[1].value)
        
        # 取得矩陣內的所有元素取平均，確保 Loss 輸出為單一純量
        self.value = np.mean((pred - target) ** 2)

    def backward(self):
        pred = np.array(self.inputs[0].value)
        target = np.array(self.inputs[1].value)
        
        # N 為矩陣的總元素數量
        N = np.size(pred)
        
        # 微積分連鎖律：d(MSE)/d_pred = 2/N * (pred - target)
        self.inputs[0].grad += self.grad * (2 * (pred - target) / N)
        
        # target 是常數不用更新，但為維持架構完整性保留算式
        self.inputs[1].grad += self.grad * (-2 * (pred - target) / N)
