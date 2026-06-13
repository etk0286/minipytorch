# core/node.py

class Node:
    def __init__(self, name=""):
        self.name = name
        self.value = None
        self.grad = 0.0           # [新增] 儲存反向傳播時的梯度
        
        self.indegree = 0
        self.inputs = []
        self.outgoing_nodes = []
        
    def forward(self):
        raise NotImplementedError("必須實作 forward() 方法")

    def backward(self):
        """
        [新增] 反向傳播邏輯。
        基礎節點不實作，交給子類別根據微積分法則來定義。
        """
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.name}': val={self.value}, grad={self.grad}>"


class InputNode(Node):
    def __init__(self, name, value):
        super().__init__(name)
        self.value = value
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
            self.value = sum([node.value for node in self.inputs])

    def backward(self):
        """
        [加法節點的微積分法則]
        如果 z = x + y，那麼 dz/dx = 1，dz/dy = 1。
        所以我們把收到的梯度 (self.grad) 乘以 1，並「累加」給所有的輸入來源。
        """
        for node in self.inputs:
            # 注意：這裡必須用 += (累加)，因為一個節點可能同時輸出給多個地方
            node.grad += self.grad * 1.0


class MultiplyNode(Node):
    def __init__(self, name="Multiply"):
        super().__init__(name)

    def forward(self):
        # 為了讓連鎖律更容易看懂，V2.0 我們先限制乘法節點只接收「兩個」輸入
        if len(self.inputs) == 2:
            self.value = self.inputs[0].value * self.inputs[1].value
        else:
            raise ValueError("MultiplyNode 目前只支援 2 個輸入")

    def backward(self):
        """
        [乘法節點的微積分法則]
        如果 z = x * y，那麼 dz/dx = y，dz/dy = x。
        也就是說，傳給 x 的梯度，要乘上 y 的值；傳給 y 的梯度，要乘上 x 的值。
        """
        if len(self.inputs) == 2:
            x_node = self.inputs[0]
            y_node = self.inputs[1]

            # 核心：你給我梯度，我乘上「另一個人的數值」後傳下去
            x_node.grad += self.grad * y_node.value
            y_node.grad += self.grad * x_node.value

    # 請加入到 core/node.py 最下方

class ReLUNode(Node):
    """
    ReLU 激勵函數：如果 x > 0 就保持原樣，否則輸出 0。
    """
    def __init__(self, name="ReLU"):
        super().__init__(name)

    def forward(self):
        # 假設只接一個輸入
        x = self.inputs[0].value
        self.value = max(0.0, x)

    def backward(self):
        # ReLU 的微積分法則：x > 0 時斜率為 1，否則為 0
        x = self.inputs[0].value
        local_grad = 1.0 if x > 0 else 0.0
        self.inputs[0].grad += self.grad * local_grad


class MSELossNode(Node):
    """
    均方誤差損失函數 (Mean Squared Error)。
    必須接收兩個輸入：預測值 (pred) 與 真實值 (target)。
    """
    def __init__(self, name="MSE_Loss"):
        super().__init__(name)

    def forward(self):
        pred = self.inputs[0].value
        target = self.inputs[1].value
        self.value = (pred - target) ** 2

    def backward(self):
        # 微積分連鎖律：d( (pred - target)^2 ) / d_pred = 2 * (pred - target)
        pred = self.inputs[0].value
        target = self.inputs[1].value
        
        # 將梯度傳遞給「預測值」的節點 (告訴前面的神經網路該怎麼調)
        self.inputs[0].grad += self.grad * 2 * (pred - target)
        
        # 目標值 (target) 是固定的資料，我們其實不需要更新它的梯度，
        # 但為了圖形完整性，還是把微積分寫完。
        self.inputs[1].grad += self.grad * (-2) * (pred - target)