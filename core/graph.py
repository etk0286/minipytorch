# core/graph.py
from collections import deque

class Graph:
    def __init__(self):
        self.nodes = []
        self.execution_nodes = [] # [新增] 用來儲存運算順序的「節點物件」清單

    def add_node(self, node):
        self.nodes.append(node)
        return node

    def add_edge(self, source, target):
        target.inputs.append(source)
        source.outgoing_nodes.append(target)
        target.indegree += 1

    def execute(self):
        """
        執行引擎的核心大腦：利用 Kahn's Algorithm 進行拓撲排序與求值。
        """
        # [修復核心] 建立一份 indegree 的副本字典。
        # 這樣在排程扣減時，就不會破壞原始的圖形結構，讓圖形可以被重複執行無限次！
        current_indegrees = {node: node.indegree for node in self.nodes}

        queue = deque()
        for node in self.nodes:
            if current_indegrees[node] == 0:
                queue.append(node)

        self.execution_nodes = [] # 清空歷史紀錄
        
        while queue:
            current_node = queue.popleft()
            current_node.forward()
            self.execution_nodes.append(current_node)

            for downstream_node in current_node.outgoing_nodes:
                # [修復核心] 扣減的是副本字典裡的值，而不是 node 裡面的真實屬性
                current_indegrees[downstream_node] -= 1
                
                if current_indegrees[downstream_node] == 0:
                    queue.append(downstream_node)

        # 防呆機制：檢查是否有無窮迴圈
        if len(self.execution_nodes) != len(self.nodes):
            raise RuntimeError("執行失敗：系統崩潰！偵測到運算圖中存在無窮迴圈 (Cycle)。")

        return [node.name for node in self.execution_nodes]

    def backward(self, target_node):
        """
        [新增] 反向傳播引擎
        target_node 通常是網路的最後一層 (Loss 函數或最終輸出)
        """
        # 1. 確保反向傳播前，所有節點的梯度都歸零 (非常重要！)
        for node in self.nodes:
            node.grad = 0.0

        # 2. 設定終點的初始梯度為 1 (因為 dz/dz = 1)
        target_node.grad = 1.0

        # 3. 將前向傳播的順序「反轉」(從尾巴推回頭)
        reversed_order = reversed(self.execution_nodes)

        # 4. 依序執行每個節點的偏微分邏輯
        for node in reversed_order:
            node.backward()