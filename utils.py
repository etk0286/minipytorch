import networkx as nx
import matplotlib.pyplot as plt
import numpy as np  # [新增] 為了判斷矩陣維度

def visualize_graph(engine_graph, title="Computational Graph"):
    # 1. 建立 NetworkX 的有向圖物件
    G = nx.DiGraph()

    # 2. 將我們的節點與連線載入
    for node in engine_graph.nodes:

        if node.value is None:
            val_str = "?"
        elif isinstance(node.value, np.ndarray):
            # 如果是 Numpy 陣列 (例如 784 個像素)，只印出維度形狀
            val_str = f"shape: {node.value.shape}"
        else:
            # 如果是單一純量 (例如 Loss 值)，印出小數點後四位
            try:
                val_str = f"{float(node.value):.4f}"
            except:
                val_str = str(node.value)
                
        label = f"{node.name}\n({val_str})"
        
        G.add_node(node.name, label=label)

        # 根據 outgoing_nodes 建立有向邊 (箭頭)
        for target in node.outgoing_nodes:
            G.add_edge(node.name, target.name)

    # 3. 計算每個節點在畫布上的座標位置
    # spring_layout 會用類似物理彈簧排斥力的演算法把節點散開
    pos = nx.spring_layout(G, seed=42)

    # 4. 開始使用 Matplotlib 繪圖
    plt.figure(figsize=(8, 6))
    labels = nx.get_node_attributes(G, 'label')

    # 畫出節點與連線
    nx.draw(G, pos,
            with_labels=True,
            labels=labels,
            node_size=3500,           # 節點大小
            node_color='#AED6F1',     # 節點顏色 (淺藍色)
            font_size=10,             # 字體大小
            font_weight='bold',
            arrowsize=20,             # 箭頭大小
            edge_color='#5D6D7E',     # 線條顏色
            width=2.0)                # 線條粗細

    # 加上標題並顯示畫布
    plt.title(title, fontsize=16, fontweight='bold')
    plt.margins(0.2) # 留白避免邊緣被切到
    plt.show()