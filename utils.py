# utils.py
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(engine_graph, title="Computational Graph"):
    # 1. 建立 NetworkX 的有向圖物件
    G = nx.DiGraph()

    # 2. 將我們的節點與連線載入
    for node in engine_graph.nodes:
        # 設定節點顯示文字：包含名稱與計算結果
        # 如果還沒計算 (None)，就顯示 ?
        val_str = "?" if node.value is None else str(node.value)
        label = f"{node.name}\n(val: {val_str})"
        
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