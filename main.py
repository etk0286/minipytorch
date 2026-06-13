# main.py
from core.node import InputNode, AddNode, MultiplyNode, MSELossNode
from core.graph import Graph
from utils import visualize_graph  # 把畫圖工具請回來！

def main():
    print("=== 啟動 AI 訓練引擎 (神經網路感知器) ===")
    graph = Graph()

    # 1. 建立節點
    node_x = graph.add_node(InputNode("Data_X", 0.0))
    node_y_true = graph.add_node(InputNode("Target_Y", 0.0))
    node_w = graph.add_node(InputNode("Weight_W", 0.5)) 
    node_b = graph.add_node(InputNode("Bias_b", 0.0))

    node_mul = graph.add_node(MultiplyNode("w*x"))
    node_add = graph.add_node(AddNode("pred"))
    node_loss = graph.add_node(MSELossNode("Loss"))

    # 2. 接線
    graph.add_edge(node_w, node_mul)
    graph.add_edge(node_x, node_mul)
    graph.add_edge(node_mul, node_add)
    graph.add_edge(node_b, node_add)
    graph.add_edge(node_add, node_loss)    
    graph.add_edge(node_y_true, node_loss) 

    # 3. 準備訓練資料 (隱藏規律：y = 3x + 2)
    X_train = [1.0, 2.0, 3.0, 4.0, 5.0]
    Y_train = [5.0, 8.0, 11.0, 14.0, 17.0]

    # --- 視覺化 1：訓練前的「憨憨」模型 ---
    # 先隨便塞一筆資料進去，讓圖形有初始值可以顯示
    node_x.value = X_train[0]
    node_y_true.value = Y_train[0]
    graph.execute()
    print("顯示【訓練前】的神經網路狀態 (請關閉視窗以開始訓練...)")
    visualize_graph(graph, title="Before Training (Initial Random Weights)")

    # 4. 開始訓練
    learning_rate = 0.01  
    epochs = 1000        
    
    # [新增] 準備一個空陣列，用來記錄歷史 Loss
    loss_history = []

    print("\n[開始訓練]")
    for epoch in range(epochs):
        total_loss = 0.0
        for x_val, y_val in zip(X_train, Y_train):
            node_x.value = x_val
            node_y_true.value = y_val

            graph.execute()
            total_loss += node_loss.value
            graph.backward(target_node=node_loss)

            node_w.value -= learning_rate * node_w.grad
            node_b.value -= learning_rate * node_b.grad

        # 計算這個 Epoch 的平均 Loss
        avg_loss = total_loss / len(X_train)
        
        # [新增] 把算出來的平均 Loss 存進歷史紀錄裡
        loss_history.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | 模型參數: w={node_w.value:.3f}, b={node_b.value:.3f}")

    print("\n[訓練完成]")
    
    # --- [新增] 視覺化 3：繪製 Loss 收斂圖 ---
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 5))
    # 畫出折線圖：X軸是 Epoch(從 0 開始)，Y軸是 loss_history
    plt.plot(range(epochs), loss_history, marker='o', markersize=3, linestyle='-')
    
    plt.title("Training Loss Convergence Curve", fontsize=16, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Mean Squared Error (Loss)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    print("顯示【收斂圖】...")
    plt.show()

    # --- 視覺化 2：訓練後的「聰明」模型 ---
    print("顯示【訓練後】的神經網路狀態...")
    visualize_graph(graph, title="After Training (Optimized Weights)")

if __name__ == "__main__":
    main()