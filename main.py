import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.node import InputNode, AddNode, MSELossNode, MatMulNode, ReLUNode
from core.graph import Graph

# ==========================================
# ⚙️ 系統模式開關：決定引擎要執行「訓練」還是「推論」
# ==========================================
# 請修改這裡的字串來切換模式："TRAIN" 或 "INFERENCE"
MODE = "INFERENCE"  


def main():
    print(f"=== 啟動 AI 引擎 (當前模式: {MODE}) ===")
    
    # ==========================================
    # 1. 載入資料 (這部分無論訓練或推論都需要用來當題庫)
    # ==========================================
    print("載入資料中 (這可能需要幾秒鐘)...")
    try:
        train_data = pd.read_csv('fashion-mnist_train.csv')
    except FileNotFoundError:
        print("❌ 找不到 'fashion-mnist_train.csv'！請確認檔案是否放在正確位置。")
        return

    # 為了能快速跑完，取前 1000 筆資料
    train_data = train_data.head(1000)
    
    Y_raw = train_data['label'].values
    X_raw = train_data.drop('label', axis=1).values

    # 正規化像素
    X_train = X_raw / 255.0
    
    # One-Hot Encoding
    num_classes = 10
    Y_train = np.eye(num_classes)[Y_raw]

    # ==========================================
    # 2. 建立計算圖 (DAG 拓樸排序結構共用區)
    # ==========================================
    graph = Graph()

    node_x = graph.add_node(InputNode("Image_X", np.zeros((1, 784))))
    node_y_true = graph.add_node(InputNode("Target_Y", np.zeros((1, 10))))

    # 隱藏層 (784 -> 128)
    W1_init = np.random.randn(784, 128) / np.sqrt(784)
    node_w1 = graph.add_node(InputNode("W1", W1_init))
    node_b1 = graph.add_node(InputNode("b1", np.zeros((1, 128))))

    # 輸出層 (128 -> 10)
    W2_init = np.random.randn(128, 10) / np.sqrt(128)
    node_w2 = graph.add_node(InputNode("W2", W2_init))
    node_b2 = graph.add_node(InputNode("b2", np.zeros((1, 10))))

    # 數學運算節點
    node_matmul1 = graph.add_node(MatMulNode("X@W1"))
    node_add1 = graph.add_node(AddNode("Hidden_Sum"))
    node_relu = graph.add_node(ReLUNode("ReLU"))       
    node_matmul2 = graph.add_node(MatMulNode("H@W2"))
    node_add2 = graph.add_node(AddNode("Pred_Y"))
    node_loss = graph.add_node(MSELossNode("Loss"))

    # 建立連線邊界 (Edges)
    graph.add_edge(node_x, node_matmul1)
    graph.add_edge(node_w1, node_matmul1)
    graph.add_edge(node_matmul1, node_add1)
    graph.add_edge(node_b1, node_add1)
    graph.add_edge(node_add1, node_relu)
    
    graph.add_edge(node_relu, node_matmul2)
    graph.add_edge(node_w2, node_matmul2)
    graph.add_edge(node_matmul2, node_add2)
    graph.add_edge(node_b2, node_add2)
    
    graph.add_edge(node_add2, node_loss)
    graph.add_edge(node_y_true, node_loss)


    # ==========================================
    # 3. 根據模式執行不同邏輯 (職責分離)
    # ==========================================
    
    if MODE == "TRAIN":
        # ------------------------------------------
        # [訓練模式] 執行拓樸排程與反向傳播 (Stack LIFO)
        # ------------------------------------------
        learning_rate = 0.05
        epochs = 15  
        loss_history = []

        print("\n[開始訓練]")
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(len(X_train)):
                node_x.value = X_train[i].reshape(1, 784)
                node_y_true.value = Y_train[i].reshape(1, 10)

                # 正向與反向傳播
                graph.execute()
                total_loss += node_loss.value
                graph.backward(target_node=node_loss)

                # 更新權重
                node_w1.value -= learning_rate * node_w1.grad
                node_b1.value -= learning_rate * node_b1.grad
                node_w2.value -= learning_rate * node_w2.grad
                node_b2.value -= learning_rate * node_b2.grad

            avg_loss = total_loss / len(X_train)
            loss_history.append(avg_loss)
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f}")

        # 訓練完成後，將陣列序列化存入硬碟
        np.savez('fashion_mnist_weights.npz', 
                 w1=node_w1.value, b1=node_b1.value, 
                 w2=node_w2.value, b2=node_b2.value)
        print("\n💾 訓練完成！模型權重已成功存檔至 'fashion_mnist_weights.npz'")

        # 繪製 Loss 曲線
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epochs+1), loss_history, marker='o', markersize=4, linestyle='-', color='#1f77b4')
        plt.title("Fashion-MNIST Training Loss", fontsize=16, fontweight='bold')
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    elif MODE == "INFERENCE":
        # ------------------------------------------
        # [推論模式] 讀取硬碟權重，僅進行單向正向走訪
        # ------------------------------------------
        if not os.path.exists('fashion_mnist_weights.npz'):
            print("\n❌ 錯誤：找不到權重檔案！請先將 MODE 設為 'TRAIN' 進行訓練。")
            return

        print("\n📂 載入模型權重中...")
        saved_weights = np.load('fashion_mnist_weights.npz')
        node_w1.value = saved_weights['w1']
        node_b1.value = saved_weights['b1']
        node_w2.value = saved_weights['w2']
        node_b2.value = saved_weights['b2']
        print("✅ 權重載入成功！系統準備好進行推論。\n")

        label_map = {
            0: "T-shirt/top", 1: "Trouser", 2: "Pullover",
            3: "Dress", 4: "Coat", 5: "Sandal",
            6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
        }

        # 為了展示，取資料集最後 100 筆當作沒看過的測試題
        test_X = X_raw[-100:] / 255.0  
        test_Y_raw = Y_raw[-100:]

        # ==========================================
        # 計算整體測試集準確率
        # ==========================================
        print("計算測試集準確率中 (僅執行正向傳播)...")
        correct_count = 0
        for i in range(len(test_X)):
            node_x.value = test_X[i].reshape(1, 784)
            graph.execute() # 僅走訪，不更新權重
            
            pred_idx = np.argmax(node_add2.value)
            if pred_idx == test_Y_raw[i]:
                correct_count += 1

        accuracy = correct_count / len(test_X) * 100
        print(f"🔥 測試集準確率: {accuracy:.2f}% ({correct_count}/{len(test_X)})\n")

        # ==========================================
        # 視覺化：隨機抽樣單張預測結果展示
        # ==========================================
        random_idx = np.random.randint(0, len(test_X))
        sample_image = test_X[random_idx]
        true_label_name = label_map[test_Y_raw[random_idx]]

        # 單向走訪預測這張隨機圖片
        node_x.value = sample_image.reshape(1, 784)
        graph.execute() 
        
        pred_idx = np.argmax(node_add2.value)
        pred_label_name = label_map[pred_idx]

        # 視覺化結果
        plt.figure(figsize=(4, 4))
        plt.imshow(sample_image.reshape(28, 28), cmap='gray') 
        
        title_color = 'green' if pred_idx == test_Y_raw[random_idx] else 'red'
        plt.title(f"AI Predict: {pred_label_name}\nTrue: {true_label_name}", 
                  color=title_color, fontsize=12, fontweight='bold')
        plt.axis('off')
        
        print("顯示【推論預測結果】...")
        plt.show()

    else:
        print("❌ 錯誤：不支援的模式！請將 MODE 設為 'TRAIN' 或 'INFERENCE'。")


if __name__ == "__main__":
    main()