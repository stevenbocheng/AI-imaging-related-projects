import cv2
import numpy as np
import matplotlib.pyplot as plt
import handTrackingModule as ht

# 設定 Matplotlib 支援繁體中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def run_real_world_benchmark():
    print("啟動真實世界相機測試...")
    print("👉 請伸出食指開始在空中畫圖 (例如畫一個圓形或寫一個字)")
    print("👉 測試完成後，請按下鍵盤上的 'q' 鍵，程式將自動為您產出對比圖！")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = ht.handDetector()
    
    # 建立一個與 math_engine.py 參數相同的卡爾曼濾波器
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
    
    kf_initialized = False

    # 用來存放測試數據的清單
    raw_trajectory_x = []
    raw_trajectory_y = []
    smoothed_trajectory_x = []
    smoothed_trajectory_y = []
    
    # 建立一張空白透明畫布用來「累積」畫圖的軌跡
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    prev_raw = None
    prev_smoothed = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        # 尋找手部特徵
        frame = detector.findHands(frame)
        hands, frame = detector.findPosition(frame, draw=True)

        if hands:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            lmList = hand["lmList"]
            
            # 手勢狀態：只有食指 (書寫模式)
            if fingers == [0, 1, 0, 0, 0]:
                raw_pos = lmList[8][1:3] # 食指指尖原本的相機座標
                raw_x, raw_y = raw_pos[0], raw_pos[1]
                
                # 初始化卡爾曼 起點
                if not kf_initialized:
                    kf.statePost = np.array([[raw_x], [raw_y], [0], [0]], dtype=np.float32)
                    kf_initialized = True
                    
                # 套用卡爾曼濾波器
                kf.predict()
                measurement = np.array([[raw_x], [raw_y]], dtype=np.float32)
                kf.correct(measurement)
                
                smoothed_x = int(kf.statePost[0, 0])
                smoothed_y = int(kf.statePost[1, 0])
                
                # 紀錄起來
                raw_trajectory_x.append(raw_x)
                raw_trajectory_y.append(raw_y)  # y 軸圖表上為了符合視覺，等等畫圖時會把 Y 反轉
                
                smoothed_trajectory_x.append(smoothed_x)
                smoothed_trajectory_y.append(smoothed_y)

                # 把「軌跡線條」畫在 canvas 上，讓使用者清楚看到畫了什麼
                if not kf_initialized: # 代表剛剛才下筆，這幀算起點（不過其實前面 if not kf_initialized 已經把 initialized 設為 True 了）
                    pass # 避開重複判斷
                
                # 如果是第一筆，把起點重置，不然會劃一條長長的線連回上次的地方
                if prev_raw is None or prev_smoothed is None:
                    prev_raw = (raw_x, raw_y)
                    prev_smoothed = (smoothed_x, smoothed_y)
                    
                cv2.line(canvas, prev_raw, (raw_x, raw_y), (0, 0, 255), 2)       # 紅色亂七八糟的線 (細一點)
                cv2.line(canvas, prev_smoothed, (smoothed_x, smoothed_y), (255, 0, 0), 4) # 藍色平滑的線 (粗一點)
                
                prev_raw = (raw_x, raw_y)
                prev_smoothed = (smoothed_x, smoothed_y)

            else:
                # 若手指縮回 (不是書寫模式)，代表斷筆，為了畫圖好看，把濾波器重置
                kf_initialized = False
                prev_raw = None
                prev_smoothed = None

        # 將累積的線條畫布疊加上去
        frame = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)
        
        # 顯示提示訊息在標題
        cv2.putText(frame, "Press 'q' to Export Benchmark Plot!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Real Tracking Benchmark (Raw='Red' vs Kalman='Blue')", frame)

        # 按 q 離開並產圖
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 防呆：如果根本沒錄到軌跡，就跳出
    if len(raw_trajectory_x) < 5:
        print("沒有錄製到足夠的軌跡，無法產生圖表！")
        return

    # === 開始生氣圖表 ===
    print("正在將您剛剛錄製的軌跡繪製成效能對比圖...")
    
    plt.figure(figsize=(10, 8))
    
    # 為了讓產出的圖表方向跟你在螢幕上看到的長得一樣，我們把 y 軸反過來（OpenCV 原點在左上，Matplotlib 在左下）
    real_raw_y = [-y for y in raw_trajectory_y]
    real_smoothed_y = [-y for y in smoothed_trajectory_y]
    
    plt.plot(raw_trajectory_x, real_raw_y, 'r--', label="Raw Camera Input (相機抓到的人手抖動)", alpha=0.6, linewidth=2, marker='o', markersize=10)
    plt.plot(smoothed_trajectory_x, real_smoothed_y, 'b-', label="Kalman Smoothed (演算法即時運算修正後)", linewidth=3)
    
    plt.title("Real-World Trajectory Smoothing Benchmark (真實人手錄製)")
    plt.xlabel("X Coordinate (Pixels)")
    plt.ylabel("Y Coordinate (Inverted Pixels)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.savefig("real_world_benchmark.png", dpi=300, bbox_inches='tight')
    print("✅ 產圖成功！請在資料夾找尋 'real_world_benchmark.png'！")

if __name__ == "__main__":
    run_real_world_benchmark()
