import numpy as np
import matplotlib.pyplot as plt
import cv2

# 設定 Matplotlib 支援繁體中文顯示 (解決方塊字問題)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 微軟正黑體
plt.rcParams['axes.unicode_minus'] = False

# 1. 模擬「手在抖動」的原始軌跡資料
# 假設我們要畫一條 y = x 的斜線，但是手很抖，加了很多雜訊
np.random.seed(42)  # 固定亂數種子，每次畫出來一樣
time_steps = 50
ideal_x = np.linspace(0, 500, time_steps)
ideal_y = np.linspace(0, 500, time_steps)

# 加上高強度雜訊 (生硬的原始相機輸入)
raw_x = ideal_x + np.random.normal(0, 15, time_steps)
raw_y = ideal_y + np.random.normal(0, 15, time_steps)

# 2. 召喚你強大的卡爾曼濾波器
kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

# 準備起點
smoothed_x = []
smoothed_y = []
kf.statePost = np.array([[raw_x[0]], [raw_y[0]], [0], [0]], dtype=np.float32)

# 3. 把抖動的資料一筆一筆餵進濾波器
for i in range(time_steps):
    kf.predict()
    measurement = np.array([[raw_x[i]], [raw_y[i]]], dtype=np.float32)
    kf.correct(measurement)
    
    smoothed_x.append(kf.statePost[0, 0])
    smoothed_y.append(kf.statePost[1, 0])

# 4. 把對比圖畫出來 (Matplotlib 超級排版)
plt.figure(figsize=(10, 6))
plt.plot(ideal_x, ideal_y, 'g--', label="Ideal Intent (使用者想畫的線)", alpha=0.5)
plt.plot(raw_x, raw_y, 'r-o', label="Raw Input (相機抓到抖動的手)", markersize=4, alpha=0.6)
plt.plot(smoothed_x, smoothed_y, 'b-X', label="Kalman Smoothed (你的演算法修正後)", markersize=6, linewidth=2)

plt.title("Trajectory Smoothing Benchmark: Raw vs Kalman Filter")
plt.xlabel("X Coordinate (Pixels)")
plt.ylabel("Y Coordinate (Pixels)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# 把圖表存成優雅的圖片檔
plt.savefig("kalman_benchmark.png", dpi=300, bbox_inches='tight')
print("✅ 圖表產出成功！請查看資料夾內的 kalman_benchmark.png")
