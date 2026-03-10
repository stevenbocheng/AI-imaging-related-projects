import cv2
import numpy as np
import streamlit as st
import handTrackingModule as ht
from math_engine import MathAIEngine, GestureController
import time

# ==========================================
# 1. UI 初始化與設定 (View)
# ==========================================
st.set_page_config(page_title="Math with Gestures using AI", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f5f5; padding: 10px; }
    h1 { margin-bottom: 0px; }
    .header { text-align: center; margin-top: -50px; padding-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='header'>Hyper-Math Vision</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([3,2])

with col1:
    run = st.checkbox('啟動相機 (Run)', value=True)
    FRAME_WINDOW = st.image([], use_column_width=True)
with col2:
    st.header("AI 解答區")
    output_text_area = st.empty()

# ==========================================
# 2. 系統元件初始化 (Controller & Model)
# ==========================================
@st.cache_resource
def init_system_components():
    """初始化並快取系統元件，避免每次 Streamlit 重新整理時重複載入"""
    try:
        math_engine = MathAIEngine()
    except ValueError as e:
        st.error(str(e))
        st.stop()
        
    gesture_ctrl = GestureController()
    detector = ht.handDetector()
    return math_engine, gesture_ctrl, detector

math_engine, gesture_ctrl, detector = init_system_components()

# ==========================================
# 3. 主循環：影像擷取與處理
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

canvas = None
ai_response = ""
# 加入 cooldown 避免連續送出太多次請求給 API
last_request_time = 0 
COOLDOWN_SECONDS = 3

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("無法讀取相機畫面。")
        break
        
    frame = cv2.flip(frame, 1) # 鏡像翻轉

    # === 光照不變性預處理 (CLAHE) 對抗背光 ===
    # 1. 將圖片轉到 LAB 色彩空間，L = Lightness (亮度)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 2. 建立 CLAHE 對比受限適應性長條圖均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # 3. 把亮度做均衡化處理
    cl = clahe.apply(l)
    
    # 4. 把處理好的亮度跟顏色合併回去，然後轉回 BGR
    limg = cv2.merge((cl, a, b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # a. 手勢追蹤
    frame = detector.findHands(frame)
    hands, frame = detector.findPosition(frame, draw=True)
    
    info = None
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        info = (fingers, lmList)

    # b. 處理手勢邏輯與繪製
    canvas, should_send = gesture_ctrl.process_gestures(info, frame.shape, canvas)

    # c. 觸發 AI 辨識
    current_time = time.time()
    if should_send and (current_time - last_request_time > COOLDOWN_SECONDS):
        ai_response = "請稍候，AI 正在計算..."
        output_text_area.info(ai_response)
        
        # 呼叫後端引擎
        ai_response = math_engine.send_to_ai(canvas)
        last_request_time = time.time()
        
        # 清空畫布準備下一題
        canvas = np.zeros_like(frame)

    # d. 更新 UI 畫面
    # 將原始畫面與黑底亮線的畫布疊加
    frame_combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    
    # OpenCV 是 BGR，Streamlit 預設吃 RGB，可以直接設定 channels="BGR"
    FRAME_WINDOW.image(frame_combined, channels="BGR")
    
    if ai_response and ai_response != "請稍候，AI 正在計算...":
        with output_text_area.container():
            st.success("解答已送達！")
            st.markdown(ai_response)

cap.release()
cv2.destroyAllWindows()
