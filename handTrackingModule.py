import cv2
import numpy as np
import mediapipe as mp


class handDetector():
    def __init__(self, mode = False, max_hands = 1, model_complexity = 1, min_det_conf = 0.7, min_tracking_confidence = 0.7):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.min_det_conf = min_det_conf
        self.min_tracking_confidence = min_tracking_confidence
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.max_hands, self.model_complexity, self.min_det_conf, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.prev_fingers = [0, 0, 0, 0, 0]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mphands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        allHands = []
        if self.results.multi_hand_landmarks:
            # zip 同時把三包資料拿出來：左/右手標籤, 2D座標, 3D座標
            for handType, handLMS, worldLMS in zip(
                self.results.multi_handedness, 
                self.results.multi_hand_landmarks, 
                self.results.multi_hand_world_landmarks
            ):
                myHand = {}         # 每抓到一隻手，就準備新袋子
                lmList = []         # 拿來裝 2D 點用的袋子
                worldLmList = []    # 拿來裝 3D 點用的袋子

                # 第一個小迴圈：拆 handLMS 包裝 (2D 相對座標)
                for id, lm in enumerate(handLMS.landmark):
                    h, w, c = img.shape
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    lmList.append([id, cx, cy, cz])
                
                # 第二個小迴圈：拆 worldLMS 包裝 (3D 公尺座標)
                for id, w_lm in enumerate(worldLMS.landmark):
                    # ⚠️ 這裡的重點：w_lm 本身就是真實的公尺，不要乘上 w 或 h
                    # 直接把 x, y, z 放進去就好！
                    worldLmList.append([w_lm.x, w_lm.y, w_lm.z])

                # 把裝好的兩個袋子放進 myHand 字典裡
                myHand["lmList"] = lmList
                myHand["worldLmList"] = worldLmList

                if handType.classification[0].label == "Right":
                    myHand["type"] = "Left"
                else:
                    myHand["type"] = "Right"
                
                allHands.append(myHand)
                
            if draw and len(allHands) > 0:
                # 這裡如果要有畫圖效果也可以照舊保留
                first_hand_lmList = allHands[0]["lmList"]
                cv2.circle(img, (first_hand_lmList[8][1], first_hand_lmList[8][2]), 5, (255, 0,0), cv2.FILLED)
                
        return allHands, img

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        fingers = []
        myHandType = myHand["type"]
        lm_list = myHand["lmList"]
        # Removing the first element from each sublist
        myLmList = [sublist[1:] for sublist in lm_list]
        worldLmList = myHand["worldLmList"]

        # Printing the updated list
        #print(myLmList)
        #print(worldLmList)

        # if not hasattr(self, 'printed_once'):
        #     print(myLmList)
        #     self.printed_once = True  

        if self.results.multi_hand_landmarks:

            # Thumb(用x軸判斷大拇指是否張開)
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers（用向量夾角與 tipIds 偏移來判斷四指是否伸直）
            for id in range(1, 5):
                # 利用指尖編號回推對應的關節點 (TIP為8, 則PIP為6, MCP為5)
                mcp_idx = self.tipIds[id] - 3
                pip_idx = self.tipIds[id] - 2
                tip_idx = self.tipIds[id]
                
                # a = np.array(myLmList[mcp_idx][:2])  # 指根
                # b = np.array(myLmList[pip_idx][:2])  # 第一關節（轉折點）
                # c = np.array(myLmList[tip_idx][:2])  # 指尖

                a = np.array(worldLmList[mcp_idx])   # [x, y, z] 公尺，完整 3D
                b = np.array(worldLmList[pip_idx])
                c = np.array(worldLmList[tip_idx])

                vec_ba = a - b  # 從關節指向指根
                vec_bc = c - b  # 從關節指向指尖
                
                # 加上 1e-6 避免分母為 0
                cosine = np.dot(vec_ba, vec_bc) / (np.linalg.norm(vec_ba) * np.linalg.norm(vec_bc) + 1e-6)
                angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

                # ----- 新增：歸一化長度輔助判斷 -----
                # 1. 算出這隻手的「專屬比例尺」(手腕0 到 掌心9 的距離)
                palm_ref = np.linalg.norm(np.array(worldLmList[9]) - np.array(worldLmList[0]))
                
                # 2. 算出這根手指「到底伸出了多長」(指尖c 到 指根a 的距離)，並除以比例尺
                finger_length = np.linalg.norm(c - a) / (palm_ref + 1e-6)
                
                # print(f"Finger {id} angle: {angle:.1f}, len: {finger_length:.2f}") 

                prev_state = self.prev_fingers[id]
                if prev_state == 0:
                    # 只要角度很直 (>160)，或是長度伸得很長 (>0.85)，都算開啟
                    if angle > 140 or finger_length > 0.85:
                        current = 1
                    else:
                        current = 0
                elif prev_state == 1:
                    # 只要角度沒垮掉 (>135)，或是長度還夠長 (>0.75)，就維持開啟
                    if angle > 135 or finger_length > 0.75:
                        current = 1
                    else:
                        current = 0
                fingers.append(current)
                
        self.prev_fingers = fingers
        return fingers
def main():
    #Create a Video Capture Object
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        ret, frame = cap.read()
        if ret:
            frame = detector.findHands(frame)
            allHands, img = detector.findPosition(frame)
            if allHands:
                #print(allHands)
                hand1 = allHands[0]
                lmList = hand1["lmList"]
                type = hand1["type"]
                cv2.circle(frame, (lmList[4][1], lmList[4][2]), 5, (0, 255, 0), cv2.FILLED)
                fingers = detector.fingersUp(hand1)
                #print(fingers)
                #print(f"H1 = {fingers.count(1)}", end = "")
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('1'):
                break
        else:
            break


if __name__ == "__main__":
    main()