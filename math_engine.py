import os
import cv2
import numpy as np
import base64
from openai import OpenAI
from dotenv import load_dotenv
import json
from sympy import sympify, solve, integrate, diff

class MathAIEngine:
    def __init__(self):
        """
        初始化 Math AI Engine，負責處理與 OpenAI 的溝通。
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API Key 未設定。請在 .env 中設定。")
            
        self.client = OpenAI(api_key=api_key)
    
    def solve_with_sympy(self, sympy_expr_str: str, expr_type: str) -> dict:
        """
        用 SymPy 本地處理表達式。
        根據 expr_type 決定要做什麼運算。
        自動偵測變數（不寫死 x）。
        """
        try:
            expr = sympify(sympy_expr_str)
            free_vars = expr.free_symbols

            # ====== 純計算 ======
            if expr_type == "arithmetic":
                if free_vars:
                    return {"success": False, "error": "算術表達式不應包含未知數"}
                value = expr.evalf()
                return {
                    "success": True,
                    "type": "arithmetic",
                    "result": str(value)
                }

            # ====== 以下都需要恰好一個變數 ======
            if len(free_vars) != 1:
                if len(free_vars) == 0:
                    return {"success": False, "error": "表達式中找不到未知數"}
                return {
                    "success": False,
                    "error": f"包含多個未知數 {free_vars}，目前不支援"
                }

            var = list(free_vars)[0]

            # ====== 解方程式 ======
            if expr_type == "equation":
                solutions = solve(expr, var)
                if solutions:
                    return {
                        "success": True,
                        "type": "equation",
                        "variable": str(var),
                        "result": str(solutions)
                    }
                else:
                    return {"success": False, "error": "SymPy 無法求出解"}

            # ====== 積分 ======
            elif expr_type == "integral":
                result = integrate(expr, var)
                return {
                    "success": True,
                    "type": "integral",
                    "variable": str(var),
                    "result": str(result)
                }

            # ====== 微分 ======
            elif expr_type == "derivative":
                result = diff(expr, var)
                return {
                    "success": True,
                    "type": "derivative",
                    "variable": str(var),
                    "result": str(result)
                }

            else:
                return {"success": False, "error": f"不支援的類型：{expr_type}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


    def send_to_ai(self, canvas: np.ndarray) -> str:
        """
        將畫布送到 AI 模型進行辨識與解答。
        """
        try:
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            coords = cv2.findNonZero(gray)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                pad = 20
                h_max, w_max = canvas.shape[:2]
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w_max, x + w + pad)
                y2 = min(h_max, y + h + pad)
                canvas = canvas[y1:y2, x1:x2]

            # 將圖片編碼成 base64
            _, buffer = cv2.imencode('.png', canvas)
            b64_image = base64.b64encode(buffer).decode('utf-8')

            prompt = """
You are a mathematical AI. Your ONLY job is to recognize and transcribe the handwritten math in the image. Do NOT solve it.
Return ONLY a valid JSON object (without any Markdown backticks) with the following keys:

- "raw_text": The literal text/equation you see in the image.
- "sympy_expr": The core expression formatted for Python's sympy library.
  - For equations with '=': move everything to the left side so it equals 0 (e.g., "2*x**2 + 3*x - 5").
  - For arithmetic: just transcribe it (e.g., "3 + 5*2").
  - For integrals: only the integrand, without the integral sign (e.g., for ∫x²dx, write "x**2").
  - For derivatives: only the expression being differentiated (e.g., for d/dx(x³), write "x**3").
- "type": One of the following strings:
  - "equation" — if solving for a variable (e.g., x² - 1 = 0)
  - "arithmetic" — if it's a simple calculation with no variables (e.g., 3 + 5)
  - "integral" — if it involves an integral sign ∫
  - "derivative" — if it involves d/dx or similar differentiation notation

  Examples:
{"raw_text": "2x^2 + 3x - 5 = 0", "sympy_expr": "2*x**2 + 3*x - 5", "type": "equation"}
{"raw_text": "3 + 5 × 2", "sympy_expr": "3 + 5*2", "type": "arithmetic"}
{"raw_text": "∫ x² dx", "sympy_expr": "x**2", "type": "integral"}
{"raw_text": "d/dx (x³ + 2x)", "sympy_expr": "x**3 + 2*x", "type": "derivative"}

"""
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )

            # === 第一階段完成：解析 AI 回傳的 JSON ===
            clean_text = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
            result_dict = json.loads(clean_text)
            print(f"[第一階段] AI 辨識結果: {result_dict}")

            raw_text = result_dict.get("raw_text", "")
            sympy_expr_str = result_dict.get("sympy_expr", "")
            expr_type = result_dict.get("type", "arithmetic")

            # === 第二階段：SymPy 本地運算 ===
            sympy_result = self.solve_with_sympy(sympy_expr_str, expr_type)
            print(f"[第二階段] SymPy 結果: {sympy_result}")

            # === 第三階段：判斷是否成功，決定輸出方式 ===
            if sympy_result["success"]:
                # SymPy 成功 → 丟給 AI 格式化輸出
                final_output = self.format_with_ai(raw_text, sympy_expr_str, sympy_result)
                print(f"[第三階段] 最終輸出: {final_output}")
                return final_output
            else:
                # SymPy 失敗 → 直接回傳錯誤，不讓 AI 自己算
                return f"📝 辨識結果：{raw_text}\n❌ 無法計算：{sympy_result['error']}"

        except json.JSONDecodeError:
            return f"AI 回傳的格式錯誤，無法解析：\n{clean_text}"
        except Exception as e:
            return f"系統發生錯誤：{str(e)}"

    def format_with_ai(self, raw_text: str, sympy_expr: str, sympy_result: dict) -> str:
        """
        第二次 AI 呼叫：拿 SymPy 的結果請 AI 產生格式化的解題說明。
        這次不傳圖片，純文字，速度更快也更便宜。
        只有在 SymPy 成功時才會呼叫。
        """
        result_type = sympy_result['type']
        variable = sympy_result.get('variable', '')

        type_descriptions = {
            "equation": "方程式求解",
            "arithmetic": "算術計算",
            "integral": "積分",
            "derivative": "微分"
        }
        type_desc = type_descriptions.get(result_type, result_type)

        prompt = rf"""
        你是一個數學老師。以下是學生的題目和 SymPy 計算出的精確答案。
        請用清楚的步驟解釋這個解答過程。

        題目：{raw_text}
        數學表達式：{sympy_expr}
        運算類型：{type_desc}
        請用繁體中文回答，格式簡潔明瞭。
        【重要格式要求】：
        所有的數學公式與運算過程，請務必使用標準的 LaTeX 語法。
        請「絕對不要」使用 `\[` 和 `\]` 來包裝公式。
        所有的公式「只能」用 `$$` 包起來（例如：$$ x^2 + y^2 = r^2 $$）。
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        return response.choices[0].message.content

class GestureController:
    def __init__(self):
        """
        負責處理手勢邏輯與畫布繪製。
        未來的平滑化演算法 (Kalman Filter) 將會實作在這裡。
        """
        self.current_gesture = None
        self.prev_pos = None
        self.gesture_count = 0
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03



    def process_gestures(self, info, frame_shape, canvas):
        """
        根據取得的手勢資訊，決定要畫線、清除畫布還是送出解答。
        回傳: (更新後的畫布, 是否觸發送出 AI)
        """
        if canvas is None:
            canvas = np.zeros(frame_shape, dtype=np.uint8)
            
        should_send = False

        if not info:
            return canvas, should_send

        fingers, lmList = info

        if fingers == self.current_gesture:
            self.gesture_count += 1
        else:
            self.current_gesture = fingers
            self.gesture_count = 1
        
        # 模式 1: 書寫模式 (只有食指伸出)
        if fingers == [0, 1, 0, 0, 0] and self.gesture_count > 3:
            current_pos = lmList[8][1:3]
            if self.prev_pos is None:
                self.prev_pos = current_pos
                self.kf.statePost = np.array([[current_pos[0]], [current_pos[1]], [0], [0]], np.float32)
            
            # 卡爾曼濾波啟動
            self.kf.predict()
            measurement = np.array([[np.float32(current_pos[0])], [np.float32(current_pos[1])]])
            self.kf.correct(measurement)
            smoothed_pos = (int(self.kf.statePost[0,0]), int(self.kf.statePost[1,0]))
            cv2.line(canvas, smoothed_pos, self.prev_pos, (255, 0, 255), 10)
            self.prev_pos = smoothed_pos
            
        # 如果不是書寫模式，重置前一個點，避免下次畫線連起來
        else:
            self.prev_pos = None

        # 模式 2: 清除模式 (只有大拇指伸出)
        if fingers == [1, 0, 0, 0, 0] and self.gesture_count > 15:
            canvas = np.zeros(frame_shape, dtype=np.uint8)
            
        # 模式 3: 送出模式 (伸出四根手指)
        if fingers == [0, 1, 1, 1, 1] and self.gesture_count > 20:
            should_send = True

        return canvas, should_send
