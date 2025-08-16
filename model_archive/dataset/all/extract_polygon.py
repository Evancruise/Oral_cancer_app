import cv2
import numpy as np

def extract_red_polygon_coords(image_path):
    """
    從指定圖片中讀取紅色多邊形，並返回其頂點座標。

    Args:
        image_path (str): 圖片檔案的路徑。

    Returns:
        numpy.ndarray or None: 如果找到輪廓，返回其座標陣列；否則返回 None。
    """
    # --- 1. 讀取圖片 ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"錯誤：無法讀取圖片於 {image_path}")
        return None

    # --- 2. BGR 轉換至 HSV 色彩空間 ---
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- 3. 定義紅色的 HSV 範圍 ---
    # 範圍 1 (較低的紅色)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    
    # 範圍 2 (較高的紅色)
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # --- 4. 根據範圍建立遮罩並合併 ---
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # --- 5. 尋找輪廓 ---
    # 使用 RETR_EXTERNAL 只找最外層輪廓
    # 使用 CHAIN_APPROX_SIMPLE 壓縮輪廓點
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("未偵測到任何紅色輪廓。")
        return None

    # --- 6. 假設最大輪廓即為目標多邊形 ---
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 輪廓座標的格式是 [[[x1, y1]], [[x2, y2]], ...]
    # 我們需要將其簡化為 [[x1, y1], [x2, y2], ...]
    coords = np.squeeze(largest_contour)

    return coords

# --- 主程式執行部分 ---
if __name__ == "__main__":
    # 請將 'your_image.png' 替換為您的圖片檔案名稱
    image_file = 'your_image.png' 
    
    polygon_coords = extract_red_polygon_coords(image_file)
    
    if polygon_coords is not None:
        print("成功提取紅色多邊形座標：")
        print(f"共找到 {len(polygon_coords)} 個頂點。")
        print("座標列表 (x, y):")
        for point in polygon_coords:
            print(f"({point[0]}, {point[1]})")

        # (可選) 視覺化結果：在原圖上繪製輪廓
        original_image = cv2.imread(image_file)
        cv2.drawContours(original_image, [largest_contour], -1, (0, 255, 0), 3) # 用綠色線條繪製
        cv2.imshow('Detected Red Polygon', original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()