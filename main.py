from flask import *
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pyocr
import io
import math
import sys
import os
import re

import pyocr.tesseract

# 定数
CRIT = np.array([54, 62, 70, 78])
ATK = np.array([41, 47, 53, 58])
NUMS_DEFAULT = np.array([41, 47, 53, 58, 54, 62, 70, 78, 54, 62, 70, 78, 0, 0, 0, 0])
FONT_TYPE = "meiryo"

app = Flask(__name__)

CORS(app)

@app.route("/", methods=["GET", "POST"])
def hello_world():
    return "ここには何も無いよ(^^)"

@app.route("/scan-img", methods=["POST"])
def scan_img():
    # POSTリクエストから画像を取得
    file = request.files['image']
    
    # 画像をPIL形式で開く
    img = Image.open(file)

    res = ArtifactReader(img)

    return jsonify({"option" : res.option, "is_crit_dmg" : res.is_crit_dmg, "is_crit_rate" : res.is_crit_rate, "is_atk" : res.is_atk, "init" : res.init_score})

@app.route("/get-dist", methods=["POST"])
def get_dist():
    # リクエストから数値を取得
    data = request.get_json()

    option = int(data['option'])
    is_crit_dmg = bool(data['crit_dmg'])
    is_crit_rate = bool(data['crit_rate'])
    is_atk = bool(data['atk'])
    init_score = int(data['init'])
    score = int(data['score'])
    count = int(data['count'])

    # NUMSをリセット
    nums = np.copy(NUMS_DEFAULT)

    # オプションに応じてNUMSを調整
    if not is_atk:
        nums[0:4] = 0
    if not is_crit_dmg:
        nums[4:8] = 0
    if not is_crit_rate:
        nums[8:12] = 0

    calc = Calculator(option, is_crit_dmg, is_crit_rate, is_atk, nums, init_score, score, count)
    y = calc.calculate()
    x = np.zeros(y.shape[0])
    for i in range(x.shape[0]):
        x[i] = i / 10.0

    # グラフを作成
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(init_score + x, y, width=0.05)

    # グラフをメモリ内の画像として保存
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')

@app.route("/get-data", methods=["POST"])
def get_data():
    # リクエストから数値を取得
    data = request.get_json()

    option = int(data['option'])
    is_crit_dmg = bool(data['crit_dmg'])
    is_crit_rate = bool(data['crit_rate'])
    is_atk = bool(data['atk'])
    init_score = int(data['init'])
    score = int(data['score'])
    count = int(data['count'])

    # NUMSをリセット
    nums = np.copy(NUMS_DEFAULT)

    # オプションに応じてNUMSを調整
    if not is_atk:
        nums[0:4] = 0
    if not is_crit_dmg:
        nums[4:8] = 0
    if not is_crit_rate:
        nums[8:12] = 0

    calc = Calculator(option, is_crit_dmg, is_crit_rate, is_atk, nums, init_score, score, count)
    y = calc.calculate()
    x = np.zeros(y.shape[0])
    for i in range(x.shape[0]):
        x[i] = i / 10.0

    percentile = [[0, score],
                    [0, 0],
                    [25, 0],
                    [50, 0],
                    [75, 0],
                    [100, 0]]
    sum = 0.0
    for i in range(x.shape[0]):
        if init_score + x[i] >= score:
            sum += y[i]
    percentile[0][0] = round(sum * 100, 1)
    for row in range(1, len(percentile), 1):
        percentile[row][1] = round(calc.getScore(x, y, percentile[row][0] / 100), 1)

    ave = 0.0
    for i in range(x.shape[0]):
        ave += (init_score + x[i]) * y[i]

    variance = 0.0
    for i in range(x.shape[0]):
        variance += ((init_score + x[i]) - ave) ** 2 * y[i]

    skewness = 0.0
    for i in range(x.shape[0]):
        skewness += ((init_score + x[i]) - ave) ** 3 * y[i] / (math.sqrt(variance) ** 3)

    kurtosis = 0.0
    for i in range(x.shape[0]):
        kurtosis += ((init_score + x[i]) - ave) ** 4 * y[i] / (variance ** 2)
    kurtosis -= 3

    ave = round(ave, 1)
    variance = round(variance, 1)
    skewness = round(skewness, 1)
    kurtosis = round(kurtosis, 1)

    return jsonify({"percentile" : percentile, "average" : ave, "variance" : variance, "skewness" : skewness, "kurtosis" : kurtosis})

class ArtifactReader():
    def __init__(self, img):
        # OCR設定
        self.tools = pyocr.get_available_tools()
        #OCRエンジンを取得する
        if len(self.tools) == 0:
            print("OCRエンジンが指定されていません")
            sys.exit(1)
        else:
            self.tool = self.tools[0]
        print("テストテストテストテスト")
        
        # 文字を読み取る
        self.img = img
        self.builder = pyocr.builders.TextBuilder(tesseract_layout=6)
        self.result = self.tool.image_to_string(self.img,lang="jpn", builder=self.builder)

        self.option = 0
        self.is_crit_dmg = False
        self.is_crit_rate = False
        self.is_atk = False
        self.init_score = 0

        # オプション数
        self.option = len(self.find(self.result, r'\+'))

        # サブオプション
        if self.contains_crti_dmg(self.result):
            self.is_crit_dmg = True
        if self.contains_crti_rate(self.result):
            self.is_crit_rate = True
        if self.contains_atk(self.result):
            self.is_atk = True

        # 初期スコア
        self.init_score = self.getScore_attack(self.result)
    
    
    def resource_path(self, relative_path):
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)
          
    def find(self, result,str):
        return [result[m.start()+len(str)-1:m.start()+len(str)+4] for m in re.finditer(str, result)]

    def getFigure(self, data):
        for str in data:
            if(str.find('%') != -1):
                return float(str.split('%')[0])
        return 0

    def getScore_attack(self, result):
        score = 0
        score += self.getFigure(self.find(result,r'会心ダメージ\+'))
        score += self.getFigure(self.find(result,r'会心率\+')) * 2
        score += self.getFigure(self.find(result,r'攻撃力\+'))
        return round(score,1)
    
    def contains_crti_dmg(self, result):
        return self.getFigure(self.find(result,r'会心ダメージ\+')) > 0
    
    def contains_crti_rate(self, result):
        return self.getFigure(self.find(result,r'会心率\+')) > 0
    
    def contains_atk(self, result):
        return self.getFigure(self.find(result,r'攻撃力\+')) > 0

class Calculator():
    def __init__(self, option, is_crit_dmg, is_crit_rate, is_atk, nums, init_score, score, count):
        self.option = option
        self.is_crit_dmg = is_crit_dmg
        self.is_crit_rate = is_crit_rate
        self.is_atk = is_atk
        self.nums = nums
        self.init_score = init_score
        self.score = score
        self.count = count

    # スコアの伸びの分布を計算 (indexが伸び幅の10倍整数)
    def getDistribution(self, nums, count):
        dp = np.zeros((count + 1, max(nums) * count + 1))
        dp[0][0] = 1.0

        for i in range(count):
            for num in nums:
                prev = dp[i][:dp.shape[1] - num]
                dp[i + 1][num:] += prev / nums.shape[0]
        
        return dp[count]
    
    def getScore(self, x, y, percent):
        if percent == 0:
            for i in range(x.shape[0] - 1, -1, -1):
                if y[i] != 0:
                    return self.init_score + x[i]
        elif percent == 1.0:
            for i in range(x.shape[0]):
                if y[i] != 0:
                    return self.init_score + x[i]
        else:
            sum = 0.0
            for i in range(x.shape[0] - 1, -1, -1):
                sum += y[i]
                if sum >= percent:
                    return self.init_score + x[i]

    def calculate(self):
        if self.option == 4:
            y = self.getDistribution(self.nums, self.count)
            return y
        else:
            nums_4op = []
            if not self.is_crit_dmg:
                tmp = np.copy(self.nums)
                tmp[12:] = CRIT
                nums_4op.append(tmp)
            if not self.is_crit_rate:
                tmp = np.copy(self.nums)
                tmp[12:] = CRIT
                nums_4op.append(tmp)
            if not self.is_atk:
                tmp = np.copy(self.nums)
                tmp[12:] = ATK
                nums_4op.append(tmp)

            main_probability = (7 - len(nums_4op)) / 7
            sub_probability = 0
            if len(nums_4op) != 0:
                sub_probability = (1 - main_probability) / len(nums_4op)

            y = np.zeros(np.amax(CRIT) * self.count + 1)

            main_y = self.getDistribution(self.nums, self.count - 1)
            y[0:main_y.shape[0]] += main_y * main_probability

            for nums in nums_4op:
                for num_4th in nums:
                    sub_y = self.getDistribution(nums, self.count - 1)
                    y[num_4th:num_4th + sub_y.shape[0]] += sub_y / len(nums) * sub_probability

            return y

if __name__ == "__main__":
    app.run(threaded=True, host="0.0.0.0", port=5000)
