from flask import *
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io
from io import BytesIO
import math
import sys
import os
import re
import requests
import base64

# 定数
CRIT = np.array([54, 62, 70, 78])
ATK = np.array([41, 47, 53, 58])
HP = np.array([41, 47, 53, 58])
EM = np.array([40, 48, 52, 58])
NUMS_DEFAULT = np.array([41, 47, 53, 58, 54, 62, 70, 78, 54, 62, 70, 78, 0, 0, 0, 0])
FONT_TYPE = "meiryo"

MAIN_OP = [("HP", "hp%"), 
           ("攻撃力", "atk%"), 
           ("防御力", "def%"), 
           ("元素チャージ効率", "er"), 
           ("元素熟知", "em"), 
           ("元素ダメージ", "element"),
           ("物理ダメージ", "physical"),
           ("会心率", "crit-rate"),
           ("会心ダメージ", "crit-dmg"),
           ("治療効果", "heal")]

POSITION = ["生の花", "死の羽", "時の砂", "空の杯", "理の冠"]

app = Flask(__name__)

CORS(app)

@app.route("/", methods=["GET", "POST"])
def hello_world():
    return "ここには何も無いよ(^^)"

@app.route("/scan-img", methods=["POST"])
def scan_img():
    # POSTリクエストから画像を取得
    file_data = request.files['image'].read()
    nparr = np.frombuffer(file_data, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    # 閾値の設定
    threshold = 140
    # 二値化(閾値140を超えた画素を255にする。)
    ret, img_edited = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
    
    img = img_edited.copy()
    if img.ndim == 2: # モノクロ
        pass
    elif img.shape[2] == 3: # カラー
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.shape[2] == 4: # 透過
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    
    # numpy.ndarray を Pillow Image に変換
    img_pil = Image.fromarray(img)

    # POSTリクエストから追加のJSONデータを取得
    score_type = request.form.get('score_type')

    res = ArtifactReader(img_pil, score_type)

    return jsonify({"option" : res.option,
                    "position" : res.pos,
                    "main_op" : res.main_op,
                    "is_crit_dmg" : res.is_crit_dmg,
                    "is_crit_rate" : res.is_crit_rate,
                    "is_atk" : res.is_atk,
                    "is_hp" : res.is_hp,
                    "is_em" : res.is_em,
                    "init" : res.init_score,
                    "score_type" : res.score_type,
                    "level" : res.level})

@app.route("/get-dist", methods=["POST"])
def get_dist():
    # リクエストから数値を取得
    data = request.get_json()

    option = int(data['option'])
    main_op = data['main_op']
    is_crit_dmg = bool(data['crit_dmg'])
    is_crit_rate = bool(data['crit_rate'])
    is_atk = bool(data['atk'])
    is_hp = bool(data['hp'])
    is_em = bool(data['em'])
    init_score = float(data['init'])
    score = float(data['score'])
    count = int(data['count'])
    score_type = data['score_type']

    # NUMSをリセット
    nums = np.copy(NUMS_DEFAULT)
    # score_typeが熟知なら変更
    if score_type == "em" :
        nums[0:4] = EM

    # オプションに応じてNUMSを調整
    if score_type == "atk" and not is_atk:
        nums[0:4] = 0
    if score_type == "hp" and not is_hp:
        nums[0:4] = 0
    if score_type == "e" and not is_em:
        nums[0:4] = 0
    if not is_crit_dmg:
        nums[4:8] = 0
    if not is_crit_rate:
        nums[8:12] = 0

    calc = Calculator(option, main_op, is_crit_dmg, is_crit_rate, is_atk, is_hp, is_em, nums, init_score, score, count, score_type)
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
    main_op = data['main_op']
    is_crit_dmg = bool(data['crit_dmg'])
    is_crit_rate = bool(data['crit_rate'])
    is_atk = bool(data['atk'])
    is_hp = bool(data['hp'])
    is_em = bool(data['em'])
    init_score = float(data['init'])
    score = float(data['score'])
    count = int(data['count'])
    score_type = data['score_type']

    # NUMSをリセット
    nums = np.copy(NUMS_DEFAULT)
    # score_typeが熟知なら変更
    if score_type == "em" :
        nums[0:4] = EM

    # オプションに応じてNUMSを調整
    if score_type == "atk" and not is_atk:
        nums[0:4] = 0
    if score_type == "hp" and not is_hp:
        nums[0:4] = 0
    if score_type == "em" and not is_em:
        nums[0:4] = 0
    if not is_crit_dmg:
        nums[4:8] = 0
    if not is_crit_rate:
        nums[8:12] = 0

    calc = Calculator(option, main_op, is_crit_dmg, is_crit_rate, is_atk, is_hp, is_em, nums, init_score, score, count, score_type)
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
    def __init__(self, img, score_type):
        self.api_key = os.getenv("GOOGLE_CLOUD_VISION_API_KEY")
        if self.api_key is None:
            print("環境変数 'GOOGLE_CLOUD_VISION_API_KEY' が設定されていません。")
            sys.exit(1)

        # Pillowの画像をバイナリデータに変換し、base64エンコード
        self.buffered = BytesIO()
        img.save(self.buffered, format="PNG")
        self.img_base64 = base64.b64encode(self.buffered.getvalue()).decode("utf-8")
        
        # 文字を読み取る
        self.url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"
        self.request_payload = {
            "requests": [
                {
                    "image": {"content": self.img_base64},
                    "features": [{"type": "TEXT_DETECTION"}],
                    "imageContext": {
                        "languageHints": ["ja", "en"]
                    }
                }
            ]
        }

        self.response = requests.post(self.url, json=self.request_payload)
        self.result = self.response.json()["responses"][0]["textAnnotations"][0]["description"]

        self.option = 0
        self.pos = None
        self.main_op = None
        self.is_crit_dmg = False
        self.is_crit_rate = False
        self.is_atk = False
        self.is_hp = False
        self.is_em = False
        self.init_score = 0
        self.score_type = score_type
        self.level = 0

        # オプション数
        self.option = len(self.find(self.result, r'\+')) - 1
        if self.option < 3:
            self.option = 3
        elif self.option > 4:
            self.option = 4

        # メインオプション
        (self.main_op, self.pos) = self.getMainOption(self.result)

        # サブオプション
        if self.contains_crti_dmg(self.result):
            self.is_crit_dmg = True
        if self.contains_crti_rate(self.result):
            self.is_crit_rate = True
        if self.contains_atk(self.result):
            self.is_atk = True
        if self.contains_hp(self.result):
            self.is_hp = True
        if self.contains_em(self.result):
            self.is_em = True

        # 初期スコア
        if score_type == "atk" :
            self.init_score = self.getScore_attack(self.result)
        elif score_type == "hp" :
            self.init_score = self.getScore_hp(self.result)
        elif score_type == "em" :
            self.init_score = self.getScore_em(self.result)
        
        # レベル
        self.level = int(self.find(self.result, r'\+')[0].split("\n")[0])
        if self.level < 0 or self.level > 20:
            self.level = 0

    def getMainOption(self, result):
        pos = self.getPosition(result.split("\n", 1)[1])
        if (pos == None):
            pos = self.getPosition(result)
        if (pos == None):
            pos = "生の花"
        
        if (pos == "生の花"):
            return ("hp", "生の花")
        if (pos == "死の羽"):
            return ("atk", "死の羽")
        
        # 時計、杯、冠の場合
        main_op = result.split("\n")[2]
        for op in MAIN_OP:
            if op[0] in main_op:
                return (op[1], pos)
        return ("hp", "生の花")

    def getPosition(self, text):
        for pos in POSITION:
            if pos in text:
                return pos
        return None
          
    def find(self, result, str):
        return [result[m.start()+len(str)-1:m.start()+len(str)+4] for m in re.finditer(str, result)]

    def getFigure(self, data):
        for str in data:
            if(str.find('%') != -1):
                return float(str.split('%')[0])
        return 0
    
    def getFigure_em(self, data):
        for str in data:
            return float(re.sub(r'\D', '', str))
        return 0

    def getScore_attack(self, result):
        score = 0
        score += self.getFigure(self.find(result.replace(" ", ""), r'会心ダメージ\+'))
        score += self.getFigure(self.find(result.replace(" ", ""),r'会心率\+')) * 2
        score += self.getFigure(self.find(result.replace(" ", ""),r'攻撃力\+'))
        return round(score,1)
    
    def getScore_hp(self, result):
        score = 0
        score += self.getFigure(self.find(result.replace(" ", ""),r'会心ダメージ\+'))
        score += self.getFigure(self.find(result.replace(" ", ""),r'会心率\+')) * 2
        score += self.getFigure(self.find(result.replace(" ", ""),r'HP\+'))
        return round(score,1)

    def getScore_em(self, result):
        score = 0
        score += self.getFigure(self.find(result.replace(" ", ""),r'会心ダメージ\+'))
        score += self.getFigure(self.find(result.replace(" ", ""),r'会心率\+')) * 2
        score += self.getFigure_em(self.find(result.replace(" ", ""),r'元素熟知\+')) / 4
        return round(score,1)
    
    def contains_crti_dmg(self, result):
        return self.getFigure(self.find(result.replace(" ", ""),r'会心ダメージ\+')) > 0
    
    def contains_crti_rate(self, result):
        return self.getFigure(self.find(result.replace(" ", ""),r'会心率\+')) > 0
    
    def contains_atk(self, result):
        return self.getFigure(self.find(result.replace(" ", ""),r'攻撃力\+')) > 0

    def contains_hp(self, result):
        return self.getFigure(self.find(result.replace(" ", ""),r'HP\+')) > 0
    
    def contains_em(self, result):
        return self.getFigure_em(self.find(result.replace(" ", ""),r'元素熟知\+')) > 0

class Calculator():
    def __init__(self, option, main_op , is_crit_dmg, is_crit_rate, is_atk, is_hp, is_em, nums, init_score, score, count, score_type):
        self.option = option
        self.main_op = main_op
        self.is_crit_dmg = is_crit_dmg
        self.is_crit_rate = is_crit_rate
        self.is_atk = is_atk
        self.is_hp = is_hp
        self.is_em = is_em
        self.nums = nums
        self.init_score = init_score
        self.score = score
        self.count = count
        self.score_type = score_type

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
            if not self.is_crit_dmg and not self.main_op == "crit-dmg":
                tmp = np.copy(self.nums)
                tmp[12:] = CRIT
                nums_4op.append(tmp)
            if not self.is_crit_rate and not self.main_op == "crit-rate":
                tmp = np.copy(self.nums)
                tmp[12:] = CRIT
                nums_4op.append(tmp)
            if self.score_type == "atk" and not self.is_atk and not self.main_op == "atk%":
                tmp = np.copy(self.nums)
                tmp[12:] = ATK
                nums_4op.append(tmp)
            if self.score_type == "hp" and not self.is_hp and not self.main_op == "hp%":
                tmp = np.copy(self.nums)
                tmp[12:] = HP
                nums_4op.append(tmp)
            if self.score_type == "em" and not self.is_em and not self.main_op == "em":
                tmp = np.copy(self.nums)
                tmp[12:] = EM
                nums_4op.append(tmp)

            main_probability = (7 - len(nums_4op)) / 7
            sub_probability = 0
            if len(nums_4op) != 0:
                sub_probability = (1 - main_probability) / len(nums_4op)

            y = np.zeros(np.amax(CRIT) * self.count + 1)

            main_y = self.getDistribution(self.nums, self.count - 1)
            y[0:main_y.shape[0]] += main_y * main_probability

            for nums in nums_4op:
                sub_y = self.getDistribution(nums, self.count - 1)
                for num_4th in nums[12:]:
                    y[num_4th:num_4th + sub_y.shape[0]] += sub_y / len(nums[12:]) * sub_probability

            return y

if __name__ == "__main__":
    app.run(threaded=True, host="0.0.0.0", port=5000)
