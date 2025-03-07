#  載入必要的模組
import pandas as pd  # 讀取 CSV 檔案
import matplotlib.pyplot as plt  # 繪製圖表
import seaborn as sns  # 美化圖表
import matplotlib  # 控制繪圖細節
from sklearn.decomposition import PCA  # 主成分分析（降維）
from sklearn.model_selection import train_test_split  # 訓練集與測試集切分
import xgboost as xgb  # XGBoost 模型
from sklearn.metrics import accuracy_score, classification_report  # 模型評估
from imblearn.over_sampling import SMOTE  # SMOTE 增強少數類別數據

# 設定 Matplotlib 使其支援中文
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  

#  讀取 EEG 數據
file_path = "EEG_Scaled_data.csv"
df = pd.read_csv(file_path)

#  檢視數據基本資訊
print("數據前幾行：")
print(df.head())  # 查看數據前 5 筆
print("\n數據欄位數：", df.shape[1])  # 查看欄位數（通道數 + target）
print("\n 缺失值統計：", df.isnull().sum().sum())  # 檢查是否有缺失值
df.info()  # 查看數據結構

#  確認 target 是否為最後一列
print("\n 確認 target 是否為最後一列：", df.columns[-5:])

#  EEG 訊號視覺化（前 500 個時間點）
plt.figure(figsize=(10, 5))
plt.plot(df.iloc[:500, 0], label="Channel_1")  
plt.plot(df.iloc[:500, 1], label="Channel_2")
plt.legend()
plt.title("EEG 訊號波形（前 500 個時間點）")
plt.xlabel("時間點")
plt.ylabel("信號強度")
plt.show()

# 目標變數 (target) 的分佈
plt.figure(figsize=(6, 4))
sns.countplot(x=df['target'])
plt.title("癲癇 vs. 非癲癇樣本數量")
plt.xlabel("0 = 非癲癇, 1 = 癲癇")
plt.ylabel("樣本數量")
plt.show()

# **降維（PCA，主成分分析）**
pca = PCA(n_components=100)  # 降到 100 維
eeg_reduced = pca.fit_transform(df.iloc[:, :-1])  # 移除 target 只進行 PCA

print("降維後的數據形狀：", eeg_reduced.shape)  # 確保形狀正確

#  **分割數據（80% 訓練，20% 測試）**
X = eeg_reduced
y = df['target']  # 目標變數（0：非癲癇，1：癲癇）

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("訓練集大小：", X_train.shape)
print("測試集大小：", X_test.shape)

#  **使用 SMOTE 增強癲癇樣本**
smote = SMOTE(sampling_strategy=1.0, random_state=42)  # 讓兩類樣本數一致
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("SMOTE 增強後的訓練集大小：", X_train_resampled.shape)

#  **調整類別權重（讓癲癇 (1) 更重要）**
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # 計算類別比例

# 訓練 **XGBoost 模型**
model = xgb.XGBClassifier(
    n_estimators=300,  # 增加樹的數量，讓模型學習更多樣本
    learning_rate=0.03,  # 降低學習速率，讓模型學習更穩定
    max_depth=8,  # 增加樹的深度，讓模型更強大
    random_state=42,
    scale_pos_weight=1.5,  # 平衡癲癇樣本的影響
    gamma=0.2,  # 剪枝，防止過擬合
    min_child_weight=3,  # 限制節點分裂，減少噪音影響
    colsample_bytree=0.85,  # 使用 85% 特徵，提升泛化能力
    subsample=0.85  # 訓練時使用 85% 數據，防止過擬合
)

# 訓練模型（使用增強後的訓練數據）
model.fit(X_train_resampled, y_train_resampled)

# 進行預測
y_pred_proba = model.predict_proba(X_test)[:, 1]  # 取得癲癇 (1) 的預測機率
threshold = 0.6  # 設定分類門檻（提高 Precision，降低 False Positive）
y_pred = (y_pred_proba > threshold).astype(int)  # 根據門檻進行分類

# **模型評估**
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost 準確率: {accuracy:.4f}")

# 顯示分類報告（Precision, Recall, F1-score）
print("\n分類報告：")
print(classification_report(y_test, y_pred, target_names=["非癲癇 (0)", "癲癇 (1)"]))
