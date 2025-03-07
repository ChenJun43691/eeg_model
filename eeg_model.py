import pandas as pd
import numpy as np
import xgboost as xgb
import joblib  # 儲存與載入模型
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

class EEGClassifier:
    def __init__(self, pca_components=100, threshold=0.6):
        """初始化 EEG 模型，設定 PCA 降維數與分類門檻"""
        self.pca_components = pca_components
        self.threshold = threshold
        self.pca = PCA(n_components=pca_components)
        self.model = None

    def load_data(self, file_path):
        """讀取 CSV 資料並進行降維"""
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1]  # 移除 target
        y = df['target']  # 目標變數（0：非癲癇，1：癲癇）
        
        # 降維
        X_reduced = self.pca.fit_transform(X)
        
        return X_reduced, y

    def train_model(self, X, y):
        """訓練 XGBoost 模型"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 使用 SMOTE 增強數據
        smote = SMOTE(sampling_strategy=1.0, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # 計算類別權重
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        # 建立 XGBoost 模型
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=8,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            gamma=0.2,
            min_child_weight=3,
            colsample_bytree=0.85,
            subsample=0.85
        )
        
        # 訓練模型
        self.model.fit(X_train_resampled, y_train_resampled)
        
        return X_test, y_test

    def evaluate_model(self, X_test, y_test):
        """評估模型準確率"""
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > self.threshold).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"XGBoost 準確率: {accuracy:.4f}")
        print("\n分類報告：")
        print(classification_report(y_test, y_pred, target_names=["非癲癇 (0)", "癲癇 (1)"]))

    def predict(self, new_data):
        """用訓練好的模型來預測新數據"""
        new_data_reduced = self.pca.transform(new_data)
        y_pred_proba = self.model.predict_proba(new_data_reduced)[:, 1]
        y_pred = (y_pred_proba > self.threshold).astype(int)
        return y_pred

    def save_model(self, model_path="eeg_xgb_model.pkl"):
        """儲存訓練好的模型"""
        joblib.dump({"model": self.model, "pca": self.pca}, model_path)
        print(f"模型已儲存至 {model_path}")

    def load_model(self, model_path="eeg_xgb_model.pkl"):
        """載入已儲存的模型"""
        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.pca = model_data["pca"]
        print(f" 模型已從 {model_path} 載入")
