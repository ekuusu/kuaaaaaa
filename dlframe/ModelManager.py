#这段代码定义了一个名为 ModelManager 的类，用于管理多种机器学习模型的训练、预测和评估任务。它涵盖了分类、聚类和概率模型等不同类型的模型，同时提供了性能评估的多种指标和可视化工具。
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import numpy as np

class ModelManager:
    def __init__(self):
        self.models = {
            'knn': KNeighborsClassifier(),
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(),
            'svm': SVC(),
            'logistic_regression': LogisticRegression(),
            'maxent': LogisticRegression(multi_class='multinomial', solver='lbfgs'),
            'adaboost': AdaBoostClassifier(),
            'em': GaussianMixture(),
            'hmm': hmm.GaussianHMM(n_components=3),
            'kmeans': KMeans()
        }
        
    def get_model(self, model_name):
        return self.models.get(model_name)
        
    def train_model(self, model_name, X_train, y_train):
        model = self.get_model(model_name)
        if model_name in ['em', 'kmeans']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            model.fit(X_train)
        elif model_name == 'hmm':
            X_train = np.array(X_train).reshape(-1, 1)
            model.fit(X_train)
        else:
            model.fit(X_train, y_train)
        return model
        
    def predict(self, model_name, model, X_test):
        if model_name in ['em', 'hmm']:
            return model.predict_proba(X_test)
        elif model_name == 'kmeans':
            return model.predict(X_test)
        else:
            return model.predict(X_test) 
        
    def evaluate_model(self, y_true, y_pred, model_name):
        """计算多个评估指标"""
        metrics = {}
        if model_name not in ['em', 'hmm', 'kmeans']:
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted'))
            metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted'))
            metrics['f1'] = float(f1_score(y_true, y_pred, average='weighted'))
            
            # 生成混淆矩阵图
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # 将图转换为base64字符串
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            metrics['confusion_matrix'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            
        return metrics 