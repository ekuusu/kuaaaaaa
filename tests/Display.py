#这段代码展示了一个简单的 Flask 应用，用于接收前端发送的训练请求，调用后端逻辑处理数据，并返回处理结果。
import sys
import os
import matplotlib
matplotlib.use('Agg')  # 使用非 GUI 后端
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
from dlframe.WebManager import WebManager

app = Flask(__name__, template_folder='../templates')
web_manager = WebManager()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json()
        print("Received data:", data)  # 打印接收到的数据
        
        # 数据验证
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid input format, expected JSON object'}), 400
        
        dataset_name = data.get('dataset')
        model_name = data.get('model')
        test_size = data.get('test_size', 0.2)  # 默认测试集比例为0.2
        
        # 尝试将 test_size 转换为浮点数
        try:
            test_size = float(test_size)
        except ValueError:
            return jsonify({'error': 'test_size must be a number'}), 400
        
        # 检查必需字段
        if not dataset_name or not model_name:
            return jsonify({'error': 'Missing required fields: dataset and model'}), 400
        
        results = web_manager.process_request(dataset_name, model_name, test_size)
        print("Results:", results)  # 打印结果

        # 确保 results 是有效的
        if results is None:
            return jsonify({'error': 'No results returned from processing'}), 500
        
        # 确保 results 是字典类型
        if not isinstance(results, dict):
            return jsonify({'error': 'Invalid results format, expected JSON object'}), 500

        return jsonify(results)
    except Exception as e:
        print("Error occurred:", str(e))  # 打印错误信息
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask application...")
    print("Template folder:", app.template_folder)
    app.run(debug=True)