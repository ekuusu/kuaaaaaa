import queue
import threading
import asyncio
import json
import websockets
import traceback

from dlframe.CalculationNodeManager import CalculationNodeManager
from dlframe.Logger import Logger
from .ModelManager import ModelManager
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine
from sklearn.metrics import accuracy_score
#这段代码实现了一个基于 WebSocket 的服务器，用于与机器学习后端进行交互，并使用 scikit-learn 处理数据集和模型。


class SendSocket:
    def __init__(self, socket) -> None:
        self.sendBuffer = queue.Queue()
        self.socket = socket
        self.sendThread = threading.Thread(target=self.threadWorker, daemon=True)
        self.sendThread.start()

    def threadWorker(self):
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        event_loop.run_until_complete(self.sendWorker())

    async def sendWorker(self):
        while True:
            content = self.sendBuffer.get()
            await self.socket.send(content)

    def send(self, content: str):
        self.sendBuffer.put(content)

class WebManager(CalculationNodeManager):
    def __init__(self, host='0.0.0.0', port=8765, parallel=False) -> None:
        super().__init__(parallel=parallel)
        self.host = host
        self.port = port
        self.model_manager = ModelManager()
        from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine
        self.datasets = {
            'iris': load_iris(),
            'digits': load_digits(),
            'breast_cancer': load_breast_cancer(),
            'wine': load_wine()
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            raise exc_val
        self.start(self.host, self.port)

    def start(self, host=None, port=None) -> None:
        if host is None:
            host = self.host
        if port is None:
            port = self.port
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        async def onRecv(socket, path):
            msgIdx = -1
            sendSocket = SendSocket(socket)
            async for message in socket:
                msgIdx += 1
                message = json.loads(message)
                # print(msgIdx, message)

                # key error
                if not all([key in message.keys() for key in ['type', 'params']]):
                    response = json.dumps({
                        'status': 500,
                        'data': 'no key param'
                    })
                    await socket.send(response)

                # key correct
                else:
                    if message['type'] == 'overview':
                        response = json.dumps({
                            'status': 200,
                            'type': 'overview',
                            'data': self.inspect()
                        })
                        await socket.send(response)

                    elif message['type'] == 'run':
                        params = message['params']
                        conf = params

                        def image2base64(img):
                            import base64
                            from io import BytesIO
                            from PIL import Image

                            # 创建一个示例NumPy数组（图像）
                            image_np = img

                            # 将NumPy数组转换为PIL.Image对象
                            image_pil = Image.fromarray(image_np)

                            # 将PIL.Image对象保存为字节流
                            buffer = BytesIO()
                            image_pil.save(buffer, format='JPEG')
                            buffer.seek(0)

                            # 使用base64库将字节流编码为base64字符串
                            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

                            return image_base64

                        Logger.global_trigger = lambda x: sendSocket.send(json.dumps({
                            'status': 200,
                            'type': x['type'],
                            'data': {
                                'content': '[{}]: '.format(x['name']) + ' '.join([str(_) for _ in x['args']]) + getattr(x['kwargs'], 'end', '\n') if x['type'] == 'print' \
                                    else image2base64(x['args'])
                            }
                        }))
                        for logger in Logger.loggers.values():
                            if logger.trigger is None:
                                logger.trigger = Logger.global_trigger

                        try:
                            self.execute(conf)
                        except Exception as e:
                            response = json.dumps({
                                'status': 200,
                                'type': 'print',
                                'data': {
                                    'content': traceback.format_exc()
                                }
                            })
                            await socket.send(response)

                    # unknown key
                    else:
                        response = json.dumps({
                            'status': 500,
                            'data': 'unknown type'
                        })
                        await socket.send(response)

        print('The backend server is running on [{}:{}]...'.format(host, port))
        print('The frontend page is at: https://picpic2013.github.io/dlframe-front/')

        event_loop.run_until_complete(websockets.serve(onRecv, host, port))
        event_loop.run_forever()

    def load_dataset(self, dataset_name):
        return self.datasets.get(dataset_name)
        
    def process_request(self, dataset_name, model_name, test_size=0.2):
        try:
            # 加载数据集
            dataset = self.load_dataset(dataset_name)
            if dataset is None:
                raise ValueError(f"Dataset {dataset_name} not found")
            
            X = dataset.data
            y = dataset.target
            
            # 分割数据集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_size), random_state=42
            )
            
            # 训练模型
            model = self.model_manager.train_model(model_name, X_train, y_train)
            if model is None:
                raise ValueError(f"Model {model_name} not found")
            
            # 预测
            predictions = self.model_manager.predict(model_name, model, X_test)
            
            # 评估模型
            metrics = self.model_manager.evaluate_model(y_test, predictions, model_name)
            
            return {
                'predictions': predictions.tolist(),
                'actual': y_test.tolist(),
                'metrics': metrics,
                'train_size': len(y_train),
                'test_size': len(y_test)
            }
        except Exception as e:
            raise Exception(f"Error processing request: {str(e)}")