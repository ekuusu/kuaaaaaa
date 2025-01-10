#这段代码实现了一个简单的日志记录器类 Logger，支持打印日志信息和显示图像，同时支持触发自定义处理逻辑（例如，发送数据到 WebSocket）。
class Logger:
    def __init__(self, name, trigger=None) -> None:
        self.name = name
        self.trigger = trigger

    def print(self, *x, **kwargs):
        if self.trigger is not None:
            self.trigger({
                'type': 'print', 
                'name': self.name, 
                'args': x, 

                'kwargs': kwargs
            })
            return
        print('[{}]: '.format(self.name), end=' ')
        print(*x, **kwargs)

    def imshow(self, img):
        if self.trigger is not None:
            self.trigger({
                'type': 'imshow', 
                'name': self.name, 
                'args': img
            })
            return
        print(img)

    @classmethod
    def get_logger(cls, name):
        logger = cls(name, trigger=Logger.global_trigger)
        cls.loggers.setdefault(id(logger), logger)
        return logger

Logger.loggers = {}
Logger.global_trigger = None