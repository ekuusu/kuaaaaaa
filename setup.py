from setuptools import setup, find_packages
#用来配置一个 Python 包的安装脚本，它指定了包的名称、版本、依赖项等内容
setup(
    name="dlframe",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'flask',
        'scikit-learn',
        'numpy',
        'hmmlearn',
        'websockets',
        'pandas'
    ]
)
