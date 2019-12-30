# 混合场景下的文字识别与文本定位说明文档

## 简介

基于Tensorflow与Keras的文本定位与文本识别

> * 文本检测：CTPN 
> * 文本识别：DenseNet + CTC
> * 后台搭建：flask

## 环境搭建

absl-py (0.8.0)
alabaster (0.7.12)
astor (0.8.0)
asyncio (3.4.3)
atomicwrites (1.3.0)
attrs (19.1.0)
Babel (2.7.0)
backports-abc (0.5)
backports.functools-lru-cache (1.5)
backports.shutil-get-terminal-size (1.0.0)
backports.weakref (1.0.post1)
bleach (1.5.0)
boto3 (1.9.149)
botocore (1.12.149)
ccxt (1.12.136)
certifi (2018.1.18)
chardet (3.0.4)
Click (7.0)
cloudpickle (1.2.2)
configparser (3.5.0)
cycler (0.10.0)
Cython (0.24)
darkflow (1.0.0)
decorator (4.3.0)
docutils (0.14)
easydict (1.9)
entrypoints (0.2.3)
enum34 (1.1.6)
ExifRead (2.1.2)
Flask (1.1.1)
funcsigs (1.0.2)
functools32 (3.2.3.post2)
futures (3.3.0)
fuzzywuzzy (0.17.0)
gast (0.3.2)
get (0.0.39)
gluonnlp (0.6.0)
graphviz (0.8.4)
grpcio (1.24.1)
h5py (2.9.0)
haversine (2.1.1)
html5lib (0.9999999)
idna (2.6)
imageio (2.6.1)
imagesize (1.1.0)
imgaug (0.3.0)
ipykernel (4.8.2)
ipyparallel (6.2.4)
ipython (5.6.0)
ipython-genutils (0.2.0)
ipywidgets (7.2.0)
itsdangerous (1.1.0)
jieba (0.40)
Jinja2 (2.10.3)
jmespath (0.9.4)
joblib (0.11)
jsonschema (2.6.0)
jupyter (1.0.0)
jupyter-client (5.2.3)
jupyter-console (5.2.0)
jupyter-core (4.4.0)
Keras (2.0.9)
Keras-Applications (1.0.7)
Keras-Preprocessing (1.0.9)
kiwisolver (1.0.1)
lockfile (0.12.2)
lxml (4.3.3)
Markdown (3.1.1)
MarkupSafe (1.0)
mask-rcnn (2.1)
matplotlib (2.2.2)
mistune (0.8.3)
mock (3.0.5)
more-itertools (5.0.0)
mxnet-cu90 (1.4.0.post0)
nbconvert (5.3.1)
nbformat (4.4.0)
networkx (2.1)
nose (1.3.7)
notebook (5.4.1)
numpy (1.16.5)
opencv-python (3.4.3.18)
opencv-python-headless (4.1.1.26)
packaging (19.2)
pandas (0.23.4)
pandocfilters (1.4.2)
pathlib (1.0.1)
pathlib2 (2.3.0)
pbr (3.1.1)
pexpect (4.4.0)
pickleshare (0.7.4)
Pillow (5.1.0)
pip (9.0.1)
pkg-resources (0.0.0)
pluggy (0.9.0)
post (0.0.26)
prompt-toolkit (1.0.15)
protobuf (3.10.0)
ptyprocess (0.5.2)
public (0.0.65)
py (1.8.0)
pycocotools (2.0.0)
Pygments (2.2.0)
pyparsing (2.2.0)
pytesseract (0.3.0)
pytest (4.4.1)
python-dateutil (2.7.3)
python-Levenshtein (0.12.0)
pytorch-pretrained-bert (0.6.2)
pytz (2018.4)
PyWavelets (0.5.2)
PyYAML (3.12)
pyzmq (18.0.1)
qtconsole (4.3.1)
query-string (0.0.28)
regex (2019.4.14)
request (0.0.26)
requests (2.21.0)
s3transfer (0.2.0)
scandir (1.7)
scikit-image (0.14.5)
scikit-learn (0.19.1)
scipy (1.1.0)
Send2Trash (1.5.0)
setuptools (39.1.0)
Shapely (1.6.4.post2)
simplegeneric (0.8.1)
singledispatch (3.4.0.3)
six (1.12.0)
skdata (0.0.4)
sklearn (0.0)
snowballstemmer (2.0.0)
Sphinx (1.8.5)
sphinxcontrib-websupport (1.1.2)
subprocess32 (3.5.0)
tensorboard (1.10.0)
tensorflow (1.3.0)
tensorflow-gpu (1.10.1)
tensorflow-tensorboard (0.1.8)
termcolor (1.1.0)
terminado (0.8.1)
testpath (0.3.1)
tgrocery (0.1.4)
torch (0.4.1)
torchsummary (1.5.1)
torchvision (0.2.1)
tornado (5.0.2)
tqdm (4.26.0)
traitlets (4.3.2)
typing (3.7.4.1)
UNKNOWN (0.0.0)
urllib3 (1.22)
utm (0.4.2)
wcwidth (0.1.7)
Werkzeug (0.16.0)
wheel (0.33.6)
widgetsnbextension (3.2.0)
xmlutils (1.4)

参考提供的Tensorflow-gpu环境

另外CTPN网络相关依赖（主要是与画框的BBOX库相关）需要编译 这里提供的代码已经编译过了

## 如何训练模型与使用模型

具体参考[这个仓库](https://github.com/YCG09/chinese_ocr)，里面有详细的训练文档与使用文档

sigleTest.py：这个文件有使用模型的最简单的demo例子

另外：如果在使用pickle加载数据集时候出现了版本问题（py3 与 py2不兼容）可以参考[这个文档](https://blog.csdn.net/qq_33373858/article/details/83862381)

这里我已经处理好了，参考pkl_data文件夹下：changeRead2.py文件

## 如何搭建后台应用 

demo_final.py:这个文件的demo_flask方法已经封装了根据训练模型预测的方法 入参是文件的存储路径 出参是 1.文字定位的图片 2.文字识别的结果字符串

run_flask.py:这个文件封装了三个接口 1.文件上传 2.下载得到的预测文件定位图片 3.返回预测的文字识别字符串结果


另外：这里需要说明的是Keras在与flask等web框架结合时，可能会出现ValueError: Tensor Tensor("out_2/div:0", shape=(?, ?, 5990), dtype=float32) is not an element of this graph.的问题

此处我已经解决好了，如果使用者自己要重新搭建时如果再次遇到这个问题 可以参考[这个博文](https://blog.csdn.net/qq_31112205/article/details/102700427)








