# what-is-this-garbage

## 主要功能
从手机端拍摄图片，经由Tensorflow Lite处理，给予用户识别的垃圾类别结果。
## 实现原理
### 模型训练
首先要对数据进行处理，数据集给了三个txt文件，分别是训练集，测试集，验证集，把他们分到对应文件夹，然后使用ImageDataGenerator进行加载，加载到keras进行模型的训练，模型训练好后可以使用TF转化器将keras模型转化成TF Lint模型，就是可以在手机端使用的模型数据，在手机端APP进行使用。
