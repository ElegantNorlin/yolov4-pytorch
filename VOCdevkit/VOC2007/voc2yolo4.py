#----------------------------------------------------------------------#
#   验证集的划分在train.py代码里面进行
#   test.txt和val.txt里面没有内容是正常的。训练不会使用到。
#----------------------------------------------------------------------#
'''
#--------------------------------注意----------------------------------#
如果在pycharm中运行时提示：
FileNotFoundError: [WinError 3] 系统找不到指定的路径。: './VOCdevkit/VOC2007/Annotations'
这是pycharm运行目录的问题，最简单的方法是将该文件复制到根目录后运行。
可以查询一下相对目录和根目录的概念。在VSCODE中没有这个问题。
#--------------------------------注意----------------------------------#
'''

'''
写在程序的开始，希望你能看到这些提示：
这个程序是划分训练集、训练时的测试集以及test测试集
默认是将数据集的训练集和测试集中的所有图片放入"./JPEGImages"文件夹下

如果你不想按照上述方法进行划分数据集，那么可以手动的生成test.txt和2007_test.txt
貌似2007_test.txt这个文件没有用到
'''

import os
import random 
random.seed(0)

xmlfilepath=r'./VOCdevkit/VOC2007/Annotations'
saveBasePath=r"./VOCdevkit/VOC2007/ImageSets/Main/"
 
#----------------------------------------------------------------------#
#   想要增加测试集修改trainval_percent
#   train_percent不需要修改
#----------------------------------------------------------------------#
# 训练时测试集的图片数量，最后生成的训练集和训练时的测试集是一样的，剩下的就收test测试集
trainval_percent=1
# 训练集图片的数量，保持1不变，通过调整trainval_percent来调整训练集的数量
train_percent=1

temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

num=len(total_xml)  
list=range(num)  
tv=int(num*trainval_percent)  
tr=int(tv*train_percent)  
# sample()方法返回一个列表，其中从序列中随机选择指定数量的项目。并且不会改变原有数组的序列
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  
 
print("train and val size",tv)
print("traub suze",tr)
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
 
# 最后生成的训练集和训练时的测试集时一摸一样的，剩下的就是test测试集
for i  in list:  
    name=total_xml[i][:-4]+'\n'  
    if i in trainval:  
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        else:  
            fval.write(name)  
    else:  
        ftest.write(name)  
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()
