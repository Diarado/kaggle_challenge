# kaggle_challenge

目前先用MART简单做了一个小测试，看看比赛的难度，环境目前比较简单，先不写requirement.txt，这边给出无脑安装手册

```
conda create --name=kaggle_env python==3.8
conda activate kaggle_env
pip install pandas
pip install polars
pip install lightgbm
pip install scikit-learn
```

train运行是训练MART的，会存到一个txt里，test是跑一个测试，然后生成.parquet格式提交的
应该还行，我自己拿训练集后1k做valid，RMSE基本就是1

反正可以先就这这个版本看看，要是效果可以就不上大模型了。

因为比赛要求输出按照他的规定，所以这个代码运行完会在当前目录生成submission.parquet
（他给的标准文档有关于这个的描述，但是我跑不动他的那个文档，我只能现在自己硬造一个，你们到时候记得改）

考虑到可能一般人的电脑无法直接打开parquet文件，我也写了个能直接生成csv的版本，目前预测结果如图所示：
![image](https://github.com/user-attachments/assets/2bd515c9-3230-482f-9cda-1d32c34b333f)
