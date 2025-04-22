# DP
跟着李沐动手学深度学习

## 2. 预备知识

### 2.1 数据操作

#### 2.1.1 入门

```python
torch.arange()
torch.shape()
torch.numal()
torch.reshape()
torch.zeros()
torch.ones()
torch.randn()
torch.tensor()
```

#### 2.1.2 运算符

```python
x+y,x-y,x * y,x/y,x ** y
torch.exp()
torch.arange(12, dtype=torch.float32).reshape(3,4)
torch.cat((X,Y), dim=0)
X == Y
X.sum()
```

#### 2.1.3 广播机制

```python
a + b
```

#### 2.1.4 索引和切片

```python
X[-1]
X[1:3]
X[1,2] = 9
X[0:2,:]
```

#### 2.1.5 节省内存

```python
before = id(Y)
Y = Y + X
before == id(Y)

Z = torch.zeros_like(Y)
id(Z)
Z[:] = X + Y
id(Z)

before = id(Y)
Y += X
before == id(Y)    
```

#### 2.1.6 转换为其他python对象

```python
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)

a = torch.tensor()
a, a.item(), float(a), int(a)
```

> [!IMPORTANT]
>
> `X.numpy()` 的作用是将一个 `PyTorch` `Tensor` 转换为 `numpy` 数组。然而，**转换是浅拷贝**，并不会改变原始数据在 `Tensor` 中的类型或类型的引用。**类型** 仍然会保持为 `torch.Tensor`，这意味着 `X` 依然是一个 `Tensor`，而 `X.numpy()` 是该张量的 `numpy` 视图。
>
> - **视图** 是原始数据的不同表现形式，但它与原始数据共享同一块内存。这意味着，对视图的修改会影响到原始数据，而对原始数据的修改也会影响到视图。
>
> - 与 **副本（Copy）** 不同，副本会生成一个新的数据，修改副本不会影响原始数据。

#### 2.1.7 小结

深度学习存储和操作数据的主要接口是张量（n维数组）。它提供了各种功能，包括基本数学运算、广播、索引、切片、内存节省和转换其他Python对象。

#### 2.1.8 练习

1. 运行本节中的代码。将本节中的条件语句`X == Y`更改为`X < Y`或`X > Y`，然后看看你可以得到什么样的张量。

<img src="pictures\image-20250417172009201.png" alt="image-20250417172009201" style="zoom:67%;" />

2. 用其他形状（例如三维张量）替换广播机制中按元素操作的两个张量。结果是否与预期相同？

<img src="pictures\image-20250417173649340.png" alt="image-20250417173649340" style="zoom: 67%;" />

> [!IMPORTANT]
>
> 如果两个张量能够进行传播当且仅当**每一个维度中存在一个张量的维度为1或者两个张量的该维度相等**。
>
> 好像不能直接扩展维度。
>
> `numpy`中：`np.expand_dims(a,axis=-1)`，表示在a的最后增加一个新的维度。
>
> `torch`中：使用reshape来实现。

### 2.2 数据预处理

#### 2.2.1 读取数据集

##### `os`库的相关用法

```python
os.getcwd()
os.chdir(path)
os.listdir(path)

for root, dirs, files in os.walk("c:\\Users\\LX\\Desktop\\学习")

os.path.join()

os.path.abspath()

os.path.isdir()
os.path.isfile()
os.path.exists()

st = os.stat("c:\\Users\\LX\\Desktop\\学习\\DP\\note.md")

import stat
stat.filemode(st.st_mode)

os.chmod(path, mode)

os.mkdir()
os.makedirs()

os.remove()
os.redir()

os.getenv()
os.envrion()
```

- `__file__`：`__file__` 只有在 **“把 `.py` 文件当成脚本运行”** 时才由解释器自动注入——它保存的就是那个**脚本文件自身的路径**。

  在 `Jupyter / IPython` 交互环境 里并没有真正的 “脚本文件”，每个单元格只是即刻执行的代码块，所以解释器根本不会生成 `__file__` 这个变量。

  <img src="pictures\image-20250419165218390.png" alt="image-20250419165218390" style="zoom: 67%;" />
  
- `os.makedirs(os.path.join("..", "data"), exist_ok=True)`
  `exist_ok`:如果创建的目录已经存在设置为True就不会报错，如果设置为False就会报错。

##### 文件对象

```python
f = open("example.txt", "r")
"r"
"w"
"a"
"b"

lines = ["hello", "world"]

with open("example.txt", "w") as f:
	f.write("hello world")
    
    f.writelines(lines)
with open("example.txt", "r") as f:
    
    content = f.read()
    
    content = f.readlines()
    
# with 上下文管理器，完成操作后会自动关掉
```

- `csv`
  逗号分隔文件

##### pandas库常用方法

```python
pd.read_csv()
pd.read_excel() #xlxs文件
pd.read_sql()

df.head()
df.tail()
df.info()
df.describe()
df.shape 

df.iloc[:,:]
df.loc[:,["column1","column2"]]

df.rename(columns={"old_column:new_column"}, inplace=True)
```

#### 2.2.2 处理缺失值

```python
df.dropna(inplace=True) # 这样才会直接在原来的df上修改
df.iloc[:,1].fillna(df.iloc[:,1].mean(),inplace=True)
df["column"] = df["column"].astype(float)

inputs= pd.get_dummies(iputs, dummy_na=True)
```

#### 2.2.3 转换为张量格式

```python
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
```

<img src="pictures\image-20250421231806080.png" alt="image-20250421231806080" style="zoom:67%;" />

#### 2.2.4 小结

- `pandas`软件包是Python中常用的数据分析工具中，`pandas`可以与张量兼容。
- 用`pandas`处理缺失的数据时，我们可根据情况选择用插值法和删除法。

#### 2.2.5 练习

- 空值为`np.nan`

创建包含更多行和列的原始数据集。

1. 删除缺失值最多的列。

```python
df.drop(columns=[df.isnull().sum().idxmax()])
- df.isnull()
- df.sum()
- df.idxmax()
```

<img src="pictures\image-20250422140052684.png" alt="image-20250422140052684" style="zoom:67%;" />

2. 将预处理后的数据集转换为张量格式。

```python
df_tensored_1 = pd.get_dummies(df_missed_1, dummy_na=True)
df_tensored_2 = torch.tensor(df_tensored_1.to_numpy(dtype=float))
```

<img src="pictures\image-20250422140515241.png" alt="image-20250422140515241" style="zoom: 50%;" />

<img src="pictures\image-20250422140554969.png" alt="image-20250422140554969" style="zoom:50%;" />

