# DP
跟着李沐动手学深度学习

## 2. 预备知识

### 2.1 数据操作

### 2.1.1 入门

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

### 2.1.1 运算符

```python
x+y,x-y,x * y,x/y,x ** y
torch.exp()
torch.arange(12, dtype=torch.float32).reshape(3,4)
torch.cat((X,Y), dim=0)
X == Y
X.sum()
```

### 2.1.3 广播机制

```python
a + b
```

### 2.1.4 索引和切片

```python
X[-1]
X[1:3]
X[1,2] = 9
X[0:2,:]
```

### 2.1.5 节省内存

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

### 2.1.6 转换为其他python对象

```python
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)

a = torch.tensor()
a, a.item(), float(a), int(a)
```

### 2.1.7 小结

深度学习存储和操作数据的主要接口是张量（n维数组）。它提供了各种功能，包括基本数学运算、广播、索引、切片、内存节省和转换其他Python对象。

### 2.1.8 练习

1. 运行本节中的代码。将本节中的条件语句`X == Y`更改为`X < Y`或`X > Y`，然后看看你可以得到什么样的张量。

<img src="pictures\image-20250417172009201.png" alt="image-20250417172009201" style="zoom:67%;" />

2. 用其他形状（例如三维张量）替换广播机制中按元素操作的两个张量。结果是否与预期相同？

<img src="pictures\image-20250417173649340.png" alt="image-20250417173649340" style="zoom: 67%;" />

> [!IMPORTANT]
>
> 如果两个张量能够进行传播当且仅当每一个维度中存在一个张量的维度为1或者两个张量的该维度相等。
>
> 好像不能直接扩展维度。
>
> `numpy`中：`np.expand_dims(a,axis=-1)`，表示在a的最后增加一个新的维度。
>
> `torch`中：使用reshape来实现。

