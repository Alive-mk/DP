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

### 2.1.4 索引切片

```python
X[-1]
X[1:3]
X[1,2]
X[0:2,:]
```

