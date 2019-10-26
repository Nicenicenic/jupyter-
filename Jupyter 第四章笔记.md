# NumPy基础：数组与向量化计算

# 4.1 Numpy ndaray: 多维数组对象


```python
# 生成2*3的数组
import numpy as np

data = np.random.randn(2, 3)  # 生成2*3的随机数组
data
```




    array([[-0.88661956, -0.30930892,  0.42738391],
           [-0.10423097,  0.96392979,  1.26119425]])




```python
data.shape  # 查看形状
```




    dtype('float64')




```python
data.dtype    # 类型
```




    dtype('float64')



## 4.1.1 生成ndarray

array函数接收任意的序列型对象，生成一个新的包含传递数据的NumPy数组


```python
# 生成一个2*4的数组

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2
```




    array([[1, 2, 3, 4],
           [5, 6, 7, 8]])




```python
print(arr2.shape)  # 看形状
arr2.ndim # 几行
```

    (2, 4)
    




    2




```python
### arange是Python内建函数range的数组版：
np.arange(15).reshape(3,5)  # reshapre 就是重新定义这个数组的形状
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])



## 4.1.2 ndarrray 的数据类型


```python
#dtype 定义数据类型  #astype 转换数据类型
arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
print(arr1.dtype)
print(arr2.dtype)
```

    float64
    int32
    


```python
arr = np.array([1.2, 2, 3, 4, 5])
print(arr.dtype)

int_arr = arr.astype(np.int32)  # 将浮点型转化为整型，小数点后的部分会被消除
print(float_arr.dtype)

int_arr    
```

    float64
    int32
    




    array([1, 2, 3, 4, 5])




```python
int_arr = np.arange(10)
demo_arr = np.array([1.22, .34, .270], dtype=np.float64)

int_arr.astype(demo_arr.dtype)  # 第二个的类型浮点型，指定给第一个数组
```




    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])




```python
empty_uint32 = np.empty(8, dtype='u4')    ##？不太懂？？？
empty_uint32
```




    array([1649122640,        361,    3801155,    5570652,    6619251,
              7536754,    3014748,    6881383], dtype=uint32)



## 4.1.3 NumPy 数组算术


```python
# 挺简单，就是两个同尺寸数组之间的算术操作  如arr*arr arr+arr 对应元素相乘、加
```

## 4.1.4 基础索引与切片


```python
# 一位数组索引挺简单，和Python的列表很类似
arr = np.arange(10)
arr
arr[5]
arr[5:8]
arr[5:8] = 12
arr
```




    array([ 0,  1,  2,  3,  4, 12, 12, 12,  8,  9])




```python
# 数组的切片 是原数组的视图，即任何对于视图的修改都会反映到原数组上。
arr_slice = arr[5:8]
print(arr_slice)
arr_slice[1] = 123
arr
# 如果想要一份数组切片的拷贝为不是视图的话，必须显式地复制这个数组，如arr[5:8].copy()
```

    [ 12 123  12]
    




    array([  0,   1,   2,   3,   4,  12, 123,  12,   8,   9])




```python
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 生成3*3的数组
print(arr2d.shape)
print(arr2d)
arr2d[2]
```

    (3, 3)
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    




    array([7, 8, 9])




```python
#两种方法选择二维数组中的元素3
print(arr2d[0,2])
arr2d[0][2]
```

    3
    




    3




```python
# 生成2*2*3的三维数组
arr3d = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(arr3d)
print(arr3d[1])  # 返回一个2*3的二维数组
arr3d[1, 0]    #返回的是一个一维数组
```

    [[[ 1  2  3]
      [ 4  5  6]]
    
     [[ 7  8  9]
      [10 11 12]]]
    [[ 7  8  9]
     [10 11 12]]
    




    array([7, 8, 9])



#### 数组的切片索引


```python
print(arr2d)
arr2d[:2]  # 沿着轴0进行切片，选择了前两行
```

    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    




    array([[1, 2, 3],
           [4, 5, 6]])




```python
arr2d[:2,1:]  #选择前两行和后两列
```




    array([[2, 3],
           [5, 6]])




```python
arr2d[1, :2]  # 取第二行选择前两列
```




    array([4, 5])




```python
arr2d[:2, 2]   # 取前两行选择第三列
```




    array([3, 6])




```python
arr2d[:, :1]   # 取所有行 选择第一列  但是这样生成的是3*1的二维数组
```




    array([[1],
           [4],
           [7]])



## 4.1.5 布尔索引


```python
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
names
```




    array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'], dtype='<U4')




```python
data
```




    array([[-0.2047,  0.4789, -0.5194, -0.5557],
           [ 1.9658,  1.3934,  0.0929,  0.2817],
           [ 0.769 ,  1.2464,  1.0072, -1.2962],
           [ 0.275 ,  0.2289,  1.3529,  0.8864],
           [-2.0016, -0.3718,  1.669 , -0.4386],
           [-0.5397,  0.477 ,  3.2489, -1.0212],
           [-0.5771,  0.1241,  0.3026,  0.5238]])




```python
names == 'Bob' #会生成一个布尔值数组
```




    array([ True, False, False,  True, False, False, False])




```python
data[names == 'Bob']    # 取第一、四行
```




    array([[-0.2047,  0.4789, -0.5194, -0.5557],
           [ 0.275 ,  0.2289,  1.3529,  0.8864]])




```python
data[names == 'Bob', :2]  #选择names==Bob的行，并索引了各个列
                            #这里names==Bob不能放在axis1的位置，因为他们长度不一样
```




    array([[-0.2047,  0.4789],
           [ 0.275 ,  0.2289]])




```python
# 布尔值数组的长度必须和 数组轴索引长度一致
bool_1 = np.array([True, True, False, False])
data[:4, bool_1]    # 布尔值数组的长度必须和 数组轴索引长度一致
```




    array([[-0.2047,  0.4789],
           [ 1.9658,  1.3934],
           [ 0.769 ,  1.2464],
           [ 0.275 ,  0.2289]])




```python
names != 'Bob'  
data[~(names == 'Bob')] ## Bob之外的数据
```




    array([[ 1.9658,  1.3934,  0.0929,  0.2817],
           [ 0.769 ,  1.2464,  1.0072, -1.2962],
           [-2.0016, -0.3718,  1.669 , -0.4386],
           [-0.5397,  0.477 ,  3.2489, -1.0212],
           [-0.5771,  0.1241,  0.3026,  0.5238]])




```python
# 可以对多个布尔值条件进行联合
mask = (names == 'Bob') | (names == 'Will') # |或, &和
print(mask)
data[mask]
```

    [ True False  True  True  True False False]
    




    array([[-0.2047,  0.4789, -0.5194, -0.5557],
           [ 0.769 ,  1.2464,  1.0072, -1.2962],
           [ 0.275 ,  0.2289,  1.3529,  0.8864],
           [-2.0016, -0.3718,  1.669 , -0.4386]])




```python
data_2 = np.random.randn(7, 4)

data_2[data_2 < 0] = 0 ## 判断并赋值，这个本质上也是用了布尔值
data_2
```




    array([[0.0009, 1.3438, 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.5601],
           [0.    , 0.1198, 0.    , 0.3329],
           [0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.2863, 0.378 , 0.    ],
           [0.3313, 1.3497, 0.0699, 0.2467],
           [0.    , 1.0048, 1.3272, 0.    ]])



## 神奇索引


```python
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i 
    ## 如arr[0]=0，就是0-7行里面的第0行，全部元素等于0
arr
```




    array([[0., 0., 0., 0.],
           [1., 1., 1., 1.],
           [2., 2., 2., 2.],
           [3., 3., 3., 3.],
           [4., 4., 4., 4.],
           [5., 5., 5., 5.],
           [6., 6., 6., 6.],
           [7., 7., 7., 7.]])




```python
arr[[4, 3, 0, 6]]  ## 用列表或数组做索引 取的就是相应的行
## 这里取的是axis=0
```




    array([[4., 4., 4., 4.],
           [3., 3., 3., 3.],
           [0., 0., 0., 0.],
           [6., 6., 6., 6.]])




```python

```


```python
arr = np.arange(32).reshape((8, 4)) # 生成8*4的二维数组
arr
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27],
           [28, 29, 30, 31]])




```python
arr[[1, 5, 7, 2], [0, 3, 1, 2]]   # 多个索引数组
## （1，0）= 4
## （5，3）= 23
## （7，1）= 29
## （2，2）= 10
## 这里先取axis=0，再取axis=1
```




    array([ 4, 23, 29, 10])




```python
arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]
#先把第2、5、8、3行取出来
#冒号表示所有行都取，然后列按照0、3、1、2的顺序排
```




    array([[ 4,  7,  5,  6],
           [20, 23, 21, 22],
           [28, 31, 29, 30],
           [ 8, 11,  9, 10]])




```python
arr_1 = arr[[1,5,7,2]]
arr_1
```




    array([[ 4,  5,  6,  7],
           [20, 21, 22, 23],
           [28, 29, 30, 31],
           [ 8,  9, 10, 11]])




```python
arr_1[:, [0,3,1,2]]    # 用列表或数组做索引 这里取的就是相应的列
```




    array([[ 4,  7,  5,  6],
           [20, 23, 21, 22],
           [28, 31, 29, 30],
           [ 8, 11,  9, 10]])



## 4.1.7 数组转置和换轴


```python
arr = np.arange(15).reshape((3, 5))
arr
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])




```python
arr.T    #转置后变成了5*3的二维数组
```




    array([[ 0,  5, 10],
           [ 1,  6, 11],
           [ 2,  7, 12],
           [ 3,  8, 13],
           [ 4,  9, 14]])




```python
np.dot(arr,arr.T)    # 计算矩阵内积
```




    array([[ 30,  80, 130],
           [ 80, 255, 430],
           [130, 430, 730]])




```python
arr_7 = np.arange(24).reshape((2, 3, 4))
arr_7
# 现在是2 * 3 * 4
# 如某一个点的坐标（1，2，0）= 20
```




    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])




```python
arr_7.transpose((1, 0, 2)) 

# 现在轴有三个，0，1，2
# 现在换一下头两个
# 这就是个方块儿，怎么放的问题
# 现在是3 * 2 * 4
# 20 = （2，1，0）
```




    array([[[ 0,  1,  2,  3],
            [12, 13, 14, 15]],
    
           [[ 4,  5,  6,  7],
            [16, 17, 18, 19]],
    
           [[ 8,  9, 10, 11],
            [20, 21, 22, 23]]])




```python
arr_7.swapaxes(1, 2)
# 现在是2 * 4 * 3
# 曾经的（0，1，2）= 6 变成 （0，2，1）= 6
```




    array([[[ 0,  4,  8],
            [ 1,  5,  9],
            [ 2,  6, 10],
            [ 3,  7, 11]],
    
           [[12, 16, 20],
            [13, 17, 21],
            [14, 18, 22],
            [15, 19, 23]]])




```python

```

# 4.2 通用函数：快速的逐元素数组函数


```python

```

# 4.3 使用数组进行面向数组编程


```python
points = np.arange(-5, 5, 0.1)
## 100 equally spaced points
## 从-5到5，精度0.1，共100个数
points
```




    array([-5. , -4.9, -4.8, -4.7, -4.6, -4.5, -4.4, -4.3, -4.2, -4.1, -4. ,
           -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3. , -2.9,
           -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2. , -1.9, -1.8,
           -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1. , -0.9, -0.8, -0.7,
           -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0. ,  0.1,  0.2,  0.3,  0.4,
            0.5,  0.6,  0.7,  0.8,  0.9,  1. ,  1.1,  1.2,  1.3,  1.4,  1.5,
            1.6,  1.7,  1.8,  1.9,  2. ,  2.1,  2.2,  2.3,  2.4,  2.5,  2.6,
            2.7,  2.8,  2.9,  3. ,  3.1,  3.2,  3.3,  3.4,  3.5,  3.6,  3.7,
            3.8,  3.9,  4. ,  4.1,  4.2,  4.3,  4.4,  4.5,  4.6,  4.7,  4.8,
            4.9])




```python
xs, ys = np.meshgrid(points, points)    # 矩阵xs的行向量是向量x的简单复制
xs  
```




    array([[-5. , -4.9, -4.8, ...,  4.7,  4.8,  4.9],
           [-5. , -4.9, -4.8, ...,  4.7,  4.8,  4.9],
           [-5. , -4.9, -4.8, ...,  4.7,  4.8,  4.9],
           ...,
           [-5. , -4.9, -4.8, ...,  4.7,  4.8,  4.9],
           [-5. , -4.9, -4.8, ...,  4.7,  4.8,  4.9],
           [-5. , -4.9, -4.8, ...,  4.7,  4.8,  4.9]])




```python
z = np.sqrt(xs ** 2 + ys ** 2)
z
```




    array([[7.0711, 7.0007, 6.9311, ..., 6.8622, 6.9311, 7.0007],
           [7.0007, 6.9296, 6.8593, ..., 6.7897, 6.8593, 6.9296],
           [6.9311, 6.8593, 6.7882, ..., 6.7179, 6.7882, 6.8593],
           ...,
           [6.8622, 6.7897, 6.7179, ..., 6.6468, 6.7179, 6.7897],
           [6.9311, 6.8593, 6.7882, ..., 6.7179, 6.7882, 6.8593],
           [7.0007, 6.9296, 6.8593, ..., 6.7897, 6.8593, 6.9296]])




```python
import matplotlib.pyplot as plt

plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()

plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
```




    Text(0.5, 1.0, 'Image plot of $\\sqrt{x^2 + y^2}$ for a grid of values')




![png](output_61_1.png)


## 4.3.1 将条件逻辑作为数组操作

numpy.where函数是三元表达式 x if condition else y 的向量版本。

np.where(cond, xarr, yarr) 第二个和第三个参数也可以是标量。


```python
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

result = np.where(cond, xarr, yarr) ## 判断，如果是xarr，如果不是yarr
result
```




    array([1.1, 2.2, 1.3, 1.4, 2.5])



##  4.3.2 数学和统计方法


```python
arr13 = np.arange(20).reshape((5, 4)) ## axis0，就是5，axis1就是4
arr13
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])




```python
print(arr13.mean()) # 平均数
print(np.mean(arr13))      # 平均数
arr13.sum() # 求和
```

    9.5
    9.5
    




    190




```python
arr13.mean(1) # 每行求平均，每4个数求平均，得到5个数
```




    array([ 1.5,  5.5,  9.5, 13.5, 17.5])




```python
arr13.sum(axis=0)  # 每列求和
```




    array([40, 45, 50, 55])




```python
arr15 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])   # 生成3*3的数组
arr15 
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
arr15.cumsum(axis=0)   ## 累积求和，纵向相加,从0位开始
```




    array([[ 0,  1,  2],
           [ 3,  5,  7],
           [ 9, 12, 15]], dtype=int32)




```python
arr15.cumprod(axis=1) ##  从1未开始乘积，沿着axis=1这个轴，也就是横向累乘
```




    array([[  0,   0,   0],
           [  3,  12,  60],
           [  6,  42, 336]], dtype=int32)



## 4.3.3 布尔值数组的方法


```python
bools = np.array([False, False, True, False])
bools.any() ## 有一个True，这个值就为True
#bools.all() ## 全部为True，这个值为True
```




    True



## 4.3.4 排序


```python
arr17 = np.random.randn(6)
arr17
```




    array([-1.5491,  0.0222,  0.7584, -0.6605,  0.8626, -0.01  ])




```python
arr17.sort()
arr17     ## 1*1 从小到大排序，改变本身
```




    array([-1.5491, -0.6605, -0.01  ,  0.0222,  0.7584,  0.8626])




```python
arr18 = np.random.randn(5, 3)
arr18
```




    array([[ 0.05  ,  0.6702,  0.853 ],
           [-0.9559, -0.0235, -2.3042],
           [-0.6525, -1.2183, -1.3326],
           [ 1.0746,  0.7236,  0.69  ],
           [ 1.0015, -0.5031, -0.6223]])




```python
arr18.sort(0)  #5*3数组，沿着纵轴排序，竖着看
arr18
```




    array([[-0.9559, -1.2183, -2.3042],
           [-0.6525, -0.5031, -1.3326],
           [ 0.05  , -0.0235, -0.6223],
           [ 1.0015,  0.6702,  0.69  ],
           [ 1.0746,  0.7236,  0.853 ]])




```python
large_arr = np.random.randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))]  # 排序完了，把长度*0.05再取整的位置的元素取出来
```




    -1.5311513550102103



## 4.3.5 唯一值和其他集合逻辑


```python
#np.unique9()返回的是数组中唯一值排序后形成的数组

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
## 定义一个数组，里面有重复值
np.unique(names)
## 去掉重复值，每个留一个
```




    array(['Bob', 'Joe', 'Will'], dtype='<U4')




```python
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
## 数字也是一样的
np.unique(ints)
```




    array([1, 2, 3, 4])




```python
#np.in1d,可以检查一个数组中的值 是否在另一个数组中，并返回一个布尔值数组。

values = np.array([6, 0, 0, 3, 2, 5, 6])

np.in1d(values, [2, 3, 6]) 
# in1d（注意这是个1）检查，第一个数组中的数是不是2或3或6
```




    array([ True, False, False,  True,  True, False,  True])



# 4.4 使用数组进行文件输入和输出


```python
arr19 = np.arange(10)
np.save('some_array', arr19) # save存储，默认未压缩，后缀名.npy
```


```python
np.load('some_array.npy')   # load 读取
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
np.savez('array_archive.npz', a=arr19, b=arr19)# savez 默认未压缩，多个数组，后缀名.npz
```


```python
arch = np.load('array_archive.npz')  # load读取，是个字典
type(arch)
```




    numpy.lib.npyio.NpzFile




```python
arch['a']
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



# 4.5 线性函数


```python

```

# 4.6 伪随机数生成


```python
samples = np.random.normal(size=(4, 4))
## 4乘4，正态分布
samples
```




    array([[ 0.2059,  0.303 , -0.226 ,  1.9055],
           [ 1.4967,  0.134 , -0.5037, -0.611 ],
           [ 0.046 ,  1.0062,  0.9306, -0.1703],
           [-0.5956, -0.4638, -1.2542, -1.0777]])




```python
from random import normalvariate # 正态变量，Python函数
N = 1000000
%timeit samples = [normalvariate(0, 1) for _ in range(N)]
```

    924 ms ± 21 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    


```python
%timeit np.random.normal(size=N)# np函数比Python快了一个数量级，40倍
```

    36.2 ms ± 1.08 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    


```python
random1 = np.random.seed(1234)
random1 # 这个1234是啥意思
```


```python
rng = np.random.RandomState(1234)
rng.randn(10) ## 这该不会是个随机数表吧
```




    array([ 0.4714, -1.191 ,  1.4327, -0.3127, -0.7206,  0.8872,  0.8596,
           -0.6365,  0.0157, -2.2427])



# 4.7 示例：随机漫步

利用python进行随机漫步


```python
import random
import matplotlib.pyplot as plt
position = 0
walk = [position] # 这个是个列表 walk=[0]
steps = 1000 # 准备实验1000次
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    # random.randint 取0或1
    # 对应if False，True
    # 对应step=-1 或 1
    position += step
    # 比如第一次 0+1 = 1
    walk.append(position)
    # 得到的值放在walk里，walk=[0,1]
```


```python
plt.plot(walk[:200]) # 取前100个画出来
```




    [<matplotlib.lines.Line2D at 0x1696b36df98>]




![png](output_102_1.png)


利用np.random 模块模拟随机漫步   一次性抽取1000次投掷硬币的结果


```python
nsteps = 100
draws = np.random.randint(0, 2, size=nsteps)
# draws = array [1,1,0,0,1,,1...] 比如
draws
```




    array([0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1,
           1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1,
           1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1,
           0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0,
           1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1])




```python
steps = np.where(draws > 0, 1, -1)
# draws > 0 = [True,True,False,False,True,False,True,..]
# np.where(condition[, x, y]) 函数原型
# draws的某个位置的为true时，输出x的对应位置的元素，否则选择y对应位置的元素
# steps = [1,1,-1,-1,1,-1,1...]
steps
```




    array([-1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1,
            1, -1,  1,  1,  1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,
           -1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,
           -1, -1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1,  1,
            1, -1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,
            1, -1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1,  1])




```python
walk = steps.cumsum()
walk
```




    array([-1, -2, -1,  0,  1,  2,  3,  4,  3,  4,  3,  2,  3,  2,  3,  2,  3,
            4,  3,  4,  5,  6,  7,  6,  7,  8,  7,  6,  7,  6,  5,  4,  5,  6,
            5,  6,  7,  6,  7,  8,  9,  8,  9, 10, 11, 10, 11, 10, 11, 10, 11,
           10,  9,  8,  7,  6,  7,  8,  7,  6,  7,  6,  5,  4,  5,  6,  5,  6,
            7,  6,  5,  4,  5,  6,  5,  4,  5,  6,  5,  4,  5,  4,  5,  6,  5,
            6,  5,  4,  5,  6,  5,  6,  7,  6,  7,  6,  7,  6,  7,  8],
          dtype=int32)




```python
print(walk.min())
walk.max()
```

    -2
    




    11




```python
np.abs(walk) >= 10
```




    array([False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False,  True,  True,
            True,  True,  True,  True,  True,  True,  True, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False])




```python
(np.abs(walk) >= 10).argmax()
# 就是最大值第一次出现的位置
# 也就是Ture第一次出现的位置
```




    43




```python
plt.plot(walk[:100])
```




    [<matplotlib.lines.Line2D at 0x1696da3d6d8>]




![png](output_110_1.png)


## 4.7.1 一次性模拟多次随机漫步


```python
nwalks = 5000    #5000组实验同时做
nsteps = 1000     # 每组掷硬币1000次

draws = np.random.randint(0, 2, size=(nwalks, nsteps))  
#生成一个5000*1000的只含0，1的二维随机数组
draws
```




    array([[0, 0, 1, ..., 1, 0, 1],
           [0, 1, 1, ..., 0, 0, 0],
           [1, 1, 0, ..., 1, 1, 0],
           ...,
           [1, 0, 1, ..., 1, 1, 1],
           [0, 1, 0, ..., 0, 1, 0],
           [0, 0, 0, ..., 0, 0, 1]])




```python
steps = np.where(draws > 0, 1, -1)
steps
```




    array([[-1, -1,  1, ...,  1, -1,  1],
           [-1,  1,  1, ..., -1, -1, -1],
           [ 1,  1, -1, ...,  1,  1, -1],
           ...,
           [ 1, -1,  1, ...,  1,  1,  1],
           [-1,  1, -1, ..., -1,  1, -1],
           [-1, -1, -1, ..., -1, -1,  1]])




```python
walks = steps.cumsum(1)    # 压缩1轴，横着加
walks   # 5000*1000 的数组
```




    array([[ -1,  -2,  -1, ..., -42, -43, -42],
           [ -1,   0,   1, ..., -18, -19, -20],
           [  1,   2,   1, ...,  44,  45,  44],
           ...,
           [  1,   0,   1, ...,  10,  11,  12],
           [ -1,   0,  -1, ...,   6,   7,   6],
           [ -1,  -2,  -3, ...,  16,  15,  16]], dtype=int32)




```python
np.abs(walks) >= 30   #5000*1000
```




    array([[False, False, False, ...,  True,  True,  True],
           [False, False, False, ..., False, False, False],
           [False, False, False, ...,  True,  True,  True],
           ...,
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False]])




```python
hits30 = (np.abs(walks) >= 30).any(1)# 横着，沿axis=1方向  横着一排判断一次
hits30          #5000*1
```




    array([ True, False,  True, ...,  True,  True,  True])




```python
hits30.sum()   # 就是5000个值里面有多少个true
```




    3426




```python
crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
# walks[hits30]  这里是一个布尔索引
crossing_times
```




    array([467, 689, 681, ..., 455, 153, 657], dtype=int64)




```python
crossing_times.mean()
```




    507.92235843549327




```python
plt.plot(walks[:50,:10])
```




    [<matplotlib.lines.Line2D at 0x16911a5d358>,
     <matplotlib.lines.Line2D at 0x16911a5d4a8>,
     <matplotlib.lines.Line2D at 0x16911a5d5f8>,
     <matplotlib.lines.Line2D at 0x16911a5d748>,
     <matplotlib.lines.Line2D at 0x16911a5d898>,
     <matplotlib.lines.Line2D at 0x16911a5d9e8>,
     <matplotlib.lines.Line2D at 0x16911a5db38>,
     <matplotlib.lines.Line2D at 0x16911a5dc88>,
     <matplotlib.lines.Line2D at 0x16911a5ddd8>,
     <matplotlib.lines.Line2D at 0x16911a5df28>]




![png](output_120_1.png)



```python
steps1 = np.random.normal(loc=0, scale=0.25,size=(3, 5))
steps1
```




    array([[ 0.1316, -0.0642, -0.0793,  0.5083, -0.1354],
           [-0.2296,  0.0962,  0.1883,  0.0112, -0.6078],
           [-0.009 , -0.1319, -0.3338,  0.5177,  0.1635]])




```python

```


```python

```
