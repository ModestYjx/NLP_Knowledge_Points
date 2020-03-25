[1] [60题PyTorch简易入门指南](https://zhuanlan.zhihu.com/p/99318332)

1. torch.randn_like()

2. item()  # 从张量中取出数字

3. numpy操作

   ```python
   import torch
   
   # a = torch.ones(5)
   # print(a)
   #
   # b = a.numpy()
   # print(b)
   #
   # a.add_(1)
   # print(a)
   # print(b)
   '''
   tensor([1., 1., 1., 1., 1.])
   [1. 1. 1. 1. 1.]
   tensor([2., 2., 2., 2., 2.])
   [2. 2. 2. 2. 2.]
   '''
   
   
   import numpy as np
   a = np.ones(5)
   b = torch.from_numpy(a)
   print(a)
   print(b)
   
   np.add(a, 1, out=a)
   print(a)
   print(b)
   '''
   [1. 1. 1. 1. 1.]
   tensor([1., 1., 1., 1., 1.], dtype=torch.float64)
   [2. 2. 2. 2. 2.]
   tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
   '''
   ```

4. 自动微分

   ```python
   import torch
   
   x = torch.ones(2, 2, requires_grad=True)
   print(x)
   
   y = x + 2
   print(y)
   print(y.grad_fn) # y就多了一个AddBackward
   
   z = y * y * 3
   out = z.mean()
   
   print(z) # z多了MulBackward
   print(out) # out多了MeanBackward
   
   '''
   tensor([[1., 1.],
           [1., 1.]], requires_grad=True)
   tensor([[3., 3.],
           [3., 3.]], grad_fn=<AddBackward0>)
   <AddBackward0 object at 0x000001C56D207BE0>
   tensor([[27., 27.],
           [27., 27.]], grad_fn=<MulBackward0>)
   tensor(27., grad_fn=<MeanBackward0>)
   '''
   ```

5. [tensor.detach() 和 tensor.data 的区别](https://blog.csdn.net/DreamHome_S/article/details/85259533)

6. 存取模型

   58.保存训练好的模型

   ```python3
   PATH = './cifar_net.pth'
   torch.save(net.state_dict(), PATH)
   ```

   59.读取保存的模型

   ```python3
   pretrained_net = torch.load(PATH)
   ```

   60.加载模型

   ```python3
   net3 = Net()
   
   net3.load_state_dict(pretrained_net)
   ```