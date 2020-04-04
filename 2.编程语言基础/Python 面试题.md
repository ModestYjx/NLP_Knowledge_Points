# Python 面试题

tags: python

---

## Reference

[1] [整理的最全 python常见面试题（基本必考）](https://blog.csdn.net/zhusongziye/article/details/80382932)

[2] [最常见的 35 个 Python 面试题及答案（2018 版）](https://www.leiphone.com/news/201808/4DWu6VsBtjvNanNi.html)

[3] [2018年最常见的Python面试题&答案（上篇）](https://juejin.im/post/5b6bc1d16fb9a04f9c43edc3)：和[2]有重复

[4] [Python 直接赋值、浅拷贝和深度拷贝解析](https://www.runoob.com/w3cnote/python-understanding-dict-copy-shallow-or-deep.html)



python数据类型有哪些：

有五个标准的数据类型：
Numbers（数字）
String（字符串）
List（列表）
Tuple（元组）
Dictionary（字典）

### 1. 垃圾回收机制

**引用计数为主，分代收集为辅。**在 python 中，如果一个对象的引用数为 0， python 虚拟机就会回收这个对象的内存。

- 导致引用计数 +1 的情况：

  > - 对象被创建： a = classname()
  > - 对象被引用： b = a
  > - 对象被作为参数，传入到一个函数中： func(a)
  > - 对象作为一个元素，存储在容器内： list_name = [a, a]

- 导致引用计数 -1 的情况：

  > - 对象的别名被显式销毁， 如： del a 
  > - 对象的别名被赋予新的对象，如：a = other_class()
  > - 一个对象离开它的作用域，如函数执行完毕时，func 函数中的局部变量
  > - 对象所在的容器被销毁，或从容器中删除对象

**循环引用导致内存泄漏**

```
c1=ClassA() # 内存 1 引用计数 +1 = 1
c2=ClassA() # 内存 2 引用计数 +1 = 1
c1.t=c2  #  内存 2 引用计数 +1 = 2
c2.t=c1  #  内存 1 引用计数 +1 = 2
del c1  # 内存 1 引用计数 -1 = 1
del c2  # 内存 2 引用计数 -1 = 1
```

如上文描述，由于**循环引用**，导致垃圾回收器都不会回收它们，所以就会导致内存泄露。

**垃圾回收机制模块： gc**