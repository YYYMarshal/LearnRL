import paddle.fluid as fluid
import paddle
import numpy as np
import matplotlib.pyplot as plt
import os


# https://www.bilibili.com/video/BV1yv411i7xd?t=468.2&p=9
def tutorials():
    # 生成数据
    np.random.seed(0)
    outputs = np.random.randint(5, size=(10, 4))
    print("outputs = ", outputs)
    res = []
    for i in range(10):
        # 假设方程为 y = 4a+6b+7c+2d
        y = 4 * outputs[i][0] + 6 * outputs[i][1] + 7 * outputs[i][2] + 2 * outputs[i][3]
        res.append(y)
    # 定义数据
    train_data = np.array(outputs).astype("float32")
    y_true = np.array(res).astype("float32")
    # 定义网络
    x = fluid.layers.data(name="x", shape=[4], dtype="float32")
    y = fluid.layers.data(name="y", shape=[4], dtype="float32")
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    # 定义损失函数
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)
    # 定义优化方法
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.05)
    sgd_optimizer.minimize(avg_cost)
    # 参数初始化
    cpu = fluid.CPUPlace()
    exe = fluid.Executor(cpu)
    exe.run(fluid.default_startup_program())
    # 开始训练，迭代500次
    for i in range(500):
        outs = exe.run(
            feed={"x": train_data, "y": y_true},
            fetch_list=[y_predict.name, avg_cost.name]
        )
        if i % 50 == 0:
            print(f"iter = {i}, cost = {outs[1][0]}")


# https://aistudio.baidu.com/aistudio/projectdetail/6584106
def addition_of_constants():
    # 定义两个张量的常量x1和x2，并指定它们的形状是[2, 2]，并赋值为1铺满整个张量，类型为int64.
    # 定义两个张量
    x1 = fluid.layers.fill_constant(shape=[2, 2], value=1, dtype='int64')
    x2 = fluid.layers.fill_constant(shape=[2, 2], value=1, dtype='int64')
    """
    定义一个操作，该计算是将上面两个张量进行加法计算，并返回一个求和的算子。
    PaddlePaddle提供了大量的操作，比如加减乘除、三角函数等，读者可以在fluid.layers找到。
    """
    # 将两个张量求和
    y1 = fluid.layers.sum(x=[x1, x2])
    # print("x1 = ", x1)
    # print("x2 = ", x2)
    # print("y1 = ", y1)
    """
    然后创建一个解释器，可以在这里指定计算使用CPU或GPU。当使用CPUPlace()时使用的是CPU，
    如果是CUDAPlace()使用的是GPU。解析器是之后使用它来进行计算过的，比如在执行计算之前
    我们要先执行参数初始化的program也是要使用到解析器的，因为只有解析器才能执行program。
    """
    # 创建一个使用CPU的解释器
    place = fluid.CPUPlace()
    exe = fluid.executor.Executor(place)
    # 进行参数初始化
    # exe.run(fluid.default_startup_program())
    """
    最后执行计算，program的参数值是主程序，不是上一步使用的是初始化参数的程序，
    program默认一共有两个，分别是default_startup_program()和default_main_program()。
    fetch_list参数的值是在解析器在run之后要输出的值，我们要输出计算加法之后输出结果值。
    最后计算得到的也是一个张量。
    """
    # 进行运算，并把y的结果输出
    result = exe.run(program=fluid.default_main_program(), fetch_list=[y1])
    print(result)


def addition_of_variables():
    # 定义两个张量，并不指定该张量的形状和值，它们是之后动态赋值的。
    # 这里只是指定它们的类型和名字，这个名字是我们之后赋值的关键。
    a = fluid.layers.create_tensor(dtype='int64', name='a')
    b = fluid.layers.create_tensor(dtype='int64', name='b')
    # 将两个张量求和，使用同样的方式，定义这个两个张量的加法操作。
    y = fluid.layers.sum(x=[a, b])
    # 创建一个使用CPU的解释器，这里我们同样是创建一个使用CPU的解析器，和进行参数初始化。
    place = fluid.CPUPlace()
    exe = fluid.executor.Executor(place)
    # 进行参数初始化
    # exe.run(fluid.default_startup_program())
    # 定义两个要计算的变量，然后使用numpy创建两个张量值，之后我们要计算的就是这两个值。
    a1 = np.array([3, 2]).astype('int64')
    b1 = np.array([1, 1]).astype('int64')
    # 这次exe.run() 的参数有点不一样了，多了一个feed参数，这个就是
    # 要对张量变量进行赋值的。赋值的方式是使用了键值对的格式，key是
    # 定义张量变量是指定的名称，value就是要传递的值。在fetch_list参数中，
    # 笔者希望把a, b, y的值都输出来，所以要使用3个变量来接受返回值。

    # 进行运算，并把y的结果输出
    out_a, out_b, result = exe.run(
        program=fluid.default_main_program(),
        feed={'a': a1, 'b': b1},
        fetch_list=[a, b, y])
    print(out_a, " + ", out_b, " = ", result)


def linear_regression():
    # 定义一个简单的线性网络
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    hidden = fluid.layers.fc(input=x, size=100, act='relu')
    net = fluid.layers.fc(input=hidden, size=1, act=None)
    # 定义损失函数
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    cost = fluid.layers.square_error_cost(input=net, label=y)
    avg_cost = fluid.layers.mean(cost)
    # 复制一个主程序，方便之后使用
    test_program = fluid.default_main_program().clone(for_test=True)
    # 定义优化方法
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01)
    optimizer.minimize(avg_cost)

    # 创建一个使用CPU的解释器
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    # 进行参数初始化
    exe.run(fluid.default_startup_program())
    # 定义训练和测试数据
    x_data = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).astype('float32')
    y_data = np.array([[3.0], [5.0], [7.0], [9.0], [11.0]]).astype('float32')
    test_data = np.array([[6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).astype('float32')
    # 开始训练100个pass
    for pass_id in range(10):
        train_cost = exe.run(program=fluid.default_main_program(),
                             feed={'x': x_data, 'y': y_data},
                             fetch_list=[avg_cost])
        print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0]))
    # 开始预测
    result = exe.run(program=test_program,
                     feed={'x': test_data, 'y': np.array([[0.0]]).astype('float32')},
                     fetch_list=[net])
    print("当x为6.0时，y为：%0.5f" % result[0][0][0])


BUF_SIZE = 500
BATCH_SIZE = 20
EPOCH_NUM = 50


def main():
    # tutorials()
    # addition_of_constants()
    addition_of_variables()
    # linear_regression()


if __name__ == '__main__':
    main()
