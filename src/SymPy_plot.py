import sympy as sp
from sympy.plotting import plot, plot_implicit, plot3d_parametric_surface

# Problem 1: 使用 plot 绘制 cos(tan(pi*x))
def problem1():
    x = sp.symbols('x')
    expr = sp.cos(sp.tan(sp.pi * x))
    # TODO: 完成绘图，区间[-1, 1]
    plot(expr, (x, -1, 1), xlabel='x', ylabel='cos(tan(pi*x))',
         title='Problem 1: Plot of cos(tan(pi*x))')

# Problem 2: 使用 plot_implicit 绘制隐函数 e^y + cos(x)/x + y = 0
def problem2():
    x, y = sp.symbols('x y')
    expr = sp.exp(y) + sp.cos(x)/x + y
    # TODO: 完成隐函数绘图，选择合适的绘图区间防止除零错误，例如 x 从 0.1 开始
    plot_implicit(expr, (x, -10, 10), (y, -10, 10),
                  xlabel='x', ylabel='y',
                  title='Problem 2: Implicit plot of exp(y)+cos(x)/x+y=0',
                  points=500)

# Problem 3: 使用 plot3d_parametric_surface 绘制参数曲面
def problem3():
    s, t = sp.symbols('s t')
    x = sp.exp(-s)*sp.cos(t)
    y = sp.exp(-s)*sp.sin(t)
    z = t
    # TODO: 完成三维参数曲面绘图
    plot3d_parametric_surface(x, y, z, (s, 0, 8), (t, 0, 5*sp.pi),
                              xlabel='x', ylabel='y', zlabel='z',
                              title='Problem 3: Parametric Surface Plot')

if __name__ == "__main__":
    problem1()
    problem2()
    problem3()
