import numpy as np
import scipy.ndimage as sim
import matplotlib.pyplot as plt

def create_small_filter():
    """
    创建3×3平均滤波器
    
    返回:
        numpy.ndarray: 3×3的滤波器矩阵，每个元素值为1/9
    """
    # 学生需要在此实现代码
    pass

def create_large_filter():
    """
    创建15×15平均滤波器
    
    返回:
        numpy.ndarray: 15×15的滤波器矩阵，每个元素值为1/225
    """
    # 学生需要在此实现代码
    pass

def process_image(input_file):
    """
    处理图像并显示原始图像和滤波后的结果
    
    参数:
        input_file (str): 输入图像文件路径
        
    功能:
        1. 读取输入图像
        2. 创建3×3和15×15平均滤波器
        3. 对图像应用两种滤波器
        4. 显示原始图像和两种滤波结果对比
        
    学生任务:
        完成以下步骤的实现代码
    """
    # 1. 读取图像 - 使用plt.imread()函数
    img = None  # 学生需要实现
    
    # 2. 创建滤波器 - 调用已实现的函数
    small_filter = None  # 学生需要调用create_small_filter()
    large_filter = None  # 学生需要调用create_large_filter()
    
    # 3. 应用卷积 - 使用sim.convolve()函数
    small_result = None  # 学生需要实现小滤波器卷积读入的图像
    large_result = None  # 学生需要实现大滤波器卷积读入的图像
    
    # 4. 显示结果 - 使用matplotlib绘制对比图
    # 创建画布
    plt.figure(figsize=(15, 5))
    
    # 显示原始图像
    plt.subplot(1, 3, 1)
    # 学生需要添加显示原始图像的代码
    plt.title('Original Image')
    
    # 显示3×3滤波结果
    plt.subplot(1, 3, 2)
    # 学生需要添加显示小滤波器结果的代码
    plt.title('3×3 Filter Result')
    
    # 显示15×15滤波结果
    plt.subplot(1, 3, 3)
    # 学生需要添加显示大滤波器结果的代码
    plt.title('15×15 Filter Result')
    
    # 调整布局并显示
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 主程序入口 - 学生需要确保data/bwCat.tif文件存在
    process_image('data/bwCat.tif')
    import numpy as np
import scipy.ndimage as sim
import matplotlib.pyplot as plt

def create_small_filter():
    """创建3×3平均滤波器"""
    return np.ones((3, 3)) / 9

def create_large_filter():
    """创建15×15平均滤波器"""
    return np.ones((15, 15)) / (15*15)

def process_image(input_file):
    """处理图像并保存结果"""
    # 读取图像
    img = plt.imread(input_file)
    
    # 创建滤波器
    small_filter = create_small_filter()
    large_filter = create_large_filter()
    
    # 应用卷积
    small_result = sim.convolve(img, small_filter)
    large_result = sim.convolve(img, large_filter)
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(small_result, cmap='gray')
    plt.title('3×3 Filter Result')
    
    plt.subplot(1, 3, 3)
    plt.imshow(large_result, cmap='gray')
    plt.title('15×15 Filter Result')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_image('data/bwCat.tif')
