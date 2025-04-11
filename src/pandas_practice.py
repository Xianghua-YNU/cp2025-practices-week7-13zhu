import pandas as pd
import matplotlib.pyplot as plt

def creat_frame():
    """
    创建一个包含学生信息的DataFrame并保存为CSV文件。
    
    该函数创建一个包含学生姓名、年龄、成绩和所在城市的数据框架，
    并将其保存为UTF-8编码的CSV文件。
    
    Returns:
        None
    """
    # 创建一个字典来模拟数据
    data = {
    '姓名': ['张三', '李四', '王五', '赵六', '陈七'],
    '年龄': [25, 30, None, 22, 28],
    '成绩': [85.5, 90.0, 78.5, 88.0, 92.0],
    '城市': ['北京', '上海', '广州', '深圳', '上海']
    }

    # 将字典转换为 DataFrame
    df = pd.DataFrame(data)

    # 将 DataFrame 保存为 CSV 文件
    df.to_csv('data/data.csv', index=False, encoding='utf-8')


def load_data():
    """任务1: 读取数据文件"""
    return pd.read_csv('data/data.csv')

def show_basic_info(data):
    """任务2: 显示数据基本信息"""
    print("数据基本信息：")
    data.info()

def handle_missing_values(data):
    """任务3: 处理缺失值"""
    missing_columns = data.columns[data.isnull().any()].tolist()
    for col in missing_columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].fillna(data[col].mean())
    return data

def analyze_statistics(data):
    """任务4: 统计分析数值列"""
    numeric_columns = data.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        mean_value = data[col].mean()
        median_value = data[col].median()
        std_value = data[col].std()
        print(f"{col} 列的均值: {mean_value}, 中位数: {median_value}, 标准差: {std_value}")

def visualize_data(data, column_name='成绩'):
    """任务5: 数据可视化"""
    data[column_name].plot.hist()
    plt.show()

def save_processed_data(data):
    """任务6: 保存处理后的数据"""
    data.to_csv('processed_data.csv', index=False)

def main():
    """主函数，执行所有数据处理流程"""
    # 1. 读取数据
    data = load_data()
    
    # 2. 显示基本信息
    show_basic_info(data)
    
    # 3. 处理缺失值
    processed_data = handle_missing_values(data.copy())
    
    # 4. 统计分析
    analyze_statistics(processed_data)
    
    # 6. 数据可视化
    visualize_data(processed_data)
    
    # 7. 保存处理后的数据
    save_processed_data(processed_data)

if __name__ == "__main__":
    main()
    
