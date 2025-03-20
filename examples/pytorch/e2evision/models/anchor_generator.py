from typing import List, Tuple, Callable, Dict, Any
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
# 定义变换函数工厂类，符合工厂模式设计
class TransformFactory:
    """变换函数工厂类，用于生成不同类型的变换函数"""
    @staticmethod
    def constant(value: float = 0.0) -> Callable:
        """恒定间隔变换函数"""
        return lambda x: value
    
    @staticmethod
    def polynomial(degree: int = 1, coefficient: float = 100.0) -> Callable:
        """多项式变换函数，可配置次数和系数"""
        return lambda x: coefficient * torch.pow(x, degree)
    
    @staticmethod
    def exponential(base: float = 3.0, coefficient: float = 100.0, order: float = 1.0) -> Callable:
        """指数变换函数，可配置底数和系数"""
        return lambda x: torch.pow(base, torch.pow(x, order) * coefficient) - 1
    
    @staticmethod
    def div_x(alpha: float = 0.8, beta: float = 9, order: float = 1.0) -> Callable:
        """除以x的变换函数，可配置alpha和beta"""
        return lambda x: beta * torch.pow(x, order) / (alpha - x)
    
    @staticmethod
    def tan_x(alpha: float = 0.6, beta: float = 9, order: float = 1.0) -> Callable:
        """tan(x)除以x的变换函数，可配置alpha和beta"""
        return lambda x: beta * torch.tan(np.pi * 0.5 * torch.pow(x, order) / alpha) 

    @staticmethod
    def calculate_max_points(min_spacing: float, max_distance: float, growth_func: Callable = None, **kwargs) -> Tuple[torch.Tensor, int]:
        """
        计算在最大距离范围内，基于最小间隔和增长函数，最多能放置多少个点
        
        Args:
            min_spacing: 最小间隔距离
            max_distance: 最大距离范围
            growth_func: 增长函数，接受点的索引(从0开始)作为输入，返回该点相对于最小间隔的增量因子
                         例如，对于第i个点，其间隔为 min_spacing * (1 + growth_func(i * min_spacing / max_distance, **kwargs))
                         如果为None，则使用恒定增长因子1.0
                         
        Returns:
            points: 所有计算出的点位置的tensor
            num_points: 点的数量
        """
        if growth_func is None:
            # 默认使用恒定增长因子1.0
            growth_func = lambda i: 0.0
            
        # 设置初始点
        points = [0.0]  # 第一个点在原点
        current_pos = 0.0
        
        # 迭代添加点，直到超出最大距离
        i = 0
        while True:
            scaled_i = i * min_spacing / max_distance
            # 计算下一个点的间隔
            factor = growth_func(torch.tensor(scaled_i), **kwargs)
            assert factor >= 0, f"factor: {factor} is less than 0, with scaled_i: {scaled_i}"
            next_spacing = min_spacing * (1 + factor)
            
            # 计算下一个点的位置
            next_pos = current_pos + next_spacing
            
            # 如果超出最大距离，则停止
            if next_pos > max_distance:
                break
                
            # 添加新点
            points.append(next_pos)
            current_pos = next_pos
            i += 1
        
        # 转换为tensor
        return torch.tensor(points), len(points)

# 用工厂模式帮我实现一个 AnchorGenerator，横向的就用均匀采样，主要纵向前方和后方分别设置一个函数
class AnchorGenerator:
    """生成3D anchor boxes的工厂类"""
    
    def __init__(
        self, 
        front_points: torch.Tensor,
        back_points: torch.Tensor,
        left_y_max: float,
        right_y_max: float,
        y_interval: float = 2.0,
        z_value: float = 0.2,
        anchor_size: Tuple[float, float, float] = (4.5, 2.0, 1.5)
    ):
        """
        初始化AnchorGenerator
        
        Args:
            front_points: 前方采样点
            back_points: 后方采样点
            left_y_max: 左侧y方向最大距离
            right_y_max: 右侧y方向最大距离
            y_interval: y方向采样间隔
            z_value: z方向上采样值
            anchor_size: anchor的尺寸 (l, w, h)
        """
        # 初始化基本参数
        self.front_x_max = front_points[-1].item() if len(front_points) > 0 else 0
        self.back_x_max = back_points[-1].item() if len(back_points) > 0 else 0
        self.left_y_max = left_y_max
        self.right_y_max = right_y_max
        self.y_interval = y_interval
        self.z_value = z_value
        self.anchor_size = anchor_size
        
        # 直接设置x网格点
        self.x_grids = torch.cat([
            -back_points.flip(0),  # 反转并添加负号表示后方
            front_points
        ])
        
        # 生成y网格点，使用固定间隔
        self.y_grids = torch.arange(-left_y_max, right_y_max + y_interval/2, y_interval)

    @classmethod
    def create(
        cls,
        front_type: str = "div_x",  # 前向采样类型: constant, linear, quadratic, exponential, div_x, tan_x
        back_type: str = "div_x",      # 后向采样类型: constant, linear, quadratic, exponential, div_x, tan_x
        front_params: Dict[str, float] = None,  # 前向采样参数
        back_params: Dict[str, float] = None,   # 后向采样参数
        front_min_spacing: float = 2.5,   # 前向最小间隔
        front_max_distance: float = 200.0, # 前向最大距离
        back_min_spacing: float = 2.5,    # 后向最小间隔
        back_max_distance: float = 100.0,  # 后向最大距离
        left_y_max: float = 3.75 * 3,         # 左侧最大距离
        right_y_max: float = 3.75 * 3,        # 右侧最大距离
        y_interval: float = 3.75,         # y方向采样间隔
        z_value: float = 0.2,
        anchor_size: Tuple[float, float, float] = (5.0, 2.0, 1.5),
        **kwargs                         # 其他参数传递给基类
    ):
        """
        工厂方法创建AnchorGenerator
        
        Args:
            front_type: 前向采样类型
            back_type: 后向采样类型
            front_params: 前向采样参数字典
            back_params: 后向采样参数字典
            front_min_spacing: 前向最小间隔
            front_max_distance: 前向最大距离
            back_min_spacing: 后向最小间隔
            back_max_distance: 后向最大距离
            left_y_max: 左侧最大距离
            right_y_max: 右侧最大距离
            y_interval: y方向采样间隔
            z_value: z方向上采样值
            anchor_size: anchor的尺寸 (l, w, h)
            **kwargs: 其他参数传递给基类
        """
        # 默认参数
        default_front_params = {
            "constant": {"value": 0.0},
            "linear": {"degree": 1, "coefficient": 30.0},
            "quadratic": {"degree": 2, "coefficient": 100.0},
            "exponential": {"base": 2.0, "coefficient": 190.0, "order": 3.0},
            "div_x": {"alpha": 0.6, "beta": 9.0, "order": 1.0},
            "tan_x": {"alpha": 0.6, "beta": 9.0, "order": 1.0}
        }
        
        default_back_params = {
            "constant": {"value": 0.0},
            "linear": {"degree": 1, "coefficient": 30.0},
            "quadratic": {"degree": 2, "coefficient": 100.0},
            "exponential": {"base": 2.0, "coefficient": 190.0, "order": 3.0},
            "div_x": {"alpha": 0.6, "beta": 9.0, "order": 1.0},
            "tan_x": {"alpha": 0.6, "beta": 9.0, "order": 1.0}
        }
        
        # 使用默认参数或更新用户提供的参数
        front_params = {**default_front_params[front_type], **(front_params or {})}
        back_params = {**default_back_params[back_type], **(back_params or {})}
        
        # 创建变换函数
        transform_funcs = {
            "constant": lambda params: TransformFactory.constant(**params),
            "linear": lambda params: TransformFactory.polynomial(**params),
            "quadratic": lambda params: TransformFactory.polynomial(**params),
            "exponential": lambda params: TransformFactory.exponential(**params),
            "div_x": lambda params: TransformFactory.div_x(**params),
            "tan_x": lambda params: TransformFactory.tan_x(**params)
        }
        
        # 获取前向和后向的变换函数
        front_growth = transform_funcs[front_type](front_params)
        back_growth = transform_funcs[back_type](back_params)
        
        # 计算采样点
        front_points, _ = TransformFactory.calculate_max_points(
            front_min_spacing, front_max_distance, front_growth)
        back_points, _ = TransformFactory.calculate_max_points(
            back_min_spacing, back_max_distance, back_growth)
        
        # 创建实例
        return cls(front_points, back_points, left_y_max, right_y_max, y_interval, z_value, anchor_size, **kwargs)

    def generate_anchors(self):
        """生成所有可能的anchors组合，使用矢量化操作"""
        # 创建网格坐标
        xx, yy = torch.meshgrid(self.x_grids, self.y_grids, indexing='ij')
        
        # 展平坐标 (N个点)
        xx_flat = xx.reshape(-1, 1)  # [N, 1]
        yy_flat = yy.reshape(-1, 1)  # [N, 1]
        
        # 创建z值和尺寸
        zz = torch.ones_like(xx_flat) * self.z_value  # [N, 1]
        
        # 解包尺寸
        length, width, height = self.anchor_size
        ll = torch.ones_like(xx_flat) * length  # [N, 1]
        ww = torch.ones_like(xx_flat) * width   # [N, 1]
        hh = torch.ones_like(xx_flat) * height  # [N, 1]
        
        # 拼接所有anchor参数: [N, 6] 每行为 (x, y, z, l, w, h)
        anchors = torch.cat([xx_flat, yy_flat, zz, ll, ww, hh], dim=1)
        
        return anchors

    def visualize_anchors_bev(self, ax, alpha=0.2, linewidth=1.0):
        """
        在给定的坐标轴上绘制anchor的BEV视图
        
        Args:
            ax: matplotlib坐标轴
            alpha: 透明度
            linewidth: 线宽
        """
        # 获取anchor
        anchors = self.generate_anchors()
        
        # 解包尺寸
        length, width, height = self.anchor_size
        
        # 绘制每个anchor的BEV视图
        for anchor in anchors:
            x, y, z, l, w, h = anchor
            
            # 计算矩形的左下角坐标
            rect_x = y - w/2
            rect_y = x - l/2
            
            # 根据位置设置颜色 (前向为蓝色，后向为红色)
            color = 'blue' if x >= 0 else 'red'
            
            # 创建矩形
            rect = Rectangle(
                (rect_x, rect_y),  # 左下角坐标
                w, l,  # 宽度和长度
                linewidth=linewidth,
                edgecolor=color,
                facecolor=color,
                alpha=alpha
            )
            
            # 添加矩形到图像
            ax.add_patch(rect)
    
    def save_bev_anchor_fig(self, output_dir="./"):
        """保存BEV视图的anchor图"""
        # 创建图像
        fig, ax = plt.subplots(figsize=(10, 40))
        
        # 可视化anchors
        self.visualize_anchors_bev(ax)
        
        # 添加车辆位置标记
        ax.scatter([0], [0], color='green', s=100, marker='*', label='Vehicle')
        
        # 设置图像范围和标签
        x_min = -self.left_y_max - 3
        x_max = self.right_y_max + 3
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-self.back_x_max - 5, self.front_x_max + 5)
        
        # 设置x轴刻度为3.75的倍数
        x_ticks = np.arange(np.floor(x_min/3.75)*3.75, np.ceil(x_max/3.75)*3.75 + 3.75, 3.75)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{x:.2f}' for x in x_ticks])
        
        ax.set_xlabel('Y (Left-Right)')
        ax.set_ylabel('X (Forward-Backward)')
        ax.set_title(f'Bird\'s Eye View of Anchors')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 保存图像
        bev_path = f"{output_dir}/bev_anchors.png"
        plt.savefig(bev_path, dpi=300)
        plt.close()
        print(f"Saved BEV visualization for anchors to {bev_path}")

# 测试代码
if __name__ == "__main__":
    
    # 图片保存路径
    output_dir = "anchor_test"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nTesting calculate_max_points with different growth functions:")
    
    # 创建不同的增长函数
    growth_funcs = {
        "Constant (0.0)": TransformFactory.constant(0.0),
        "Linear (alpha*i)": TransformFactory.polynomial(1, 30),
        "Quadratic (alpha*i^2)": TransformFactory.polynomial(2, 100),
        "Exponential (3^i-1)": TransformFactory.exponential(2.0, 190.0, 3.0),
        "Div_x (beta*x/(alpha-x))": TransformFactory.div_x(alpha=0.6, beta=9, order=1.0),
        "Tan_x (beta*tan(x)/x)": TransformFactory.tan_x(alpha=0.6, beta=9, order=1.0)
    }
    
    # 设置采样参数
    min_spacing = 2.5  # 最小间隔（米）
    max_distance = 200.0  # 最大距离（米）
    
    # 创建图像用于比较不同增长函数
    plt.figure(figsize=(10, 60))
    
    # 比较不同增长函数生成的点
    for i, (name, func) in enumerate(growth_funcs.items()):
        # 计算采样点
        points, num_points = TransformFactory.calculate_max_points(min_spacing, max_distance, func)
        
        # 绘制点 - 使用torch tensor
        x_coords = torch.ones_like(points) * i
        plt.scatter(x_coords.numpy(), points.numpy(), label=f"{name} ({num_points} points)")
        
        print(f"  {name}: Generated {num_points} points with max distance {points[-1].item():.2f}m")
        
        # 标注间隔 - 调整标注位置
        for j in range(len(points) - 1):
            spacing = points[j+1].item() - points[j].item()
            plt.annotate(f"{spacing:.1f}", 
                         (i - 0.1, points[j].item() + spacing/2), 
                         fontsize=8, alpha=0.7)
    
    plt.grid(True, alpha=0.3)
    plt.title("Comparison of Sampling Points Generated by Different Growth Functions")
    plt.ylabel("Distance (m)")  # 交换标签
    plt.xlabel("Growth Function Type")  # 交换标签
    plt.legend()
    
    # 保存图像
    compare_path = f"{output_dir}/growth_functions_comparison.png"
    plt.savefig(compare_path, dpi=300)
    plt.close()
    print(f"Saved growth functions comparison to {compare_path}")
    
    # 测试不同类型的AnchorGenerator
    print("\nTesting different types of AnchorGenerator:")
    
    # 创建不同类型的生成器配置
    generator_configs = {
        "Exponential-Exponential": {
            "front_type": "exponential",
            "back_type": "exponential",
            "front_params": {"base": 2.0, "coefficient": 190.0, "order": 3.0},
            "back_params": {"base": 2.0, "coefficient": 190.0, "order": 3.0}
        },
        "Div-Div": {
            "front_type": "div_x",
            "back_type": "div_x",
            "front_params": {"alpha": 0.6, "beta": 9.0, "order": 2.0},
            "back_params": {"alpha": 0.6, "beta": 9.0, "order": 2.0}
        },
        "Tan-Tan": {
            "front_type": "tan_x",
            "back_type": "tan_x",
            "front_params": {"alpha": 0.6, "beta": 9.0, "order": 2.0},
            "back_params": {"alpha": 0.6, "beta": 9.0, "order": 2.0}
        }
    }
    
    # 为每个配置创建生成器并可视化
    for name, config in generator_configs.items():
        # 创建生成器
        generator = AnchorGenerator.create(**config)
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(10, 40))
        
        # 可视化anchors
        generator.visualize_anchors_bev(ax)
        
        # 添加车辆位置标记
        ax.scatter([0], [0], color='green', s=100, marker='*', label='Vehicle')
        
        # 设置图像范围和标签
        x_min = -generator.left_y_max - 3
        x_max = generator.right_y_max + 3
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-generator.back_x_max - 5, generator.front_x_max + 5)
        
        # 设置x轴刻度为3.75的倍数
        x_ticks = np.arange(np.floor(x_min/3.75)*3.75, np.ceil(x_max/3.75)*3.75 + 3.75, 3.75)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{x:.2f}' for x in x_ticks])
        
        ax.set_xlabel('Y (Left-Right)')
        ax.set_ylabel('X (Forward-Backward)')
        ax.set_title(f'Bird\'s Eye View of Anchors ({name})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 保存图像
        bev_path = f"{output_dir}/bev_{name.lower().replace('-', '_')}.png"
        plt.savefig(bev_path, dpi=300)
        plt.close()
        print(f"Saved BEV visualization for {name} to {bev_path}")
    
    print("Test completed successfully!")
