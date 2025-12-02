import re
import os
import shutil
import tempfile
from collections import defaultdict
from tqdm import tqdm
import matlab.engine
from pathlib import Path
import numpy as np
from skimage.io import imread


def cidre_correct(in_dir,out_dir,fmt='tif',**cidre_params):
    """Corrects the images in the input directory using CIDRE.

    Parameters
    ----------
    in_dir : str
        Path to the input directory.
    out_dir : str
        Path to the output directory.
    fmt : str
        Format of the images.
    **cidre_params : dict
        Optional CIDRE parameters:
        - lambda_v: 空间正则化强度 (默认=6.0)
        - lambda_z: 零光正则化强度 (默认=0.5)
        - correction_mode: 校正模式 0/1/2 (默认=2)
        - q_percent: 计算Q的数据比例 (默认=0.25)
        - z_limits: 零光表面限制 [最小值, 最大值]
        - bit_depth: 图像位深度 (2^8, 2^12, 或 2^16)
        - max_lbfgs_iterations: 最大迭代次数 (默认=500)

    Returns
    -------
    None.

    """
    eng = matlab.engine.start_matlab()
    cidre_path = (Path(__file__).parent)/'CIDRE'
    eng.addpath(str(cidre_path))
    
    # 如果提供了参数，使用 cidre_with_params，否则使用 cidre_silent
    if cidre_params:
        # 构建 MATLAB 参数结构体
        # 注意：所有数值参数必须转换为 MATLAB 的 double 类型，以确保兼容性
        params = eng.struct()
        for key, value in cidre_params.items():
            if key == 'z_limits' and isinstance(value, (list, tuple)):
                # z_limits 需要是 MATLAB 数组
                params[key] = matlab.double(value)  
            elif key == 'bit_depth':
                # bit_depth 必须是实际的最大整数值，使用 double 类型（MATLAB 的 log2 需要 double）
                params[key] = matlab.double([float(value)])
            elif isinstance(value, (int, float)):
                # 其他数值参数转换为 MATLAB double
                params[key] = matlab.double([float(value)])
            else:
                params[key] = value
        eng.cidre_with_params(str(Path(in_dir)/f'*.{fmt}'),out_dir,params,nargout=0)
    else:
        eng.cidre_silent(str(Path(in_dir)/f'*.{fmt}'),out_dir,nargout=0)
    
    eng.quit()
    # 删除 CIDRE 生成的模型文件
    model_file = Path(out_dir)/'cidre_model.mat'
    if model_file.is_file():
        os.remove(model_file)


def cidre_walk(in_dir,out_dir,fmt='tif',lambda_v=9.0,lambda_z=1.5,correction_mode=0,
               q_percent=0.12,bit_depth=65536,max_lbfgs_iterations=400,z_limits=None,
               exclude_bright_images=True,bright_threshold_percentile=95.0,exclude_last_n_cycles=None):
    """Corrects images grouped by channel (not by cyc_chn) using CIDRE.
    
    This function groups all cycles of the same channel together and builds
    a single CIDRE model for each channel, then applies the correction to
    maintain the original directory structure (cyc_X_channel).

    Parameters
    ----------
    in_dir : str
        Path to the input directory containing cyc_*_* subdirectories.
    out_dir : str
        Path to the output directory.
    fmt : str
        Format of the images (default='tif').
    lambda_v : float, optional
        空间正则化强度，控制照明校正的平滑程度 (默认=8.0，针对抗过拟合)
        较高值可减少对局部强样本背景的响应，避免过拟合
    lambda_z : float, optional
        零光正则化强度，控制背景均匀性 (默认=1.2)
        用于强制背景更均匀，减少对复杂样本背景的拟合
    correction_mode : int, optional
        校正模式 0/1/2 (默认=0，保持绝对强度值)
        - 0: 保持零光，保留原始强度范围和零光水平（用于定量分析）
        - 1: 动态范围校正，保留原始强度范围
        - 2: 直接校正，最彻底的校正
    q_percent : float, optional
        计算Q的数据比例，控制鲁棒性 (默认=0.18，更鲁棒)
        较小值更鲁棒，可降低强样本背景的影响
    bit_depth : int, optional
        图像位深度，作为最大整数值 (默认=65536，即2^16，表示16位图像)
        可选值: 256 (2^8, 8位), 4096 (2^12, 12位), 65536 (2^16, 16位)
    max_lbfgs_iterations : int, optional
        最大迭代次数 (默认=400)
        平衡计算速度和校正效果
    z_limits : list or tuple, optional
        零光表面限制 [最小值, 最大值] (默认=None，自动检测)
        可通过拍摄暗帧图像来确定背景范围
    exclude_bright_images : bool, optional
        是否自动排除超亮的图像 (默认=True)
        如果为 True，会计算所有图像的平均强度，排除超过指定百分位数的图像
    bright_threshold_percentile : float, optional
        超亮图像的阈值百分位数 (默认=95.0)
        平均强度超过此百分位数的图像会被排除，避免过拟合
    exclude_last_n_cycles : int, optional
        排除最后 N 个 cycle 的图像 (默认=None)
        如果指定，会排除最后 N 个 cycle 的所有图像（例如 exclude_last_n_cycles=3 会排除最后 3 个 cycle）

    Returns
    -------
    None.

    Notes
    -----
    - 默认参数已进一步优化，针对强样本背景带来的过拟合问题
    - lambda_v=9.0, lambda_z=1.5, q_percent=0.12 提供更强的抗过拟合能力
    - 同一 channel 的所有 cycle 图像会一起建一个 CIDRE model
    - 输出目录结构保持与输入相同（cyc_X_channel）
    - 如果仍有 artifact，可以：
      1. 进一步提高 lambda_v 到 9.5
      2. 降低 q_percent 到 0.10
      3. 启用 exclude_bright_images=True 自动过滤超亮图像
      4. 使用 exclude_last_n_cycles 排除有问题的 cycle

    Examples
    --------
    >>> # 使用默认推荐参数（抗过拟合，自动过滤超亮图像）
    >>> cidre_walk('input_dir', 'output_dir')
    
    >>> # 排除最后 3 个 cycle 的图像
    >>> cidre_walk('input_dir', 'output_dir', exclude_last_n_cycles=3)
    
    >>> # 自定义参数，更激进的抗过拟合
    >>> cidre_walk('input_dir', 'output_dir', 
    ...            lambda_v=9.5, lambda_z=2.0, q_percent=0.10)
    
    >>> # 指定背景范围，并排除超亮图像（使用更严格的阈值）
    >>> cidre_walk('input_dir', 'output_dir', 
    ...            z_limits=[95, 110], bright_threshold_percentile=90.0)
    
    >>> # 禁用自动过滤（如果确定所有图像都正常）
    >>> cidre_walk('input_dir', 'output_dir', exclude_bright_images=False)

    """
    # 收集所有 cyc_*_* 目录
    p = Path(in_dir).glob('cyc_[0-9]*_*')
    pattern = r'^cyc_\d+_(\w+)$'  # 提取 channel 名称
    sub_dirs = [x for x in p if x.is_dir()]
    sub_dirs = [x for x in sub_dirs if re.match(pattern,x.name)]
    
    # 按 channel 分组
    channel_groups = defaultdict(list)
    for sub_dir in sub_dirs:
        match = re.match(pattern, sub_dir.name)
        if match:
            channel = match.group(1)
            channel_groups[channel].append(sub_dir)
    
    if not channel_groups:
        print("警告: 未找到符合 cyc_*_* 格式的目录")
        return
    
    # 确保输出目录存在
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # 启动 MATLAB 引擎
    eng = matlab.engine.start_matlab()
    cidre_path = (Path(__file__).parent)/'CIDRE'
    eng.addpath(str(cidre_path))
    
    # 对每个 channel 进行处理
    for channel, dirs in tqdm(channel_groups.items(), desc='CIDRE correcting by channel'):
        # 检查是否所有目录都已处理完成
        all_processed = True
        total_src = 0
        total_dest = 0
        
        for sub_dir in dirs:
            out_sub_dir = Path(out_dir)/sub_dir.name
            out_sub_dir.mkdir(parents=True, exist_ok=True)
            src_cnt = len(list(sub_dir.glob(f'*.{fmt}')))
            dest_cnt = len(list(out_sub_dir.glob(f'*.{fmt}')))
            total_src += src_cnt
            total_dest += dest_cnt
            if src_cnt != dest_cnt:
                all_processed = False
        
        # 如果所有图像都已处理，跳过
        if all_processed and total_src > 0:
            continue
        
        # 创建临时目录收集该 channel 所有 cycle 的图像
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            temp_input_dir = temp_path / 'input'
            temp_output_dir = temp_path / 'output'
            temp_input_dir.mkdir()
            temp_output_dir.mkdir()
            
            # 收集所有图像并应用过滤
            all_image_paths = []  # 存储 (sub_dir, img_path) 元组
            for sub_dir in dirs:
                # 如果指定了 exclude_last_n_cycles，检查是否应该排除这个 cycle
                if exclude_last_n_cycles is not None:
                    # 从目录名提取 cycle 编号：cyc_X_channel
                    match = re.match(r'^cyc_(\d+)_', sub_dir.name)
                    if match:
                        cycle_num = int(match.group(1))
                        # 找到所有 cycle 编号
                        all_cycles = sorted([int(re.match(r'^cyc_(\d+)_', d.name).group(1)) 
                                            for d in dirs if re.match(r'^cyc_(\d+)_', d.name)])
                        # 如果这个 cycle 在最后 N 个中，跳过
                        if cycle_num in all_cycles[-exclude_last_n_cycles:]:
                            continue
                
                images = list(sub_dir.glob(f'*.{fmt}'))
                for img_path in images:
                    all_image_paths.append((sub_dir, img_path))
            
            # 如果启用 exclude_bright_images，计算所有图像的平均强度并过滤
            if exclude_bright_images and all_image_paths:
                print(f"  检测 {channel} 通道的图像强度（用于过滤超亮 tile）...")
                image_means = []
                for sub_dir, img_path in tqdm(all_image_paths, desc='  读取图像强度', leave=False):
                    try:
                        img = imread(str(img_path))
                        # 计算平均强度和最大强度（最大强度更能反映超亮区域）
                        mean_intensity = float(np.mean(img))
                        max_intensity = float(np.max(img))
                        # 使用平均强度和最大强度的组合来判断（超亮 tile 通常两者都很高）
                        # 如果最大强度非常高，即使平均强度不高，也可能是超亮 tile
                        combined_score = mean_intensity * 0.7 + (max_intensity / 65536.0 * 10000) * 0.3
                        image_means.append((sub_dir, img_path, mean_intensity, max_intensity, combined_score))
                    except Exception as e:
                        print(f"  警告: 无法读取 {img_path.name}: {e}")
                        image_means.append((sub_dir, img_path, 0.0, 0.0, 0.0))
                
                # 计算阈值（基于组合分数）
                scores_only = [m[4] for m in image_means]
                threshold = np.percentile(scores_only, bright_threshold_percentile)
                
                # 过滤图像
                filtered_paths = []
                excluded_count = 0
                for sub_dir, img_path, mean_intensity, max_intensity, combined_score in image_means:
                    if combined_score <= threshold:
                        filtered_paths.append((sub_dir, img_path))
                    else:
                        excluded_count += 1
                        print(f"  排除超亮 tile: {img_path.name} (平均: {mean_intensity:.1f}, 最大: {max_intensity:.1f}, 分数: {combined_score:.1f}, 阈值: {threshold:.1f})")
                
                all_image_paths = filtered_paths
                print(f"  {channel} 通道: 保留 {len(all_image_paths)} 张图像，排除 {excluded_count} 张超亮 tile")
            
            # 创建文件链接到临时目录（优先使用硬链接，速度快且不需要管理员权限）
            # 优先级：硬链接 > 符号链接 > 复制
            image_mapping = {}  # 记录临时文件名到原始目录的映射
            
            for sub_dir, img_path in all_image_paths:
                # 创建唯一的临时文件名（包含 cycle 信息）
                temp_name = f"{sub_dir.name}_{img_path.name}"
                temp_link = temp_input_dir / temp_name
                linked = False
                
                # 方法1: 尝试硬链接（最快，不需要管理员权限，但要求同一卷）
                try:
                    os.link(str(img_path.resolve()), str(temp_link))
                    linked = True
                except (OSError, AttributeError):
                    pass
                
                # 方法2: 如果硬链接失败，尝试符号链接
                if not linked:
                    try:
                        temp_link.symlink_to(img_path.resolve())
                        linked = True
                    except (OSError, AttributeError):
                        pass
                
                # 方法3: 如果都失败，最后才复制（最慢）
                if not linked:
                    shutil.copy2(img_path, temp_link)
                
                image_mapping[temp_name] = (sub_dir, img_path.name)
            
            # 调用 CIDRE 对临时目录中的所有图像进行校正
            # 构建 MATLAB 参数结构体（使用默认推荐参数）
            # 注意：所有数值参数必须转换为 MATLAB 的 double 类型，以确保兼容性
            params = eng.struct()
            params['lambda_v'] = matlab.double([float(lambda_v)])  # 空间正则化强度
            params['lambda_z'] = matlab.double([float(lambda_z)])  # 零光正则化强度
            params['correction_mode'] = matlab.double([int(correction_mode)])  # 校正模式：0/1/2
            params['q_percent'] = matlab.double([float(q_percent)])  # 鲁棒均值比例
            # bit_depth 必须是实际的最大整数值：256 (8位), 4096 (12位), 65536 (16位)
            # 使用 double 类型，因为 MATLAB 的 log2 函数需要 double
            params['bit_depth'] = matlab.double([float(bit_depth)])
            params['max_lbfgs_iterations'] = matlab.double([int(max_lbfgs_iterations)])  # 最大迭代次数
            if z_limits is not None:
                params['z_limits'] = matlab.double(z_limits)  # 零光限制 [最小值, 最大值]
            
            eng.cidre_with_params(
                str(temp_input_dir / f'*.{fmt}'),
                str(temp_output_dir),
                params,
                nargout=0
            )
            
            # 将校正后的图像复制回对应的输出目录
            for temp_name, (orig_dir, orig_name) in image_mapping.items():
                corrected_img = temp_output_dir / temp_name
                if corrected_img.exists():
                    out_sub_dir = Path(out_dir) / orig_dir.name
                    out_sub_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(corrected_img, out_sub_dir / orig_name)
            
            # 删除临时输出目录中的模型文件（如果存在）
            model_file = temp_output_dir / 'cidre_model.mat'
            if model_file.exists():
                os.remove(model_file)
    
    eng.quit()
    

def main():
    pass


if __name__ == '__main__':
    main()