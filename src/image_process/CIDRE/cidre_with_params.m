function cidre_with_params(basedir, destdir, params)
% 带参数的CIDRE照明校正函数（静默模式）
%
% 输入:
%   basedir - 输入图像路径（支持通配符，如 '/path/*.tif'）
%   destdir - 输出目录路径
%   params  - 参数结构体，包含以下可选字段:
%             lambda_v: 空间正则化强度 (默认=6.0)
%             lambda_z: 零光正则化强度 (默认=0.5)
%             correction_mode: 校正模式 0/1/2 (默认=2)
%             q_percent: 计算Q的数据比例 (默认=0.25)
%             z_limits: 零光表面限制 [最小值, 最大值]
%             bit_depth: 图像位深度 (2^8, 2^12, 或 2^16)
%             max_lbfgs_iterations: 最大迭代次数 (默认=500)
%
% 示例:
%   params.lambda_v = 7.0;
%   params.lambda_z = 1.0;
%   cidre_with_params('/data/*.tif', '/output/', params);

% 构建参数列表
param_list = {};

if isfield(params, 'lambda_v')
    param_list{end+1} = 'lambda_v';
    param_list{end+1} = params.lambda_v;
end

if isfield(params, 'lambda_z')
    param_list{end+1} = 'lambda_z';
    param_list{end+1} = params.lambda_z;
end

if isfield(params, 'correction_mode')
    param_list{end+1} = 'correction_mode';
    param_list{end+1} = params.correction_mode;
end

if isfield(params, 'q_percent')
    param_list{end+1} = 'q_percent';
    param_list{end+1} = params.q_percent;
end

if isfield(params, 'z_limits')
    param_list{end+1} = 'z_limits';
    param_list{end+1} = params.z_limits;
end

if isfield(params, 'bit_depth')
    param_list{end+1} = 'bit_depth';
    param_list{end+1} = params.bit_depth;
end

if isfield(params, 'max_lbfgs_iterations')
    param_list{end+1} = 'max_lbfgs_iterations';
    param_list{end+1} = params.max_lbfgs_iterations;
end

% 添加destination参数
param_list{end+1} = 'destination';
param_list{end+1} = destdir;

% 静默调用CIDRE（抑制输出）
evalc("out = cidre(basedir, param_list{:});");

end






