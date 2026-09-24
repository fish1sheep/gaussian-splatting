#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        # 遍历 ParamGroup 实例的所有对象
        for key, value in vars(self).items():
            shorthand = False
            # 检查属性名是否已下划线开头
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            # 获取属性值的数据类型
            t = type(value)
            value = value if not fill_none else None 
            # 如果属性名已下划线开头，短名称是单短线+首字符key[0:1];
            # 长名称只能双下划线+key
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        # 遍历args的所有属性
        for arg in vars(args).items():
            # 如果属性名在ParamGroup实例中，则将其添加到group实例中
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3  # 球谐函数维度
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"   # 设置GPU
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False     # 是否使用python计算球谐函数
        self.compute_cov3D_python = False   # 是否使用python计算3D协方差矩阵
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000                # 迭代次数
        self.position_lr_init = 0.00016         # 位置初始学习率
        self.position_lr_final = 0.0000016      # 位置最终学习率
        self.position_lr_delay_mult = 0.01      # 位置学习率衰减因子
        self.position_lr_max_steps = 30_000     # 位置学习最大步数
        self.feature_lr = 0.0025                # 特征学习率
        self.opacity_lr = 0.025                 # 不透明度学习率
        self.scaling_lr = 0.005                 # 尺度学习率
        self.rotation_lr = 0.001                # 旋转学习率
        self.exposure_lr_init = 0.01        
        self.exposure_lr_final = 0.001      
        self.exposure_lr_delay_steps = 0    
        self.exposure_lr_delay_mult = 0.0   
        self.percent_dense = 0.01               # 稠密化百分比
        self.lambda_dssim = 0.2                 # DSSIM 比重
        self.densification_interval = 100       # 致密化间隔
        self.opacity_reset_interval = 3000      # 不透明度重置间隔
        self.densify_from_iter = 500            # 稠密开始迭代次数
        self.densify_until_iter = 15_000        # 稠密终止迭代次数
        self.densify_grad_threshold = 0.0002    # 稠密梯度阈值
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
