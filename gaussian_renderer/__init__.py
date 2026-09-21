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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建一个张量，计算2D屏幕空间坐标的平均值的梯度 形状和pc.get_xyz相同
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # 获取相机的水平视角的半正切值和垂直视角的半正切值，用来计算投影矩阵
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # 高斯光栅化设置
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),    # 图像高度
        image_width=int(viewpoint_camera.image_width),      # 图像宽度
        tanfovx=tanfovx,                                    # 水平视角的半正切值
        tanfovy=tanfovy,                                    # 垂直视角的半正切值
        bg=bg_color,                                        # 背景颜色
        scale_modifier=scaling_modifier,                    # 缩放因子
        viewmatrix=viewpoint_camera.world_view_transform,   # 视图矩阵（世界坐标系转换到相机坐标系）
        projmatrix=viewpoint_camera.full_proj_transform,    # 投影矩阵（相机坐标系转换到像素坐标系）
        sh_degree=pc.active_sh_degree,                      # 已激活球谐函数阶数
        campos=viewpoint_camera.camera_center,              # 相机位置
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    # 光栅化实例
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 3D点的位置，2D坐标，透明度
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 如果3D协方差矩阵提前计算，使用它；否则，它将从缩放 / 旋转计算而来
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        # 计算三维协方差矩阵
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 如果颜色被提供，使用它们
    # 否则，如果希望在Python中从SHs预计算颜色，则执行此操作
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python: # False 跳过不执行
            # 计算颜色
            """
            1. 重新排列pc.get_features的顺序，使其与SHs系数一致
            2. dir_pp 计算相机中心每个点的方向向量
            3. dir_pp_normalized 是将方向向量单位化
            4. 借助 shs_view 球谐函数系数+ dir_pp_normalized 单位方向向量+ eval_sh 计算颜色值
            5. 将计算的RGB颜色值加0.5,范围从[-1,1]转换到[0,1]空间
            """
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # 3d高斯 光栅化图像 渲染， 并获取它们在图像上的半径
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,                  # 3D高斯分布均值
            means2D = means2D,                  # 2D高斯分布矩阵
            dc = dc,                            
            shs = shs,                          # 球谐函数系数特征
            colors_precomp = colors_precomp,    # 预处理颜色张量
            opacities = opacity,                # 透明度
            scales = scales,                    # 尺度缩放因子
            rotations = rotations,              # 旋转矩阵
            cov3D_precomp = cov3D_precomp)      # 预处理三维协方差矩阵
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out
