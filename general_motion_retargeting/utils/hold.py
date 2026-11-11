"""I5 等包含世界根节点、人体和物体的米制 Z-up BVH 适配器。"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from .lafan_vendor import utils
from .lafan_vendor.extract import read_bvh


def load_hold_bvh_file(
    bvh_file,
    *,
    unit_scale=1.0,
    yaw_degrees=0.0,
    human_height=1.75,
):
    """读取人体全局姿态，返回 ``(frames, human_height)``。

    frames 中每帧为 {骨骼名: [位置, 四元数]}，位置单位为米，
    四元数顺序为 wxyz，与项目其他 BVH 加载器一致。

    参数：
        bvh_file: BVH 文件路径。
        unit_scale: 长度到米的换算系数。I5 使用 1.0，厘米数据用 0.01。
        yaw_degrees: 绕世界 Z 轴旋转的角度，同时作用于位置和朝向。
                     I5 默认保持原坐标；该参数用于调整水平朝向。
        human_height: 动捕演员身高，单位为米。默认 1.75 是假设值，
                      不是从蹲下、跳跃等动作姿态中测得的身高。

    使用逐关节通道解析并保留世界根节点的变换，再提取 Hips 子树。
    返回结果不含世界根及其下独立的箱体等物体。脚端位置取 Foot，
    朝向取 ToeBase。保留原始帧数，不进行重采样、缩放人体骨架或接地。
    """
    if not np.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError("unit_scale must be a finite positive number")
    if not np.isfinite(yaw_degrees):
        raise ValueError("yaw_degrees must be finite")
    if not np.isfinite(human_height) or human_height <= 0:
        raise ValueError("human_height must be a finite positive number in meters")

    data = read_bvh(bvh_file, per_joint=True)
    if data.pos.shape[0] == 0:
        raise ValueError(f"BVH contains no motion frames: {bvh_file}")

    bone_ids = {name: i for i, name in enumerate(data.bones)}
    required_bones = (
        "Hips", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"
    )
    missing = [name for name in required_bones if name not in bone_ids]
    if missing:
        raise ValueError(
            f"Hold BVH is missing required human bones: {', '.join(missing)}"
        )

    # FK 必须先包含世界根，避免过滤节点后丢失父节点变换。
    global_quats, global_positions = utils.quat_fk(
        data.quats, data.pos, data.parents
    )

    # BVH 按父节点先于子节点的顺序保存关节。
    human_ids = {bone_ids["Hips"]}
    for i, parent in enumerate(data.parents):
        if parent in human_ids:
            human_ids.add(i)
    if any(bone_ids[name] not in human_ids for name in required_bones):
        raise ValueError("Foot and ToeBase bones must be descendants of Hips")

    indices = sorted(human_ids)
    names = [data.bones[i] for i in indices]
    world_rotation = R.from_euler("z", yaw_degrees, degrees=True)
    positions = (
        global_positions[:, indices] * unit_scale
    ) @ world_rotation.as_matrix().T
    quats = utils.quat_mul(
        world_rotation.as_quat(scalar_first=True), global_quats[:, indices]
    )

    frames = []
    for frame_index in range(len(positions)):
        result = {
            name: [positions[frame_index, i].copy(), quats[frame_index, i].copy()]
            for i, name in enumerate(names)
        }
        for side in ("Left", "Right"):
            result[f"{side}FootMod"] = [
                result[f"{side}Foot"][0].copy(),
                result[f"{side}ToeBase"][1].copy(),
            ]
        frames.append(result)

    return frames, float(human_height)
