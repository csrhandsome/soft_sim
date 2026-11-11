# TOML 迁移回归验证：2026-11-11

执行位置：ASUS `/data/codes/grade_two/my_projects/robot/soft_sim`。
清理前初始提交：`afe61388df28763cded19cc6705e693d0f209f2f`，作者 three，无共同作者。

| 检查 | 结果 |
|---|---|
| `unittest discover -s tests -v`，启用 EGL 双相机渲染 | 18 项全部通过，未跳过 |
| `benchmarks/test_soft_grasp_frames.py` | 2 项通过，坐标容差 2e-7 |
| `benchmarks/validate_grasp_run.py` | 历史抓取轨迹和双相机数据通过 |
| `benchmarks/check_robotwin.py` | SAPIEN / mplib 成功，69 步轨迹，480×640 RGB |
| `benchmarks/check_robodojo_planner.py` | GPU CuRobo 成功，41×7 关节轨迹 |
| 新 PhysX FEM 抓取及独立结果校验 | 通过，见下方原始数据路径 |
| `bash scripts/sync_envs.sh all --locked` | 三个 TOML 项目均成功 |
| 三环境 `uv pip check` | 全部通过 |
| 干净临时 checkout 重放补丁、重复准备 | 通过，零 gitlink，配置路径存在 |
| 固定提交归档的首次解包、补丁和重复准备 | 通过，使用本地同格式归档 |
| 根仓库 / RoboDojo / RoboTwin 的 `git ls-files --stage` | 均无 160000 条目 |

## 新抓取实跑

命令见根 README。输出独立保存于 `data/runs/soft_planned_grasp_toml/`，历史 `soft_planned_grasp_55` 未覆盖。

```json
{
  "success": true,
  "max_lift_mm": 82.1414440870285,
  "final_lift_mm": 76.9258439540863,
  "min_held_lift_mm": 76.85261964797974,
  "hold_duration_s": 1.0,
  "xy_drift_mm": 30.167685337997764,
  "cameras": {
    "hold": {
      "rgb_std": [
        39.21814800736878,
        35.13238327084248
      ],
      "finite_depth_pixels": [
        65536,
        65536
      ]
    },
    "verify_hold": {
      "rgb_std": [
        54.6415391335316,
        51.350460145653756
      ],
      "finite_depth_pixels": [
        65536,
        65536
      ]
    }
  }
}
```

原始数据包括 `result.json`、`trajectory.json`、各阶段双相机 PNG / NPZ 和 `validation.json`。
日志在 `data/logs/backend-tests.log`、`robotwin-regression.log`、`robodojo-planner-toml.log`、`robodojo-grasp-toml.log`、`robodojo-grasp-validation-toml.log`。
这些仿真输出和日志留在服务器本地，由 `.gitignore` 排除。

本轮物体保持自由刚体，没有绑定或吸附；结果覆盖单物体的一次回归，包含动态软指接触与双相机有效性。
横向漂移约 30.2 mm，较历史记录的 15.7 mm 大，仍满足既有 35 mm 阈值；没有调整验证阈值。

## 依赖迁移

IsaacLab / Assets / CuRobo 的版本保持 0.54.3 / 0.2.4 / 0.0.post1.dev100；实际只重装了这三个 editable 包以更新来源。
IsaacLab 的 Starlette 兼容补丁保留，CuRobo 增加无 Git 元数据时的版本回退。
Linux 专用 TOML 使用本机 Isaac Sim 5.1 Linux wheel 的依赖元数据，规避上游不同平台 wheel 的 metadata 差异。
旧 freeze 文本与独立环境锁文件由三个 uv 项目的 TOML / uv.lock 替代。
上游 checkout 的本地移除改动由 `patches/` 可重放补丁记录；不直接提交到或推送上游。

首次网络下载验证受 GitHub 归档带宽限制，180 秒收到约 9.8 MB 后超时；没有影响当前安装与运行。准备脚本的默认总下载时限已调整为 1800 秒，可在源码 TOML 条目中用 `download-timeout` 覆盖。归档解包与兼容补丁已用本地相同提交的归档重放验证；未声称完成一次全新联网安装。
