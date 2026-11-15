# 生成三视角抓取视频

在项目根目录运行。先等待 `collect` 完成并提交录制，再用它打印的 `run_id` 生成视频：

```bash
python scripts/run.py preview --run-id <run_id> --video data/generated/videos/output.mp4
```

新采集的视频左侧是原 RoboDojo 场景的桌面相机 `cam_head`，右侧上下分别是左、右指内相机。
桌面相机沿用上游 `env_cfg/camera/camera_config.yml` 中的位姿和镜头配置，
`external_rgb` 直接读取该相机，不另建旁观相机。实际相机路径、内参及 USD 坐标系下的
camera-to-world 矩阵记录在 capture 的 calibration 中。三路画面使用同一个录制帧；
输出为 880×480，默认沿用采集帧率（本次为 20 fps）。字幕读取录制中的材料和任务结果。

桌面相机版本已重新采集，视频现位于 `data/generated/videos/tpu85a_table_camera.mp4`：

```bash
python scripts/run.py preview \
  --run-id 20261114T091003Z_general-pickup_s22_72e465 \
  --video data/generated/videos/tpu85a_table_camera.mp4
```

该版本使用 TPU 85A 西瓜红哑光外观，共 268 帧、13.4 秒、20 fps，任务判定抓取成功。
相机来源及视频检查结果见该 run 的 `reports/video_validation.json`。

下面两个历史 run 使用的是旧的旁观相机，不能用于确认桌面相机视角。
按当前存储路径重新导出的命令：

```bash
# 拓竹 TPU 90A，官方西瓜红深红端
python scripts/run.py preview \
  --run-id 20261114T082810Z_general-pickup_s22_e4c34c \
  --video data/generated/videos/tpu90a_watermelon_red.mp4

# 拓竹 TPU 85A，自定义西瓜红外观
python scripts/run.py preview \
  --run-id 20261114T084030Z_general-pickup_s22_7d4b9b \
  --video data/generated/videos/tpu85a_watermelon_red.mp4
```

MP4 与同名 JSON 统一保存到 `data/generated/videos/`；JSON 记录来源、帧数、帧率和分辨率。
`data/generated` 链接到 HIKSEMI 硬盘的 `/media/three/HIKSEMI/soft_sim_data/generated`，
视频与这些派生索引均由 `.gitignore` 忽略。该命令从录制数据合成画面，重新生成视频不需要运行 Isaac Sim。

修正桌面视角需要重新采集，不能通过 `preview` 把旧 RGB 转换成新视角。例如重录 85A：

```bash
python scripts/run.py collect --task general_pickup --seed 22 \
  --material bambu_tpu_85a --dataset-id x5_pickup85a_table_camera_v1 \
  --solver-iterations 80 --vertex-velocity-damping 20 --texture-max-resolution 512
python scripts/run.py preview --run-id <新打印的run_id> --video data/generated/videos/tpu85a_table_camera.mp4
```

软指现默认使用哑光材质，抑制镜面高光。上面两个历史 run 的 RGB 已固定，
重新运行 `preview` 不会改变其中的反光；要生成哑光效果视频，需先重新 `collect`，
再用新打印的 `run_id` 生成视频。

材料预设和源网格的组织方式见 [assets 说明](../assets/README.md)。
