# RoboDojo / RoboTwin 普通源码快照迁移

两个目录原先已无 submodule 登记，迁移时进一步移出各自的 `.git`；RoboTwin 内旧 CuRobo 的 `.git` 当时也一起备份。用户随后要求只保留根目录一个 Git，故所有备份中的 Git 元数据也已删除；根项目 `.git` 保持原状。

## 当前配置

- 四个上游源码统一登记在根 `pyproject.toml` 的 `[tool.soft-sim.source-repositories.*]`。
- RoboDojo 固定 `f57e1633ffed59761bc8e568cfe7530c4b33381c`，路径 `RoboDojo/`。
- RoboTwin 固定 `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`，路径 `RoboTwin/`。
- 各目录的 `.soft-sim-source.json` 记录 URL 与提交号。准备脚本拒绝覆盖未知目录或含 `.git` 的目录。
- `scripts/clone_benchmarks.sh` 保留名称供兼容；实际使用 curl 下载固定提交归档，不执行 Git 命令。
- 归档中的 `.gitmodules` 在准备时移除；IsaacLab / CuRobo 的兼容路径仍指向 `.deps/`。
- `patches/` 保留历史 Git 补丁记录，当前快照不依赖 Git apply。

## 历史备份与数据保留

备份目录：`data/backups/plain-sources-20261111-101719/`。

| 原路径 | 迁移时的临时备份（现已删除） |
|---|---|
| `RoboDojo/.git` | `robodojo.git/` |
| `RoboTwin/.git` | `robotwin.git/` |
| `RoboTwin/envs/curobo/.git` | `robotwin-legacy-curobo.git/` |

目录内仍保存迁移记录 `migration.json`、原 Git status / diff 和两个固定提交的源码归档。上表列出的 Git 元数据目录，以及更早备份中的嵌套 `.git` / submodule 元数据，已按用户要求删除；共清理 10 处，未删除源码、资产或虚拟环境。清理清单见 `data/backups/git-cleanup-*.json`。

迁移前两个 benchmark 的本地 Git 差异与已记录的 submodule 移除补丁完全一致。迁移前后 1,822 个源码/资产文件的大小、修改时间与符号链接目标清单相同；源码快照迁移时只新增来源标记、移出 Git 元数据；之后额外删除了历史备份中的 Git 元数据。已下载资产、源码、旧 `.venv` 与 `envs/` 下的新环境均保留。

## 验证

- 在空临时目录中，用原固定提交的 GitHub 同格式归档重放两个快照的首次准备和重复准备，通过。
- 验证归档 URL 去除了 `.git` 后缀、准备过程未调用 Git、归档中的 `.gitmodules` 已移除、兼容链接有效、未知目录受保护。
- 扫描两个源码目录，不含嵌套 `.git` / `.gitmodules`（不扫描已保留虚拟环境和下载资产）。
- `bash scripts/clone_benchmarks.sh all` 重复准备通过。
- `bash scripts/sync_envs.sh all --offline --dry-run` 三环境均无需改动。
- 规划/仿真坐标回归 2 项通过；之前完成的真实 FEM 抓取及双相机数据独立复核通过。

本轮归档解包验证使用备份中的相同提交归档，未声称重新完成一次全量联网安装；未重复执行物理仿真。上轮全新环境和实际 GPU/FEM 运行记录见 [干净环境验证](clean-environments-2026-11-11.md)。
