# 编码板文件索引

整理日期：2026-10-06；主项目与依赖路径保留。

## 当前目录

- `MLVC_310P1/`：主项目，源码、配置、模型、build、testdata 和现有修改均保留。
- `deps/`：FFmpeg / x264 依赖及构建源码，保留原路径。
- `archive/2026-10-06/`：历史文件归档；完整原路径与新路径见其中的 `manifest.json`。

## 历史文件分类

- `archive/2026-10-06/backups/`：旧版本代码和基线程序，2 个顶层条目。
- `archive/2026-10-06/configs/`：历史测试配置及配置备份，7 个顶层条目。
- `archive/2026-10-06/datasets/`：历史测试数据和编码输出，4 个顶层条目。
- `archive/2026-10-06/experiments/`：性能分析及实验记录，5 个顶层条目。
- `archive/2026-10-06/logs/`：构建、编码及测试日志，26 个顶层条目。
- `archive/2026-10-06/packages/`：源码交付包、Git bundle、依赖安装包，15 个顶层条目。
- `archive/2026-10-06/reports/`：分析报告和运动统计，2 个顶层条目。
- `archive/2026-10-06/source/`：临时源码副本，5 个顶层条目。
- `archive/2026-10-06/tools/`：独立验证程序及对应源码，6 个顶层条目。
- `archive/2026-10-06/traces/`：历史帧 trace 和校验清单，5 个顶层条目。

## 编码启动

```bash
cd /root/workplace/grifcc/MLVC_310P1
bash scripts/bash_run_encode.sh configs/encoder.toml
```

## 查找历史结果

主项目内旧验收文档仍记录整理前的历史路径；请按归档中的路径清单查找。历史实验脚本可能包含旧绝对路径，重新运行前应修改为归档位置或按清单恢复对应目录。

本次仅分类移动历史文件，没有删除数据、提交代码或启动服务。
