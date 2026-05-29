# pku-objnav
# PKU Object Navigation

这是一个基于 YOLO 目标检测、语义先验和贝叶斯搜索的目标导航项目。项目主要用于在模拟环境中测试智能体如何根据目标物体信息进行搜索、检测和导航。

## 项目简介

本项目尝试将以下方法结合起来完成 Object Navigation 任务：

- YOLO 目标检测
- AI2-THOR 模拟环境
- 基于 metadata 的目标检测
- 语义先验建模
- 贝叶斯搜索策略
- 随机搜索与最近目标搜索 baseline
- SPUB 风格的目标导航实验

## 项目结构

```text
yolo_objnav/
├── approach_utils.py
├── batch_spub_experiments.py
├── bayes_search.py
├── build_ai2thor_prior.py
├── detection_utils.py
├── detector.py
├── metadata_detector.py
├── nav_utils.py
├── run_nearest_search_metadata.py
├── run_random_agent.py
├── run_spub_onav.py
├── run_yolo_bayes_approach.py
├── semantic_prior.py
├── sim_env.py
├── test_ai2thor_min.py
├── test_metadata_detector.py
├── test_navigate_to_point.py
├── test_reachable.py
├── test_yolo_ai2thor.py
├── test_yolo_with_prior.py
├── yolo_detector.py
├── outputs/
└── README.md
```
## 代码内容
### 目标检测
```
detector.py：检测器基础逻辑

yolo_detector.py：基于 YOLO 的目标检测器

detection_utils.py：检测结果处理工具

metadata_detector.py：基于模拟环境 metadata 的检测器
```
### 导航与环境
```
sim_env.py：模拟环境封装

nav_utils.py：导航相关工具函数

run_random_agent.py：随机搜索 baseline

run_nearest_search_metadata.py：基于 metadata 的最近目标搜索

run_spub_onav.py：SPUB 风格目标导航实验

run_yolo_bayes_approach.py：YOLO + 贝叶斯搜索导航流程
```
### 语义先验与搜索
```
semantic_prior.py：语义先验建模

build_ai2thor_prior.py：从 AI2-THOR metadata 构建物体先验

bayes_search.py：贝叶斯搜索策略

approach_utils.py：导航方法相关工具函数
```
## 环境依赖

建议使用 Python 3.9 或以上版本。
