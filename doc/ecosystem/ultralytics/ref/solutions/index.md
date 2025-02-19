# {mod}`ultralytics` 解决方案: 解决现实世界问题

Ultralytics 解决方案提供了 YOLO 模型的尖端应用，为各行各业提供如物体计数、模糊处理和安全系统等现实解决方案，从而提高了效率和准确性。探索 YOLOv11 在实际应用中的强大影响力。

![Ultralytics解决方案缩略图](https://github.com/ultralytics/docs/releases/download/0/ultralytics-solutions-thumbnail.avif)


精心挑选的 Ultralytics 解决方案列表，可用于创建出色的计算机视觉项目。

```{toctree}
:hidden:

../guides/object-counting
../guides/object-blurring
../guides/object-cropping
../guides/workouts-monitoring
../guides/region-counting
../guides/security-alarm-system
../guides/heatmaps
../guides/instance-segmentation-and-tracking
../guides/vision-eye
../guides/speed-estimation
../guides/distance-calculation
../guides/queue-management
../guides/parking-management
../guides/analytics
../guides/streamlit-live-inference
```

- [目标计数](../guides/object-counting) 🚀: 学习使用YOLO11进行实时目标计数。掌握在实时视频流中准确计数对象的技巧。
- [目标裁剪](../guides/object-cropping) 🚀: 使用YOLO11掌握精确从图像和视频中提取目标的目标裁剪技术。
- [目标模糊处理](../guides/object-blurring) 🚀: 应用YOLO11进行目标模糊处理，以保护图像和视频处理中的隐私。
- [锻炼监控](../guides/workouts-monitoring) 🚀: 探索如何使用YOLO11监控锻炼。学习如何实时追踪和分析各种健身程序。
- [区域内目标计数](../guides/region-counting) 🚀: 使用YOLO11在特定区域内计数目标，实现不同区域的准确检测。
- [安全报警系统](../guides/security-alarm-system) 🚀: 创建安全报警系统，当检测到新目标时触发警报。根据具体需求定制系统。
- [热图](../guides/heatmaps) 🚀: 利用检测热图来可视化矩阵中的数据强度，为计算机视觉任务提供清晰的洞察。
- [实例分割与目标跟踪](../guides/instance-segmentation-and-tracking) 🚀 NEW: 实施[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)和YOLO11的目标跟踪，实现精确的对象边界和持续监控。
- [VisionEye视图对象映射](../guides/vision-eye) 🚀: 开发模仿人眼对特定对象的聚焦的系统，增强计算机辨别和优先考虑细节的能力。
- [速度估计](../guides/speed-estimation) 🚀: 使用YOLO11和目标跟踪技术估计对象速度，这对于自动驾驶车辆和交通监控等应用至关重要。
- [距离计算](../guides/distance-calculation) 🚀: 使用YOLO11中的[边界框](https://www.ultralytics.com/glossary/bounding-box)质心计算对象之间的距离，这对于空间分析必不可少。
- [队列管理](../guides/queue-management) 🚀: 实现高效的队列管理系统，以减少等待时间并提高生产率，使用YOLO11。
- [停车管理](../guides/parking-management) 🚀: 使用YOLO11组织和指导停车区的车流，优化空间利用率和用户体验。
- [分析](../guides/analytics) 📊 NEW: 进行全面的数据分析，发现模式并做出明智决策，利用YOLO11进行描述性、预测性和规范性分析。
- [使用Streamlit的实时推理](../guides/streamlit-live-inference) 🚀: 利用YOLO11的强大功能直接通过友好的Streamlit界面在您的网络浏览器中进行实时[目标检测](https://www.ultralytics.com/glossary/object-detection)。

## 解决方案使用指南

````{admonition} 命令信息
`yolo solutions 解决方案名称 参数`:
- `solutions` 是必需的关键字。
- `解决方案名称`（可选）可以是以下之一：`['count', 'heatmap', 'queue', 'speed', 'workout', 'analytics']`。
- `参数`（可选）是自定义的 `参数=值` 对，例如 `show_in=True`，用于覆盖默认设置。

CLI 示例

```bash
yolo solutions count show=True  # 进行对象计数

yolo solutions source="path/to/video/file.mp4"  # 指定视频文件路径
```
````