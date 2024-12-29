# 使用 Ultralytics 进行目标计数

## 什么是目标计数？

使用[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/)进行目标计数涉及在视频和摄像头流中准确地识别和计数特定对象。YOLO11 擅长实时应用，凭借其最先进的算法和[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)能力，为诸如人群分析和监控等多种场景提供高效且精确的对象计数。

````{prf:definition}
:label: object-counting

目标计数（Object Counting）是计算机视觉领域中的基本任务，它旨在识别并计算图像或视频中的对象数量。此过程通常涉及到多个步骤，包括对象的检测、分类以及计数。在实际应用中，对象计数技术被广泛应用于人群密度估计、交通监控、野生动物调查等多个领域。

````

## 什么是目标计数？

使用[Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics/)进行对象计数涉及在视频和摄像头流中准确识别并计数特定对象。YOLOv11在实时应用中表现出色，凭借其先进的算法和[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)能力，为人群分析和监控等不同场景提供高效且精确的目标计数服务。

<table>
  <tr>
    <td align="center">
      <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Ag2e-5_NpS0"
        title="YouTube video player" frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen>
      </iframe>
      <br>
      <strong>Watch:</strong> Object Counting using Ultralytics YOLOv8
    </td>
    <td align="center">
      <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Fj9TStNBVoY"
        title="YouTube video player" frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen>
      </iframe>
      <br>
      <strong>Watch:</strong> Class-wise Object Counting using Ultralytics YOLO11
    </td>
  </tr>
</table>

## 目标计数的优势？

- **资源优化**：目标计数通过提供精确的计数，帮助实现高效的资源管理，并优化如库存管理等应用中的资源分配。
- **增强安全性**：目标计数通过准确追踪和计数实体，增强了安全监控能力，有助于主动威胁检测。
- **明智决策**：目标计数为决策提供了宝贵的见解，优化了零售、交通管理和多个其他领域的流程。

## Real World Applications

|                                                                        Logistics                                                                        |                                                                         Aquaculture                                                                          |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Conveyor Belt Packets Counting Using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/conveyor-belt-packets-counting.avif) | ![Fish Counting in Sea using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/fish-counting-in-sea-using-ultralytics-yolov8.avif) |
|                                                 Conveyor Belt Packets Counting Using Ultralytics YOLO11                                                 |                                                        Fish Counting in Sea using Ultralytics YOLO11                                                         |

使用 YOLO11 进行目标计数的示例

`````{tab-set}
````{tab-item} CLI
```bash
# Run a counting example
yolo solutions count show=True

# Pass a source video
yolo solutions count source="path/to/video/file.mp4"

# Pass region coordinates
yolo solutions count region=[(20, 400), (1080, 404), (1080, 360), (20, 360)]
```
````
````{tab-item} Python

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video/file.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model="yolo11n.pt",
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = counter.count(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```
````
````{tab-item} OBB Object Counting

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video/file.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# line or region points
line_points = [(20, 400), (1080, 400)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    show=True,
    region=line_points,
    model="yolo11n-obb.pt",
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = counter.count(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```
````
````{tab-item} Count in Polygon

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video/file.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360), (20, 400)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model="yolo11n.pt",
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = counter.count(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```
````
````{tab-item} Count in Line

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video/file.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
line_points = [(20, 400), (1080, 400)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    show=True,
    region=line_points,
    model="yolo11n.pt",
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = counter.count(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```
````
````{tab-item} Specific Classes

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video/file.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    show=True,
    model="yolo11n.pt",
    classes=[0, 1],
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = counter.count(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```
````
`````

### `ObjectCounter` 参数

`ObjectCounter` 参数表：

| 名称       | 类型    | 默认值                    | 描述                                                                          |
| --------- | ------ | ------------------------- | ---------------------------------------------------------------------------- |
| `model`   | `str`  | `None`                    | Ultralytics YOLO 模型文件的路径                                                  |
| `region`  | `list` | `[(20, 400), (1260, 400)]`  | 定义计数区域的点列表                                                           |
| `line_width` | `int`  | `2`                       | 边界框的线条粗细                                                             |
| `show`     | `bool` | `False`                    | 控制是否显示视频流的标志                                                        |
| `show_in`  | `bool` | `True`                     | 控制是否在视频流上显示进入计数的标志                                           |
| `show_out` | `bool` | `True`                     | 控制是否在视频流上显示离开计数的标志                                           |

## **常见问题解答**

### 如何使用 Ultralytics YOLO11 在视频中计数对象？

要使用 Ultralytics YOLO11 在视频中进行对象计数，您可以按照以下步骤操作：

1. 导入必要的库（`cv2`, `ultralytics`）。
2. 定义计数区域（例如，多边形、线条等）。
3. 设置视频捕获并初始化对象计数器。
4. 处理每一帧以跟踪对象并在定义的区域内进行计数。

以下是在特定区域内计数的简单示例：

```python
import cv2

from ultralytics import solutions


def count_objects_in_region(video_path, output_video_path, model_path):
    """Count objects in a specific region within a video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        im0 = counter.count(im0)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


count_objects_in_region("path/to/video.mp4", "output_video.avi", "yolo11n.pt")
```

在这个例子中，`classes_to_count=[0, 2]` 表示它统计类别为 `0` 和 `2` 的对象（例如，人和汽车）。

### 为什么在实时应用中选择 YOLO11 而不是其他[目标检测](https://www.ultralytics.com/glossary/object-detection)模型？

Ultralytics YOLO11相较于Faster R-CNN、SSD以及之前的YOLO版本，提供了以下几个优势：

1. **速度与效率**：YOLO11提供实时处理能力，非常适合需要高速推理的应用，如监控和自动驾驶。
2. **准确性**：它在目标检测和跟踪任务上提供了最先进的准确性，减少了误报的数量，并提高了系统的整体可靠性。
3. **易于集成**：YOLO11可以无缝集成到各种平台和设备中，包括移动和边缘设备，这对于现代AI应用至关重要。
4. **灵活性**：支持各种任务，如目标检测、分割和跟踪，并且可以通过配置模型来满足特定的用例需求。

查看 Ultralytics [YOLO11文档](https://docs.ultralytics.com/models/yolo11/)，深入了解其功能和性能比较。

### 可以运用YOLOv11进行高级应用，如人群分析和交通管理吗？

是的，Ultralytics YOLO11非常适合用于高级应用，如人群分析和交通管理，因为它具备实时检测能力、可扩展性和集成灵活性。其先进的功能允许在动态环境中进行高精度的对象跟踪、计数和分类。例如使用案例包括：

- **人群分析**：监控和管理大型集会，确保安全并优化人群流动。
- **交通管理**：追踪和统计车辆，分析交通模式，并实时管理拥堵情况。

