# 草莓掩码与三指抓取点生成流程

本文档说明如何使用 Grounded-SAM 生成草莓掩码，并基于掩码生成三指抓取点。包含两套流程：

- 单草莓：一张图中只处理一个草莓，输出一组掩码和抓取点。
- 多草莓：一张图中处理多个草莓，按优先级排序后分别输出每个草莓的掩码和抓取点。

默认项目目录：

```bash
cd /data/data/workspace/zhangyize/Grounded-Segment-Anything-main
source .venv/bin/activate
```

如有 GPU，命令中使用：

```bash
--device cuda
```

没有 GPU 时改为：

```bash
--device cpu
```

## 一、单草莓流程

### 1. 准备输入图像

假设输入图像为：

```text
testdata/strawberry.jpg
```

输出目录示例：

```text
strawberry_outputs_all/outputs_strawberry_13
```

### 2. 使用 Grounded-SAM 生成草莓掩码

```bash
python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image testdata/strawberry.jpg \
  --output_dir strawberry_outputs_all/outputs_strawberry_13 \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "strawberry" \
  --device cuda
```

生成结果：

```text
strawberry_outputs_all/outputs_strawberry_13/
├── raw_image.jpg
├── mask_0.png
├── mask.json
├── mask.jpg
└── grounded_sam_output.jpg
```

其中：

- `raw_image.jpg`：输入原图副本。
- `mask_0.png`：第一个草莓的二值掩码。
- `mask.json`：检测框、类别、置信度等信息。
- `grounded_sam_output.jpg`：检测框和掩码叠加可视化。
- `mask.jpg`：所有掩码的可视化图。

### 3. 根据单个掩码生成三指抓取点

```bash
python tools/generate_strawberry_grasp.py \
  --mask strawberry_outputs_all/outputs_strawberry_13/mask_0.png \
  --image strawberry_outputs_all/outputs_strawberry_13/raw_image.jpg \
  --layout triangle-paired \
  --thumb-side top \
  --output-json strawberry_outputs_all/outputs_strawberry_13/grasp_points_triangle.json \
  --output-vis strawberry_outputs_all/outputs_strawberry_13/grasp_points_triangle_visualization.jpg
```

最终单草莓输出：

```text
strawberry_outputs_all/outputs_strawberry_13/
├── raw_image.jpg
├── mask_0.png
├── mask.json
├── mask.jpg
├── grounded_sam_output.jpg
├── grasp_points_triangle.json
└── grasp_points_triangle_visualization.jpg
```

这里的 `grasp_points_triangle_visualization.jpg` 里除了抓取点，也会画出方向确认线。

### 4. 单草莓抓取点 JSON 说明

`grasp_points_triangle.json` 中主要字段：

```json
{
  "grasp_points_2d": {
    "P_index": [x, y],
    "P_middle": [x, y],
    "P_thumb": [x, y]
  },
  "grasp_pair_center_2d": [x, y],
  "direction_line_2d": {
    "from_ignored_tip_vertex": [x, y],
    "to_support_midpoint": [x, y],
    "vector": [dx, dy]
  },
  "diagnostics": {
    "bbox_xywh": [x, y, w, h],
    "centroid": [x, y],
    "layout": "triangle-paired",
    "thumb_side": "top"
  }
}
```

含义：

- `P_index`：食指接触点。
- `P_middle`：中指接触点。
- `P_thumb`：拇指接触点。
- `grasp_pair_center_2d`：食指和中指夹持对的中心点。
- `direction_line_2d`：草莓方向确认线，从三角形舍弃点连到另外两个三角形顶点连线的中点。
- `bbox_xywh`：草莓掩码外接框，格式为 `[x, y, width, height]`。
- `centroid`：草莓掩码中心。

方向线在单草莓里就是抓取方向的确认线，在多草莓里每个 `target_XX` 都会单独保留这条线，同时额外生成一张整图方向确认图。

## 二、多草莓流程

多草莓流程分两步：

1. 先用 Grounded-SAM 对整张图生成多个 `mask_*.png`。
2. 再用多目标批处理脚本按优先级排序，并为每个草莓生成独立目录和抓取点。

排序规则：

1. 越靠下优先级越高，即 `center_y` 越大越优先。
2. 如果位置接近或需要进一步排序，越靠右优先级越高，即 `center_x` 越大越优先。

### 1. 准备输入图像

假设输入图像为：

```text
testdata/multi_strawberry.jpg
```

原始多掩码输出目录：

```text
strawberry_outputs_all/outputs_multi_raw_03
```

最终多目标输出目录：

```text
strawberry_outputs_all/outputs_multi_03
```

### 2. 生成多草莓 Grounded-SAM 原始掩码

```bash
python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image testdata/multi_strawberry.jpg \
  --output_dir strawberry_outputs_all/outputs_multi_raw_03 \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "strawberry" \
  --device cuda
```

生成结果示例：

```text
strawberry_outputs_all/outputs_multi_raw_03/
├── raw_image.jpg
├── mask_0.png
├── mask_1.png
├── mask_2.png
├── mask_3.png
├── mask.json
├── mask.jpg
└── grounded_sam_output.jpg
```

其中每个 `mask_i.png` 对应一个被检测到的草莓。

### 3. 按优先级生成每个草莓的抓取点

```bash
python tools/generate_multi_strawberry_grasps.py \
  --sam-output-dir strawberry_outputs_all/outputs_multi_raw_03 \
  --output-dir strawberry_outputs_all/outputs_multi_03 \
  --layout triangle-paired \
  --thumb-side top \
  --overwrite
```

输出目录结构：

```text
strawberry_outputs_all/outputs_multi_03/
├── all_grasp_points.json
├── strawberry_direction_visualization.jpg
├── target_00/
│   ├── raw_image.jpg
│   ├── mask_0.png
│   ├── mask.json
│   ├── grasp_points_triangle.json
│   └── grasp_points_triangle_visualization.jpg
├── target_01/
│   ├── raw_image.jpg
│   ├── mask_0.png
│   ├── mask.json
│   ├── grasp_points_triangle.json
│   └── grasp_points_triangle_visualization.jpg
└── target_02/
    ├── raw_image.jpg
    ├── mask_0.png
    ├── mask.json
    ├── grasp_points_triangle.json
    └── grasp_points_triangle_visualization.jpg
```

说明：

- `target_00` 是优先级最高的草莓。
- `target_01` 是第二优先级草莓。
- 每个 `target_XX/mask_0.png` 都是该目标自己的二值掩码。
- 每个 `target_XX/grasp_points_triangle.json` 都是该目标自己的三指抓取点。
- `all_grasp_points.json` 汇总所有目标的顺序、掩码框和抓取点。
- `strawberry_direction_visualization.jpg` 是整张图上的草莓方向确认图。

### 4. 草莓方向确认线

在 `triangle-paired` 布局中，脚本会先把草莓轮廓抽象成一个三角形。三角形中会有一个 `ignored_tip_vertex`，也就是当前抓取点生成时被舍弃的顶点。

方向确认线定义为：

```text
ignored_tip_vertex -> midpoint(thumb_support_vertex, pair_support_vertex)
```

也就是：

1. 找到三角形的舍弃点 `ignored_tip_vertex`。
2. 找到另外两个三角形顶点：`thumb_support_vertex` 和 `pair_support_vertex`。
3. 计算这两个顶点连线的中点。
4. 从舍弃点连到该中点，作为草莓方向确认线。

该结果会写入每个目标的：

```text
target_XX/grasp_points_triangle.json
```

字段如下：

```json
{
  "direction_line_2d": {
    "from_ignored_tip_vertex": [x, y],
    "to_support_midpoint": [x, y],
    "vector": [dx, dy]
  }
}
```

整张图的方向确认可视化会写入：

```text
strawberry_outputs_all/outputs_multi_03/strawberry_direction_visualization.jpg
```

整图方向确认图里，每个目标都会显示：

- 草莓掩码轮廓
- 三角形拟合结果
- 舍弃点到中点的方向箭头
- `target_XX` 标签

### 5. 多草莓汇总 JSON 说明

`all_grasp_points.json` 主要结构：

```json
{
  "sam_output_dir": "strawberry_outputs_all/outputs_multi_raw_03",
  "output_dir": "strawberry_outputs_all/outputs_multi_03",
  "sort_order": "descending center_y, then descending center_x",
  "direction_standard": "line from ignored_tip_vertex to the midpoint of thumb_support_vertex and pair_support_vertex",
  "direction_visualization": "strawberry_outputs_all/outputs_multi_03/strawberry_direction_visualization.jpg",
  "failed_targets": 0,
  "targets": [
    {
      "target": "target_00",
      "source_mask_index": 0,
      "sort_center": [x, y],
      "mask_bbox_xywh": [x, y, w, h],
      "status": "ok",
      "grasp_points_2d": {
        "P_index": [x, y],
        "P_middle": [x, y],
        "P_thumb": [x, y]
      },
      "grasp_pair_center_2d": [x, y],
      "direction_line_2d": {
        "from_ignored_tip_vertex": [x, y],
        "to_support_midpoint": [x, y],
        "vector": [dx, dy]
      }
    }
  ]
}
```

字段说明：

- `source_mask_index`：该目标来自原始 Grounded-SAM 的第几个 `mask_i.png`。
- `sort_center`：用于排序的目标中心点。
- `mask_bbox_xywh`：该目标掩码外接框。
- `status`：`ok` 表示抓取点生成成功。
- `failed_targets`：抓取点生成失败的目标数量。
- `direction_standard`：方向确认线定义。
- `direction_visualization`：整图方向确认可视化路径。

如果某个目标无法形成稳定抓取三角形，该目标目录仍会保留：

```text
target_XX/
├── raw_image.jpg
├── mask_0.png
├── mask.json
└── grasp_error.json
```

默认情况下，单个目标失败不会中断后续目标处理。如果希望遇到失败立即停止，可以加：

```bash
--strict
```

## 三、推荐命名规范

单草莓：

```text
strawberry_outputs_all/outputs_strawberry_01
strawberry_outputs_all/outputs_strawberry_02
...
```

多草莓：

```text
strawberry_outputs_all/outputs_multi_raw_01
strawberry_outputs_all/outputs_multi_01

strawberry_outputs_all/outputs_multi_raw_02
strawberry_outputs_all/outputs_multi_02
```

其中：

- `outputs_multi_raw_XX`：Grounded-SAM 原始多目标分割结果。
- `outputs_multi_XX`：排序后的多目标抓取点结果。

## 四、常用检查命令

查看某个输出目录下的文件：

```bash
find strawberry_outputs_all/outputs_multi_03 -maxdepth 2 -type f | sort
```

查看多目标汇总结果：

```bash
cat strawberry_outputs_all/outputs_multi_03/all_grasp_points.json
```

查看 Grounded-SAM 检测结果：

```bash
cat strawberry_outputs_all/outputs_multi_raw_03/mask.json
```

统计每个原始 mask 的像素和 bbox：

```bash
python - <<'PY'
import cv2, glob, numpy as np

for path in sorted(glob.glob("strawberry_outputs_all/outputs_multi_raw_03/mask_*.png")):
    mask = cv2.imread(path, 0)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        print(path, "empty")
        continue
    bbox = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    )
    print(path, "pixels=", len(xs), "bbox_xywh=", bbox)
PY
```

## 五、完整命令模板

### 单草莓完整模板

```bash
cd /data/data/workspace/zhangyize/Grounded-Segment-Anything-main
source .venv/bin/activate

python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image testdata/strawberry.jpg \
  --output_dir strawberry_outputs_all/outputs_strawberry_13 \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "strawberry" \
  --device cuda

python tools/generate_strawberry_grasp.py \
  --mask strawberry_outputs_all/outputs_strawberry_13/mask_0.png \
  --image strawberry_outputs_all/outputs_strawberry_13/raw_image.jpg \
  --layout triangle-paired \
  --thumb-side top \
  --output-json strawberry_outputs_all/outputs_strawberry_13/grasp_points_triangle.json \
  --output-vis strawberry_outputs_all/outputs_strawberry_13/grasp_points_triangle_visualization.jpg
```

### 多草莓完整模板

```bash
cd /data/data/workspace/zhangyize/Grounded-Segment-Anything-main
source .venv/bin/activate

python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image testdata/multi_strawberry.jpg \
  --output_dir strawberry_outputs_all/outputs_multi_raw_03 \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "strawberry" \
  --device cuda

python tools/generate_multi_strawberry_grasps.py \
  --sam-output-dir strawberry_outputs_all/outputs_multi_raw_03 \
  --output-dir strawberry_outputs_all/outputs_multi_03 \
  --layout triangle-paired \
  --thumb-side top \
  --overwrite
```
