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

## 六、远程服务器 HTTP 推理流程

本节记录本机和服务器分离后的自动流程。目标是让本机只负责相机、机械臂、G20、PCAN、手眼标定和真实抓取，服务器只负责接收图像、运行 Grounded-SAM、生成多目标三指抓取点并返回结果。

完整数据流：

```text
本机 SY1080P 拍照
  -> HTTP 上传 image + metadata 到服务器
  -> 服务器保存原始输入
  -> 服务器运行多草莓 Grounded-SAM 分割
  -> 服务器运行 generate_multi_strawberry_grasps.py
  -> 服务器返回 targets JSON 或 result.zip
  -> 本机保存到 outputs_multi_auto_YYYYMMDD_HHMMSS
  -> 本机校验 raw_image 与本机拍照图一致
  -> 本机调用 run_table_strawberry_multi_quick.py 执行抓取
```

### 1. 本机与服务器职责边界

本机负责：

- SY1080P 相机拍照。
- xArm 状态检查与运动执行。
- G20 左手控制。
- PCAN 通信。
- 手眼标定计算。
- 抓取高度、放置位姿、避障和真实抓取流程。
- 上传本机拍摄图和拍照 metadata。
- 下载或接收服务器抓取点结果。
- 在执行抓取前做安全校验。

服务器负责：

- 接收本机上传的 jpg 图像和 metadata json。
- 保留上传原始图像字节。
- 检查输入图像是否为 `640x480`。
- 运行 `grounded_sam_demo.py` 生成多草莓实例掩码。
- 运行 `tools/generate_multi_strawberry_grasps.py` 生成每颗草莓的三指抓取点。
- 按既有规则排序目标：`center_y` 越大越优先，若接近则 `center_x` 越大越优先。
- 返回同步 JSON 或可下载 zip。
- 保存每次任务的输入、输出和 manifest。

服务器不负责：

- 控制 xArm。
- 控制 G20。
- 控制 SY1080P。
- 执行手眼标定。
- 真实抓取和放置。

### 2. 远程互联传输技术说明

当前实现使用 HTTP/1.1 REST 接口，服务端框架为 FastAPI，运行时由 Uvicorn 提供 ASGI HTTP 服务。

使用的传输方式：

- 协议：HTTP。
- 服务端：FastAPI + Uvicorn。
- 上传格式：`multipart/form-data`。
- 上传文件字段：
  - `image`：本机拍摄的 jpg 原图。
  - `metadata`：本机拍照时保存的 json 元数据。
- 普通返回格式：JSON。
- 大结果返回格式：zip 文件，通过 HTTP 下载。
- 健康检查：`GET /health`。
- 同步推理接口：`POST /api/strawberry/multi_grasp`。
- 任务状态接口：`GET /api/strawberry/jobs/{job_id}`。
- 结果 zip 接口：`GET /api/strawberry/jobs/{job_id}/result.zip`。

为什么采用 HTTP + multipart：

- 本机 Windows 端可以直接用 `requests` 上传文件，不需要额外 RPC 框架。
- `multipart/form-data` 适合同时传 jpg 二进制和 json metadata。
- JSON 响应便于本机直接解析目标点。
- zip 响应便于保留完整目录结构，兼容现有 `run_table_strawberry_multi_quick.py` 所需的目录格式。
- 服务器与本机之间只交换感知结果，不让服务器接触机械臂控制，降低真实执行风险。

当前推荐先使用同步接口。同步接口会在一次 POST 中完成：

```text
上传 -> 推理 -> 抓取点生成 -> 返回 JSON/zip
```

后续如果推理时间变长，可以扩展为异步接口：

```text
POST 上传 -> 返回 job_id
GET /api/strawberry/jobs/{job_id} 轮询状态
GET /api/strawberry/jobs/{job_id}/result.zip 下载结果
```

当前服务已经保留了 job 查询和 zip 下载接口，因此后续升级异步时目录和返回结构不需要大改。

### 3. 服务器新增文件

服务器项目中新增：

```text
server_strawberry_grasp.py
tools/start_strawberry_grasp_server.py
```

`server_strawberry_grasp.py` 提供 FastAPI 应用和接口逻辑。

`tools/start_strawberry_grasp_server.py` 用于启动服务，并从首选端口开始自动选择空闲端口。默认从 `8765` 开始，如果被占用会尝试 `8766`、`8767` 等。

运行时任务目录默认保存到：

```text
strawberry_server_jobs/
```

该目录应保持本地运行数据，不提交到 Git。

### 4. 服务器启动

进入项目：

```bash
cd /data/data/workspace/zhangyize/Grounded-Segment-Anything-main
source .venv/bin/activate
```

推荐启动方式：

```bash
python tools/start_strawberry_grasp_server.py \
  --preferred-port 8765 \
  --device cuda
```

输出中会显示实际监听地址，例如：

```text
Starting strawberry grasp server on http://0.0.0.0:8765
Jobs dir: /data/data/workspace/zhangyize/Grounded-Segment-Anything-main/strawberry_server_jobs
```

如果 `8765` 被占用，脚本会自动选择后续空闲端口。此时本机 `--server-url` 必须使用实际端口。

也可以直接用 Uvicorn 启动默认配置：

```bash
uvicorn server_strawberry_grasp:app --host 0.0.0.0 --port 8765
```

但直接 Uvicorn 不会自动切换端口，因此更推荐使用启动脚本。

### 5. 服务器健康检查

本机或服务器上可检查：

```bash
curl http://SERVER_IP:8765/health
```

正常返回示例：

```json
{
  "status": "ok",
  "project_root": "/data/data/workspace/zhangyize/Grounded-Segment-Anything-main",
  "jobs_dir": "/data/data/workspace/zhangyize/Grounded-Segment-Anything-main/strawberry_server_jobs",
  "device": "cuda"
}
```

### 6. 同步推理接口

接口：

```text
POST http://SERVER_IP:8765/api/strawberry/multi_grasp
```

表单字段：

```text
image          jpg 文件，本机 SY1080P 拍照原图
metadata       json 文件，本机拍照元数据
return_format  json 或 zip
```

推荐先用：

```text
return_format=json
```

这样本机可以直接读取返回的 `targets`。如果要完整目录，则使用：

```text
return_format=zip
```

### 7. curl 测试命令

服务器本机测试：

```bash
curl -X POST http://127.0.0.1:8765/api/strawberry/multi_grasp \
  -F "image=@testdata/test_16.jpg" \
  -F "metadata=@testdata/test_16.json;type=application/json" \
  -F "return_format=json"
```

如果没有现成 metadata 文件，可临时创建：

```bash
cat > /tmp/test_16_metadata.json <<'JSON'
{
  "capture_name": "test_16",
  "tcp_pose": [0, 0, 0, 0, 0, 0]
}
JSON

curl -X POST http://127.0.0.1:8765/api/strawberry/multi_grasp \
  -F "image=@testdata/test_16.jpg" \
  -F "metadata=@/tmp/test_16_metadata.json;type=application/json" \
  -F "return_format=json"
```

从本机 Windows 访问时，把 `127.0.0.1` 换成服务器 IP：

```text
http://SERVER_IP:8765/api/strawberry/multi_grasp
```

### 8. JSON 返回格式

同步 JSON 返回示例：

```json
{
  "job_id": "20260618_161949_203990",
  "status": "done",
  "raw_image_sha256": "c3...",
  "width": 640,
  "height": 480,
  "target_count": 4,
  "failed_targets": 0,
  "targets": [
    {
      "id": "target_00",
      "order": 0,
      "status": "ok",
      "source_mask_index": 3,
      "sort_center": [290.75, 379.17],
      "mask_bbox_xywh": [269, 361, 45, 40],
      "grasp_points_2d": {
        "P_index": [296, 390],
        "P_middle": [304, 385],
        "P_thumb": [277, 363]
      },
      "grasp_pair_center_2d": [300, 388],
      "direction_line_2d": {
        "from_ignored_tip_vertex": [309.0, 363.0],
        "to_support_midpoint": [289.0, 376.0],
        "vector": [-20.0, 13.0]
      },
      "score": 0.5
    }
  ],
  "server_output_dir": "/data/data/workspace/zhangyize/Grounded-Segment-Anything-main/strawberry_server_jobs/20260618_161949_203990/output/outputs_multi_auto_20260618_161949_203990",
  "all_grasp_points_json": "/data/data/workspace/zhangyize/Grounded-Segment-Anything-main/strawberry_server_jobs/20260618_161949_203990/output/outputs_multi_auto_20260618_161949_203990/all_grasp_points.json",
  "result_zip_url": "/api/strawberry/jobs/20260618_161949_203990/result.zip",
  "timing": {
    "grounded_sam_s": 3.2,
    "grasp_generation_s": 0.4,
    "total_s": 3.8
  }
}
```

本机抓取前至少检查：

- `status == "done"`。
- `width == 640` 且 `height == 480`。
- `target_count > 0`。
- `failed_targets` 不应等于 `target_count`。
- 每个可抓目标 `status == "ok"`。
- 每个可抓目标都有 `grasp_points_2d.P_index`、`grasp_points_2d.P_middle`、`grasp_points_2d.P_thumb`。
- 每个可抓目标都有 `grasp_pair_center_2d`。
- `raw_image_sha256` 与本机上传前计算的 sha256 一致；若不一致，再做 RMSE 校验，超阈值则停止抓取。

### 9. zip 返回格式

如果请求：

```text
return_format=zip
```

服务器直接返回 zip 文件。zip 解压后根目录为：

```text
outputs_multi_auto_{job_id}/
```

内部结构：

```text
outputs_multi_auto_{job_id}/
├── raw_image.jpg
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

这个结构与多草莓本地输出兼容，本机可以把它保存为：

```text
C:\Users\张一泽\Desktop\workspace\testdata\outputs_multi_auto_YYYYMMDD_HHMMSS
```

然后继续复用现有抓取入口：

```powershell
& "C:\conda_envs\grasp-perception\python.exe" `
  "C:\Users\张一泽\Desktop\workspace\robot_try\robot_try\scripts\run_table_strawberry_multi_quick.py" `
  --capture test_xxx `
  --server-dir "C:\Users\张一泽\Desktop\workspace\testdata\outputs_multi_auto_YYYYMMDD_HHMMSS" `
  --run-id auto_YYYYMMDD_HHMMSS `
  --apply
```

### 10. 服务器任务目录

每个请求会创建一个独立 job 目录：

```text
strawberry_server_jobs/
└── 20260618_161949_203990/
    ├── input/
    │   ├── raw_image.jpg
    │   └── metadata.json
    ├── output_raw/
    │   ├── raw_image.jpg
    │   ├── mask_0.png
    │   ├── mask_1.png
    │   ├── mask.json
    │   ├── mask.jpg
    │   └── grounded_sam_output.jpg
    ├── output/
    │   └── outputs_multi_auto_20260618_161949_203990/
    │       ├── raw_image.jpg
    │       ├── all_grasp_points.json
    │       ├── strawberry_direction_visualization.jpg
    │       ├── target_00/
    │       └── target_01/
    ├── manifest.json
    └── result.zip
```

说明：

- `input/raw_image.jpg` 是本机上传的原始图片字节。
- `input/metadata.json` 是本机上传的拍照元数据。
- `output_raw/` 是 Grounded-SAM 原始多目标分割结果。
- `output/outputs_multi_auto_{job_id}/` 是排序后的多目标抓取点目录。
- `manifest.json` 记录 job 状态、sha256、输出路径、耗时、目标数量等。
- `result.zip` 是本机可下载的完整结果包。

服务器会把上传原图复制回输出目录中的 `raw_image.jpg`，避免输出图变成模型预处理后的缩放图或裁剪图。

### 11. 错误返回

服务器错误统一返回 JSON：

```json
{
  "job_id": "20260618_001",
  "status": "failed",
  "error": "no strawberry detected"
}
```

常见错误：

- `image must be 640x480`：上传图分辨率不是 `640x480`。
- `uploaded image is not a readable image`：上传的不是可读取图像。
- `metadata is not valid UTF-8 JSON`：metadata 不是合法 JSON。
- `no strawberry detected`：Grounded-SAM 没有输出 `mask_*.png`。
- `grasp point generation failed for all targets`：所有目标都未能生成抓取点。
- `model inference timed out`：模型推理超时。
- `command failed`：底层脚本执行失败，错误信息中会包含命令和最后若干行日志。

本机收到 `status == "failed"` 时必须停止抓取，不应继续调用机械臂执行。

### 12. 本机一键命令目标

本机最终目标入口：

```powershell
& "C:\conda_envs\grasp-perception\python.exe" `
  "C:\Users\张一泽\Desktop\workspace\robot_try\robot_try\scripts\run_table_strawberry_auto_pipeline.py" `
  --name auto_001 `
  --camera-index 0 `
  --server-url "http://SERVER_IP:8765" `
  --apply
```

本机脚本建议执行顺序：

1. 调 `capture_sy1080p_synced.py` 拍照。
2. 得到 `test_xxx.jpg` 和 `test_xxx.json`。
3. 计算本机 jpg 的 sha256。
4. 用 `requests.post(..., files=..., data={"return_format": "json"})` 上传。
5. 等待服务器同步返回。
6. 校验返回 `status`、分辨率、sha256、目标数量和抓取点字段。
7. 如果需要完整目录，则根据 `result_zip_url` 下载 zip 并解压。
8. 保存为 `outputs_multi_auto_YYYYMMDD_HHMMSS`。
9. 生成本机 manifest。
10. 调 `run_table_strawberry_multi_quick.py --server-dir ... --apply`。
11. 保存总报告到 `robot_try/robot_try/config/last_table_strawberry_auto_pipeline_xxx.json`。

### 13. 本机上传示例代码

```python
from pathlib import Path
import requests

server_url = "http://SERVER_IP:8765"
image_path = Path(r"C:\Users\张一泽\Desktop\workspace\testdata\test_xxx.jpg")
metadata_path = Path(r"C:\Users\张一泽\Desktop\workspace\testdata\test_xxx.json")

with image_path.open("rb") as image_file, metadata_path.open("rb") as metadata_file:
    response = requests.post(
        f"{server_url}/api/strawberry/multi_grasp",
        files={
            "image": (image_path.name, image_file, "image/jpeg"),
            "metadata": (metadata_path.name, metadata_file, "application/json"),
        },
        data={"return_format": "json"},
        timeout=300,
    )

response.raise_for_status()
data = response.json()
if data.get("status") != "done":
    raise RuntimeError(data)

targets = data["targets"]
```

### 14. 本机安全校验要求

执行真实抓取前必须校验：

- 本机拍照文件存在。
- 本机 metadata 文件存在。
- metadata 中 TCP 位姿存在。
- 服务器返回 `status == "done"`。
- 服务器返回 `width == 640`、`height == 480`。
- 服务器返回 `raw_image_sha256` 与本机上传前 sha256 一致。
- 如果 sha256 不一致，必须计算本机 jpg 与服务器 `raw_image.jpg` 的 RMSE；RMSE 超阈值时停止。
- `target_count > 0`。
- 每个 `status == "ok"` 的目标都有完整 `P_index`、`P_middle`、`P_thumb`、`grasp_pair_center_2d`。
- xArm 当前无 error/warn。
- G20 和 PCAN 初始化成功。
- 现有标定和抓取配置文件不被自动修改。

不要在服务器端执行任何机械臂控制。服务器返回的点只能作为感知结果，本机必须在执行前二次校验并负责所有安全动作。
