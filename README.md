# 工业事件日志时钟校准 API

基于 **Python + FastAPI** 的时钟漂移校准服务:给定一份可信的**参考事件日志**(JSONL)
和一份存在时钟漂移、缺失事件、插入(杂散)事件的**待校准日志**,服务会:

1. 以**事件标签**作为匹配依据,用**单调动态规划(带权 LCS,Fenwick 树优化)**求两侧
   事件的最大权一致匹配——天然跳过缺失/插入事件,且匹配在两条时间轴上都严格保序;
2. 以匹配上的事件对为**锚点**,拟合参考时间关于待校准时间的**分段线性时钟映射**
   (锚点间线性插值,锚点区间外用边缘斜率外推),支持偏移、速率漂移与时钟跳变;
3. 输出每条记录校准后的时间戳、匹配置信度(锚点留一法残差的高斯衰减)、未匹配的
   插入/缺失记录,以及各分段的斜率与漂移率(ppm)。

## 目录结构

```
app/
  main.py       FastAPI 路由(POST /align、GET /health)
  matching.py   单调 DP 匹配 + 分段线性时钟映射
  io.py         JSONL 解析与结果装配
  timeutil.py   时间戳解析(秒/毫秒/微秒/ISO-8601)与格式化
  models.py     内部数据结构
examples/       示例 JSONL(含漂移、跳变、缺失与插入)
scripts/
  make_examples.py  生成示例数据
tests/          pytest 测试(算法 + HTTP 层)
Dockerfile
docker-compose.yml
```

## 一键启动(Docker Compose)

```bash
docker compose up --build -d
```

服务监听 **8000** 端口:

| 地址                       | 说明                |
| -------------------------- | ------------------- |
| http://localhost:8000/health | 健康检查            |
| http://localhost:8000/docs   | Swagger 交互文档    |
| http://localhost:8000/redoc  | ReDoc 文档          |
| http://localhost:8000/openapi.json | OpenAPI 模式  |

停止:`docker compose down`

## 本地直接运行(可选)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 接口

### GET /health

```bash
curl http://localhost:8000/health
# {"status":"ok","service":"clock-align","version":"1.0.0"}
```

### POST /align

接受两种等价的提交方式。

**方式一:JSON 请求体**(`Content-Type: application/json`),
`reference` / `target` 可以是事件数组,也可以直接传 JSONL 文本字符串:

```bash
curl -s http://localhost:8000/align \
  -H 'Content-Type: application/json' \
  -d '{
        "reference": [
          {"timestamp": 1790000000.0, "label": "LINE_START"},
          {"timestamp": 1790000030.0, "label": "HEATER_ON"}
        ],
        "target": [
          {"timestamp": 1790000005.6, "label": "LINE_START"},
          {"timestamp": 1790000036.2, "label": "HEATER_ON"}
        ],
        "iterations": 3,
        "time_weight": 0.5
      }'
```

**方式二:上传 JSONL 文件**(`multipart/form-data`):

先生成示例数据(`python scripts/make_examples.py`,已附带在 `examples/`),
然后:

```bash
curl -s http://localhost:8000/align \
  -F "reference=@examples/reference.jsonl;type=application/x-ndjson" \
  -F "target=@examples/target.jsonl;type=application/x-ndjson" \
  -F "iterations=3" | python -m json.tool
```

### 人工锚点(anchors)

可选字段 `anchors` 是一组**人工锁定的匹配对**,每项形如
`{"reference_index": i, "target_index": j}`(索引均为 0 基,即事件在各自
输入数组/JSONL 中的行号)。人工锚点是**不可跳过**的匹配:算法先按锚点把
两侧序列切成相邻区间(首锚点之前、相邻锚点之间、末锚点之后),在每个区间内
**独立执行原有的保序带权匹配**,再合并全部锚点统一拟合分段线性时钟映射。
自动匹配因此:

* 不会占用任何已锁定事件(锁定事件不出现在候选窗口中);
* 不会跨越人工锚点(每个区间的候选对严格夹在两侧锚点之间)。

人工锚点在响应中标记 `"source": "manual"`(置信度固定为 `1.0`),自动匹配
标记 `"source": "auto"`;`summary.manual_anchor_count` 返回人工锚点数。

**JSON 请求示例**:

```bash
curl -s http://localhost:8000/align \
  -H 'Content-Type: application/json' \
  -d '{
        "reference": [
          {"timestamp": 1790000000.0, "label": "LINE_START"},
          {"timestamp": 1790000030.0, "label": "HEATER_ON"},
          {"timestamp": 1790000060.0, "label": "TEMP_RAMP"}
        ],
        "target": [
          {"timestamp": 1790000005.6, "label": "LINE_START"},
          {"timestamp": 1790000036.2, "label": "HEATER_ON"},
          {"timestamp": 1790000067.4, "label": "TEMP_RAMP"}
        ],
        "anchors": [
          {"reference_index": 0, "target_index": 0},
          {"reference_index": 2, "target_index": 2}
        ]
      }'
```

**multipart 请求示例**(锚点通过同名表单字段以 JSON 文本提交):

```bash
curl -s http://localhost:8000/align \
  -F "reference=@examples/reference.jsonl;type=application/x-ndjson" \
  -F "target=@examples/target.jsonl;type=application/x-ndjson" \
  -F 'anchors=[{"reference_index":0,"target_index":0},{"reference_index":30,"target_index":30}]' \
  | python -m json.tool
```

以下情况返回 **400**,错误信息以 `anchors[k]` 等形式给出可定位原因:

* 索引越界(`reference_index` / `target_index` 超出各自事件数,或为负);
* 重复占用(同一参考或待校准事件被多个锚点锁定);
* 标签不一致(锚定的两个事件 `label` 不同);
* 次序交叉(按参考索引排序后,待校准索引不是严格递增——保序匹配无法同时满足)。

#### 请求参数

| 字段          | 类型            | 默认 | 说明                                              |
| ------------- | --------------- | ---- | ------------------------------------------------- |
| `reference`   | 数组或 JSONL 串 | 必填 | 参考日志                                          |
| `target`      | 数组或 JSONL 串 | 必填 | 待校准日志                                        |
| `anchors`     | 锚点对象数组    | 无   | 人工锚点,每项 `{reference_index, target_index}`;multipart 下以 JSON 文本字段提交 |
| `time_field`  | string          | 自动 | 时间戳字段名;缺省依次探测 `timestamp/time/ts/datetime/date` |
| `label_field` | string          | 自动 | 标签字段名;缺省依次探测 `label/tag/event/type/name`        |
| `iterations`  | int 1–10        | 3    | 匹配/拟合精化轮数                                 |
| `time_weight` | float 0–2       | 0.5  | 时间一致性高斯项相对标签匹配分(1.0)的权重        |

时间戳支持:数字(秒 / 毫秒 / 微秒,按量级自动判定)与 ISO-8601 字符串
(如 `2026-09-22T08:00:00Z`)。单次请求每侧上限 50,000 条记录。

#### 响应结构

```jsonc
{
  "summary": {
    "reference_count": 60,
    "target_count": 61,
    "matched_count": 59,        // 锚点数(人工 + 自动)
    "manual_anchor_count": 2,   // 人工锁定锚点数
    "auto_anchor_count": 57,    // 自动匹配锚点数
    "inserted_count": 2,        // 待校准侧多余(插入)事件
    "missing_count": 1,         // 参考侧缺失事件
    "mean_confidence": 0.82,    // 锚点平均置信度 [0,1]
    "min_confidence": 0.0,      // 时钟跳变点处的锚点接近 0
    "residual_rms": 0.0,        // 锚点处映射残差 RMS(秒)
    "residual_scale": 0.058,    // 留一法残差尺度(秒)
    "mapping_type": "PiecewiseLinearMapping",
    "elapsed_seconds": 0.012
  },
  "clock_mapping": {
    "type": "PiecewiseLinearMapping",
    "global_slope": 1.00003,
    "segments": [               // 相邻锚点构成的分段
      {
        "index": 0,
        "start_target_time": 1790000005.014,
        "start_target_time_iso": "2026-09-22T...Z",
        "end_target_time": 1790000035.553,
        "start_reference_time": 1790000000.0,
        "end_reference_time": 1790000030.0,
        "offset_at_start": -5.014,   // 该段起点时钟偏差(秒)
        "slope": 1.00003,
        "drift_ppm": 30.0            // 速率漂移(parts-per-million)
      }
    ]
  },
  "anchors": [ { "reference_index": 0, "target_index": 0, "label": "...",
                 "source": "manual",                  // manual = 人工锁定,auto = 自动匹配
                 "reference_time": ..., "target_time": ...,
                 "mapped_time": ..., "residual": 0.0, "confidence": 1.0 } ],
  "aligned_events": [
    // 待校准侧每条记录;status = matched | inserted
    // matched 时 source = manual | auto,inserted 时为 null
    { "index": 0, "status": "matched", "label": "LINE_START", "source": "manual",
      "original_time": 1790000005.014,
      "original_time_iso": "2026-09-22T...Z",
      "calibrated_time": 1790000000.0,
      "calibrated_time_iso": "2026-09-22T...Z",
      "confidence": 0.98,
      "matched_reference_index": 0,
      "residual": 0.0,
      "extrapolated": false }
  ],
  "missing_reference_events": [
    // 参考侧有、待校准侧没有的事件;status = missing
    { "index": 20, "status": "missing", "label": "LINE_START",
      "reference_time": 1790000600.0,
      "reference_time_iso": "2026-09-22T...Z",
      "record": { ... } }
  ]
}
```

说明:

* `inserted` 记录(杂散事件)仍然给出 `calibrated_time`(按映射外推/插值),
  但 `confidence = 0`,`matched_reference_index = null`;
* `extrapolated = true` 表示该时间点位于锚点区间之外,使用边缘段斜率外推;
* 跨时钟跳变的分段 `drift_ppm` 会显著偏大,可据此定位跳变位置;跳变点两侧锚点
  的置信度会明显降低。

## 算法概要

* **候选对**:只有标签相同的 `(参考事件, 待校准事件)` 才能配对。每对的权重为
  `1.0 + time_weight · exp(-½(r/σ)²)`,其中 `r = 映射(t_target) - t_ref`;
  时间一致的对权重约 1.4–1.5,跳过事件得 0 分,因此插入/缺失被自然舍弃,无需
  手工空跳罚分。
* **单调 DP**:要求两侧索引都严格递增,即最大权公共子序列。按 target 位置分批、
  以 ref 位置为 Fenwick 树下标做前缀最大值 DP,复杂度 O(C log m)(C 为同标签
  候选对数),万级记录秒级完成。
* **分段拟合**:锚点(t_target → t_ref)按时间排序,逐段线性插值;区间外用首/
  末段斜率外推。第一轮先做**自适应自举**:同时构造两种候选配对——按原始时间取
  最近同标签事件(抗插入/缺失)与按顺序位置配对(抗大速率漂移),选用偏移自洽性
  (MAD 最小)的一组拟合全局最小二乘直线;随后多轮"匹配 → 分段拟合 → 用锚点留一
  法残差收紧 σ",默认 3 轮收敛。
* **人工锚点**:给定人工锚点时,序列被锚点切成若干半开区间(首锚点之前、相邻
  锚点之间、末锚点之后),原保序匹配在每个区间内独立运行——候选对的两侧索引都
  被严格限制在区间内,故自动匹配既不能复用已锁事件,也不能跨过锚点;两个及以上
  锚点时直接以穿过全部锚点的全局最小二乘直线作为各区间的初始映射。所有人工锚点
  与区间自动锚点合并后统一拟合分段线性映射;人工锚点置信度固定为 1.0,自动锚点
  的置信度只由自动锚点的留一法残差尺度归一化,避免人工锁定被时钟跳变相邻段压低。
* **置信度**:对每个锚点计算其由相邻锚点局部直线预测的留一法残差,以全体残差的
  P90 为尺度做高斯衰减;时钟断点附近残差大、置信度低,其余锚点接近 1。

## 测试

```bash
pip install pytest httpx python-multipart
python -m pytest -q
```

覆盖:已知分段漂移恢复、纯速率漂移、重复同标签保序匹配、标签全无交集、单锚点、
外推标记、JSONL 文本输入、字段缺失报错以及 `/health`、`/align` 的
JSON / multipart / 400 / 415 路径;人工锚点覆盖区间切分不跨锚、锁定事件不被
自动匹配占用、单锚点/全锚点、JSON 与 multipart 两种提交,以及索引越界、重复
占用、标签不一致、次序交叉、非法 JSON 的 400 定位报错。
