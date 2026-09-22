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

#### 请求参数

| 字段          | 类型            | 默认 | 说明                                              |
| ------------- | --------------- | ---- | ------------------------------------------------- |
| `reference`   | 数组或 JSONL 串 | 必填 | 参考日志                                          |
| `target`      | 数组或 JSONL 串 | 必填 | 待校准日志                                        |
| `time_field`  | string          | 自动 | 时间戳字段名;缺省依次探测 `timestamp/time/ts/datetime/date` |
| `label_field` | string          | 自动 | 标签字段名;缺省依次探测 `label/tag/event/type/name`        |
| `iterations`  | int 1–10        | 3    | 匹配/拟合精化轮数                                 |
| `time_weight` | float 0–2       | 0.5  | 时间一致性高斯项相对标签匹配分(1.0)的权重        |
| `anchors`     | 数组            | 无   | 人工锚点,每项 `{"reference_index": i, "target_index": j}`,见下 |

时间戳支持:数字(秒 / 毫秒 / 微秒,按量级自动判定)与 ISO-8601 字符串
(如 `2026-09-22T08:00:00Z`)。单次请求每侧上限 50,000 条记录。

### 人工锚点(`anchors`)

当自动匹配在重复标签、长时间偏移或强时钟跳变处可能选错时,可用 `anchors`
**强制指定若干必须成立的匹配对**(索引即输入数组/JSONL 的行号,从 0 开始):

* 人工锚点是**不可跳过**的匹配:既不会被自动匹配覆盖,也保证出现在响应
  `anchors` 中(`source = "manual"`,自动匹配为 `source = "auto"`);
* 人工锚点把参考/待校准序列切分为**相邻锚点区间**,各区间独立执行原有保序
  单调 DP,再合并拟合——因此自动匹配**不能占用已锁定事件,也不能跨越锚点**;
* 首轮时钟映射直接由人工锚点拟合(≥2 个锚点为分段线性,1 个为常数偏移),
  无需依赖全局自举;
* 以下情况返回 **400**,错误信息可定位到具体锚点下标与字段:
  索引越界、同一事件被重复占用、两侧标签不一致、两侧次序交叉
  (按 `reference_index` 排序后 `target_index` 必须严格递增)。

**JSON 请求示例:**

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
          {"timestamp": 1790000066.8, "label": "TEMP_RAMP"}
        ],
        "anchors": [
          {"reference_index": 0, "target_index": 0},
          {"reference_index": 2, "target_index": 2}
        ]
      }' | python -m json.tool
```

**multipart/form-data 请求示例:** `anchors` 可以是**一个 JSON 数组**表单字段,
也可以为每个锚点重复提交一个 JSON 对象表单字段:

```bash
# 方式 A:一个 JSON 数组
curl -s http://localhost:8000/align \
  -F "reference=@examples/reference.jsonl;type=application/x-ndjson" \
  -F "target=@examples/target.jsonl;type=application/x-ndjson" \
  -F 'anchors=[{"reference_index":0,"target_index":0},{"reference_index":59,"target_index":59}]' \
  | python -m json.tool

# 方式 B:每个锚点一个重复的 anchors 表单字段
curl -s http://localhost:8000/align \
  -F "reference=@examples/reference.jsonl;type=application/x-ndjson" \
  -F "target=@examples/target.jsonl;type=application/x-ndjson" \
  -F 'anchors={"reference_index":0,"target_index":0}' \
  -F 'anchors={"reference_index":59,"target_index":59}' \
  | python -m json.tool
```

#### 响应结构

```jsonc
{
  "summary": {
    "reference_count": 60,
    "target_count": 61,
    "matched_count": 59,        // 锚点数(人工 + 自动)
    "manual_anchor_count": 2,   // 人工锚点数(anchors 中 source=manual)
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
  "anchors": [ { "reference_index": 0, "target_index": 0,
                 "source": "manual",        // manual = 人工锚点;auto = DP 自动匹配
                 "label": "...",
                 "reference_time": ..., "target_time": ...,
                 "mapped_time": ..., "residual": 0.0, "confidence": 0.98 } ],
  "aligned_events": [
    // 待校准侧每条记录;status = matched | inserted
    { "index": 0, "status": "matched", "label": "LINE_START",
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

* **人工锚点**:存在人工锚点时,首轮映射直接由锚点拟合;随后每轮把两条序列按
  锚点位置切成相邻区间,各区间分别执行下面的单调 DP(子序列上以段内秩重编号),
  再按序合并。锁定事件不进入任何候选对,自动匹配因而既无法占用它们,也无法跨
  越锚点边界;人工锚点始终保留在最终锚点集中并参与分段拟合与留一法置信度评分。
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
* **置信度**:对每个锚点计算其由相邻锚点局部直线预测的留一法残差,以全体残差的
  P90 为尺度做高斯衰减;时钟断点附近残差大、置信度低,其余锚点接近 1。

## 测试

```bash
pip install pytest httpx python-multipart
python -m pytest -q
```

覆盖:已知分段漂移恢复、纯速率漂移、重复同标签保序匹配、标签全无交集、单锚点、
外推标记、JSONL 文本输入、字段缺失报错以及 `/health`、`/align` 的
JSON / multipart / 400 / 415 路径;人工锚点覆盖强制匹配与区间隔离、不跨锚点/
不占锁定事件、越界/重复占用/标签不一致/次序交叉四类 400,以及 JSON 与
multipart(数组字段与重复对象字段)两种提交方式。
