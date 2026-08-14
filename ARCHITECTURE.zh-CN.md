# req-frontend：从输入文件到测量报告

本文自底向上解释 `req-frontend` 的术语、目录边界和实际 wiring。它描述当前代码，
不是未来设计草案。

## 一条 request 经过哪些层

```text
CSV cells
  ↓
InputFileSchema
  = InputFileFormat + TraceTag[]
  ↓
schema/format/*：读取、反序列化、验证
  ↓
IndependentRequest 或 SessionPlans<SessionRound>
  ↓
workload.rs：选择 loader、截断、缩放 arrival、统计 workload
  ↓
executor/：按 independent 或 session 语义 release
  ↓
tokens.rs：构造实际 token ids
  ↓
backend/：发送 HTTP、折叠 stream events
  ↓
record.rs + timeline.rs：记录 per-request / per-event 事实
  ↓
slo.rs + summary.rs：比较声明值与测量值并聚合
```

最重要的分界是：

- `schema/` 解释输入文件已经声明的事实；
- `workload.rs` 应用本次运行的 replay 操作；
- `executor/` 和 `backend/` 执行并测量；
- `record.rs`、`slo.rs`、`summary.rs` 输出结果。

## 1. CSV header 不是完整 schema

考虑一份 native 文件：

```csv
id,arrival_time,input_len,output_len,ttft_slo_ms,tpot_slo_ms,e2e_slo_ms,priority
req-0,0.0,512,64,300,25,1200,3
```

只看 cells 和 header，consumer 仍然不知道：

- 这份文件描述 text generation 还是其他 request family；
- SLO 和 priority columns 是否是有意声明的；
- row 是否属于某种带 predecessor 语义的 session format；
- `input_len` 是完整 prompt，还是 canonical session row 的 fresh suffix。

因此 startup 必须显式构造一个 `InputFileSchema`。代码不会根据 header 猜测
request family，也不允许同一文件的不同 row 各自携带一个 `request_type` union。

## 2. 完整 format 决定 request family

```text
InputFileFormat + TraceTag[] = InputFileSchema
```

| 类型 | 回答的问题 | 当前例子 |
|---|---|---|
| `InputFileFormat` | family 是什么、row 怎样编码、由谁 load？ | `TextGenerationIndependent` |
| `RequestFamily` | 该完整 format 固有的 request 语义是什么？ | `TextGeneration`, `ImageToText` |
| `TraceTag` | 文件额外声明了哪些 column bundle？ | `Slo`, `Priority`, `Session` |
| `InputFileSchema` | 最终精确允许哪些 columns？ | format 加合法 tags 后的结果 |

### `InputFileFormat` 是完整选择

format 名同时编码 family 与布局，例如：

```text
text-generation-independent
text-generation-session-execution-v2
image-to-text-independent
text-to-image-independent
```

因此不存在一个 generic `Native` 再与另一个 family selector 拼接的中间状态，也不存在
`SessionExecutionV2 + ImageToText` 这样的非法组合。

`TextGenerationIndependent` 的基础 columns 是：

```text
id, arrival_time, input_len, output_len
```

`TextGenerationSessionExecutionV2` 的 canonical columns 是：

```text
request_id, session_id, round_idx, arrival_time_ms,
prefix_len, input_len, output_len, tool_wait_after_ms
```

canonical format 中的 `prefix_len`、round order 和 tool wait 都已在 generation time
materialize。replay 时不能重新猜 context policy。

### `RequestFamily` 是 format 的属性

`InputFileFormat::request_family()` 返回 format 固有的 family。这个 family 不是从 CSV
header 或 row 推断，也不是第二个 CLI 参数。不同 row 可以有不同长度、arrival、SLO、
priority 和 session round，但不能在同一 CSV 中切换 family。

### `TraceTag` 是额外的 column contract

```text
Slo
└── ttft_slo_ms, tpot_slo_ms, e2e_slo_ms

Priority
└── priority

Session
└── session_id, prefix_kv, tool_wait_after_ms   # native format 的 tag columns

Speculative
└── accept_rate
```

`Slo` 与 `Priority` 有意分开：SLO 是可测量 latency metric 的上限；priority 是 scheduler
可能使用的排序 hint。声明其中一个不会自动声明另一个。

## 3. `schema/` 的内部层级

```text
schema/
├── input_file_schema.rs
│   └── InputFileFormat + TraceTag[] → exact header contract
│
├── format/
│   ├── text_generation/
│   │   ├── independent.rs
│   │   └── session.rs
│   ├── image_to_text.rs
│   ├── video_to_text.rs
│   ├── audio_to_text.rs
│   ├── text_to_image.rs
│   ├── text_to_video.rs
│   ├── text_to_speech.rs
│   ├── image_to_video.rs
│   ├── omni_generation.rs
│   └── load_utils.rs
│       └── 每个 family format 都拥有 COLUMNS、typed Row、validation、load；
│           load_utils 只共享重复的 CSV/tag mechanics
│
├── family/
│   ├── media.rs
│   └── omni.rs
│       └── family-specific declared values；不负责 replay
│
└── tag/
    ├── slo.rs
    ├── priority.rs
    └── speculative.rs
        └── 每种 tag 的 per-row declared value
```

这些文件不在同一抽象层。`input_file_schema.rs` 只组合完整 format 与 tags；`format/`
拥有 physical row decoding；`family/` 和 `tag/` 提供可组合的字段类型。

### 为什么 session grouping 在 format loader 中

对 `SessionExecutionV2` 而言，以下条件属于文件格式本身：

- 同一 session 的 rows 必须连续；
- `round_idx` 必须从 0 连续递增；
- round 0 不能声明 prefix；
- 同一 session 的 arrival 必须一致；
- session blocks 必须按 arrival 排列；
- request id 必须唯一。

因此 loader 在返回前验证完整文件，并按文件顺序构造：

```rust
type SessionPlans = Vec<(String, Vec<SessionRound>)>;
```

这个 grouping 不是可选 runtime policy；它是 canonical bytes 的结构含义，所以属于
`schema/format/text_generation/session.rs`。把它放进另一个
`trace/session.rs` 只会制造第二个
解释同一 format 的地方。

`SessionRound` 也不是原始 CSV row。`ExecutionRow` 是 canonical 基础 columns；loader
另外读取已声明的 `RequestSlo` 和 `RequestPriority`，验证后合成为可交给 runtime 的
`SessionRound`：

```text
ExecutionRow + RequestSlo + RequestPriority
  ↓ format loader validates and groups
SessionRound
```

## 4. 为什么仍然需要 `workload.rs`

`workload.rs` 不再解析 CSV，也不再定义 independent/session 的第二套 loader。它只负责
本次运行才出现的操作：

```rust
load_workload(path, input_file_schema, max_items)
    -> ReplayWorkload
```

其内部流程是：

```text
1. 检查 HTTP replay runtime 是否支持该合法 schema
2. 按 InputFileFormat 调用该 family 自己的 format loader
3. 得到完全验证的 IndependentRequest 或 SessionPlans
4. 再应用 --max-items
5. 如指定 --rate，缩放 top-level arrival times
6. 计算 workload summary 与 offered-rate units
```

这一层不能合并回 `schema/`，因为 `--max-items`、`--rate` 和“这个 HTTP client 暂时只会
发送 text generation”都不是输入文件的含义。模拟器可以共享同一个 schema loader，
但使用不同的 replay policy。

特别地，loader 会先验证完整文件，再截断。否则 `--max-items 1` 会把第二行之后的损坏
静默藏起来。

`ReplayWorkload` 保留两种 runtime shape：

```rust
enum ReplayWorkload {
    Sessions(SessionPlans),
    IndependentRequests(Vec<IndependentRequest>),
}
```

这不是 per-row request-family union。它是 startup 根据整份文件选择一次的执行路径。

## 5. `runner.rs` 是 wiring root

`run_once_reusing` 按以下顺序连接各层：

```text
Args
  ↓ InputFileFormat::parse(...)
InputFileSchema::new(format, tags)
  ↓
slo_source::resolve(...)
  ↓
workload::load_workload(...)
  ↓
optional arrival-rate scaling
  ↓
WorkloadSummary + context-limit preflight
  ↓
token corpus / backend preflight
  ↓
executor::execute(...)
  ↓
RunSummary
```

CLI 只接受完整 format，不再有第二个 family selector：

```text
--input-file-format text-generation-independent
--input-file-format text-generation-session-execution-v2
```

## 6. `executor/` 负责 release 语义

```text
executor/independent.rs
└── 每个 request 按自己的 trace arrival release

executor/session.rs
└── round 0 按 session arrival release
    later round 等 predecessor 完成，再等待 tool_wait_after_ms

executor/admission.rs
└── max-concurrency 与 deterministic admission order
```

format loader 只证明 session topology 合法；executor 才执行时间相关的 dependency。这样
“文件说 predecessor 是谁”和“runtime 何时实际 release”不会混在一个层里。

## 7. `tokens.rs` 构造实际 prompt

independent request 从 synthetic token pool 取得 `input_len` 个 tokens。

session round 则构造：

```text
previous realized context[..prefix_len] + fresh tokens[input_len]
```

`prefix_len` 表示 cache-eligible prefix，不保证 server 实际 cache hit。真实 server 报告的
cached prompt tokens 与 declared/planned prefix 会分别记录，不能互相替代。

## 8. `backend/` 发送并测量 stream

`backend/wire/` 只处理协议差异：OpenAI、vLLM native token endpoint、SGLang native token
endpoint。`backend/client.rs` 和 `backend/stream.rs` 将不同 wire events 折叠成同一组可审计
measurement。

关键 timing 定义是：

- TTFT：HTTP send 到第一个 non-empty generated event；优先使用 token-id clock；
- TPOT：第一个 timed token event 之后的 token delivery 平均间隔；
- E2E：submission 到 completion；
- arrival release lag：计划 arrival 到 client task 实际恢复之间的延迟。

这四个量不能互相代替。尤其不能把 client scheduling lag 偷偷算进 server TTFT。

## 9. declared SLO 与 measured timing 分开保存

`schema/tag/slo.rs` 只定义输入文件声明的 per-request bounds：

```rust
RequestSlo {
    ttft_slo_ms: Option<f64>,
    tpot_slo_ms: Option<f64>,
    e2e_slo_ms: Option<f64>,
}
```

`schema/tag/priority.rs` 独立定义：

```rust
RequestPriority {
    priority: Option<i64>,
}
```

根目录的 `slo.rs` 位于更高层，负责 runtime measurement、attainment 和 aggregation。
它不属于 schema tag 层，因为这些计算只有执行完成后才存在。

每个实际声明的 metric 单独判断：

```text
attained(metric) = measured(metric) <= declared_bound(metric)
```

一个 request 声明多个 bounds 时，全部满足才算 combined attained。空 cell 表示该 request
没有为该 metric 提出 bound，不表示 0，也不表示自动达标。

## 10. 输出层

`record.rs` 写 per-request JSONL，将 source declaration 与 generation outcome 放在同一条
record 中。SLO fields 与 priority field 分开 flatten：

```text
declared_ttft_slo_ms
declared_tpot_slo_ms
declared_e2e_slo_ms
declared_priority
```

`timeline.rs` 异步写 per-event Parquet；channel 满时丢 timeline sample，而不阻塞被测的
submission path。`summary.rs` 生成 run-level metrics 和 SLO aggregation。

## 目录 owner map

| 路径 | 唯一职责 |
|---|---|
| `schema/input_file_schema.rs` | 在完整 format 上组合 tags 并验证 header |
| `schema/format/` | 读取、验证、组织 format 声明的结构 |
| `schema/family/` | family-specific declared value types |
| `schema/tag/` | orthogonal per-row declaration types |
| `workload.rs` | runtime selection、truncate、rate scaling、workload summary |
| `runner.rs` | 一次 run 的 wiring root |
| `executor/` | release、dependency、admission |
| `tokens.rs` | concrete token construction |
| `backend/` | protocol、stream fold、integrity、preflight |
| `record.rs` | per-request JSONL contract |
| `timeline.rs` | per-event timeline |
| `slo.rs` | measured SLO verdict 与 aggregation |
| `summary.rs` | run-level report |
| `bin/tracegen/` | raw source → canonical trace；generation-time policy |
| `bin/sweep/` | 多个 run point 的搜索与边界判断 |
| `bin/selfcheck/` | 用可控 stub 验证 release、TTFT、TPOT 等测量 fidelity |

最短的心智模型是：

```text
InputFileSchema 定义文件是什么。
format loader 证明文件确实如此。
workload 决定本次怎样 replay 已验证内容。
executor/backend 执行并测量。
record/summary 把声明与事实放在一起。
```
