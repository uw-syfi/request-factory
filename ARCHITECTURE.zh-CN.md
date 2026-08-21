# req-frontend：从输入文件到 replay 报告

本文解释 `req-frontend` 自己的端到端数据流：一次 run 如何声明输入文件、解析并验证
rows、构造 replay workload、按时间 release 请求、与 serving backend 交互，最后写出日志与
汇总。SLO、priority、session 和 speculative 都是这条主线上的局部语义，不是架构主轴。

## 1. 一次 run 的完整路径

`runner::run_once_reusing` 是 wiring root。正常运行依次经过：

```text
YAML config
  │ launcher：validate → resolve paths → build argv
  ▼
Rust Args
  │
  ├─ parse --input-file-format + --trace-tag
  ▼
InputFileSchema
  │  声明整份文件的 format、request family、合法 columns
  ▼
format::<family>::load(path, schema)
  │  header validation → CSV decoding → row validation → structural validation
  ▼
typed file contents
  ├─ Vec<IndependentRequest>
  └─ SessionPlans = Vec<(session_id, Vec<SessionRound>)>
  ▼
ReplayWorkload
  │  --max-items、--rate、arrival mode、workload summary
  ▼
executor
  │  arrival wait → admission → prompt construction → context-limit decision
  ▼
GenerationClient
  │  backend-specific JSON → HTTP stream → normalized events
  ▼
GenerationResult
  │
  ├─ StepLog channel ──────→ JSONL + replay/SLO aggregation
  └─ TimelineSink channel ─→ per-event Parquet（可选）
  ▼
RunSummary（可选写入 summary JSON，同时返回给 caller）
```

每层只回答一个问题：

| 层 | 回答的问题 | 主要输出 |
|---|---|---|
| schema declaration | 文件声称自己是什么？ | `InputFileSchema` |
| format loader | 文件内容是否符合声明？ | typed rows / grouped sessions |
| workload | 本次 run 如何使用这些已验证内容？ | `ReplayWorkload` |
| executor | 每个 workload unit 何时以及按何种依赖执行？ | concrete generation attempt |
| backend | 怎样发送请求并统一解释 text/media stream？ | `GenerationResult` |
| output | 实际发生了什么？ | `StepLog`、timeline、`RunSummary` |

## 2. 用户接口是 task + structured YAML

普通用户不直接拼接 Rust binary 的几十个 flags。支持的入口是：

```bash
uv run python -m launcher run configs/run.yaml
uv run python -m launcher sweep configs/sweep.yaml
uv run python -m launcher tracegen configs/tracegen.yaml
uv run python -m launcher selfcheck configs/selfcheck.yaml
```

`run` 和 `sweep` 共享同一组嵌套 blocks：

```text
input       输入文件、完整 format、tags
corpus      text corpus、tokenizer、token-pool limit（仅 text replay）
server      endpoint、backend、model、sampling
replay      arrival、capacity、context-limit 与 failure policy
measurement timeline 与可选 run-level SLO
output      一个 output directory；具体 artifact 文件名有稳定默认值
```

`sweep` 另外增加 `search` block，表达 mode、rate range 与停止条件。launcher 严格拒绝
unknown keys 和错误 value types，解析相对路径，build 对应 Rust binary，然后把 resolved
values 转成底层 argv。YAML 是 supported operator contract；Rust flags 是 launcher 与 engine
之间的内部接口。

`tracegen` 使用 `generator.type` 选择 `synthetic` 或 `coding-session`，其余字段放在对应
generator block；`selfcheck` 以独立 config 表达 tokenizer、output directory、pair count 与
owned loopback port。这样所有 Rust execution modes 都经过同一个 launcher lifecycle。

launcher 不读取 CSV、不构造 prompt，也不计算 metric。它只拥有 run 外围 lifecycle 和
terminal presentation。完整 engine stdout/stderr 写入 `terminal.log`；默认终端只展示 build、
replay progress、最终 workload/success/throughput/latency/cache summary 与 artifact paths。

## 3. Startup 先声明整份输入文件

CSV header 本身不足以决定语义。相同的 `input_len` 可能表示 independent request 的完整
prompt，也可能表示 session round 新追加的 suffix。因此 CLI 必须选择一个完整 format：

```text
--input-file-format text-generation-independent
--input-file-format text-generation-session-execution-v2
--input-file-format image-to-text-independent
```

startup 将 format 与额外 tags 合成为：

```rust
InputFileSchema {
    input_file_format: InputFileFormat,
    tags: Vec<TraceTag>,
}
```

三者的关系是：

- `InputFileFormat` 决定 physical columns、loader、结构规则以及 `RequestFamily`；
- `RequestFamily` 是 format 的派生属性，不是另一个 CLI selector，也不能逐 row 改变；
- `TraceTag` 增加与 family 正交的 column bundle，例如 `slo` 或 `priority`；
- `InputFileSchema` 是 base format 与合法 tags 合并后的精确 header contract。

### Benchmark 边界按 modality 组合

现有 CSV format 继续精确描述 trace artifact；新的 asset-backed benchmark
adapter 则把源数据转换为 `RequestSpec`。它的输入和输出是相互独立的 typed
list，均支持 text、image、audio、video 与 tensor，并允许混合或重复输入。

Backend 使用 `CapabilityProfile` 声明可接受的输入集合、可生成的输出集合，
以及是否支持混合输入和多个输出。因此扩展成本是 input encoder、output
observer 和 protocol adapter 的加和，而不是 modality pair 的矩阵。只有当模型
确实存在耦合语义时，才增加 pair-specific validation。

Asset-backed executor 把通用调度状态与自己的 client、`AssetStore` 组合；
text executor 则与 tokenizer 和 synthetic token pool 组合。这样媒体 workload
不会继承 text-only 的 corpus、prefix-cache preflight 或启动要求。稳定 contract
与扩展步骤见 [Adding modality-compositional benchmarks](docs/ADDING_BENCHMARKS.md)。

例如 `text-generation-session-execution-v2` 已经同时表达 text generation family 和
session execution layout。系统中不存在 `SessionExecutionV2 + ImageToText` 这样的半组合
状态，也不会根据 header 猜 family。

## 4. Schema parsing 之后究竟发生什么

`InputFileSchema` 只声明 contract；真正打开文件的是 `workload::load_workload`。它先根据
完整 format 选择唯一 loader：

```text
TextGenerationIndependent
  └─ format/text_generation/independent.rs::load
       └─ Vec<IndependentRequest>

TextGenerationSessionExecutionV2
  └─ format/text_generation/session.rs::load
       └─ SessionPlans
```

每个 family format 文件都拥有：

```text
COLUMNS
typed Row / runtime-ready row type
per-row validation
load(path, InputFileSchema)
```

`format/load_utils.rs` 只共享机械步骤：打开 CSV、核对 header、遍历 records、解析 tag
columns。它不决定 family，也不制造 generic request union。

### Independent 文件的输出

每行独立验证后成为一个 `IndependentRequest`：

```text
CSV record
  ↓ decode base fields + declared tag fields
IndependentRequest
  ├─ id / arrival_time
  ├─ input_len / output_len
  ├─ per-request SLO（若声明）
  └─ priority（若声明）
```

loader 最终返回 `Vec<IndependentRequest>`，文件顺序被保留。

### Session 文件的输出

session loader 先把一行解码为 `ExecutionRow`，再合并 tags，随后验证文件级结构：

- 同一 session 的 rows 连续；
- `round_idx` 从 0 连续递增；
- round 0 没有 prefix；
- 同一 session 的 arrival 一致；
- session blocks 按 arrival 排列；
- request id 唯一。

验证完成后，rows 被组织为：

```rust
type SessionPlans = Vec<(String, Vec<SessionRound>)>;
```

这个 grouping 属于 format parsing，因为 round order、prefix 和 tool wait 是输入 bytes 的
固有含义，不是本次 run 临时选择的 replay policy。

### 解析成功不等于当前 client 能执行

shared schema 定义了多个 request families。runtime 会执行两个 text format 与
asset-backed `multimodal-independent-v1`；只有 dimensions/token counts、没有实际 assets
的 media CSV 仍只可解析而不可执行。`load_workload` 会在 runtime boundary 拒绝它们，
不会凭空生成 media。

## 5. 为什么 loader 之后还有 `ReplayWorkload`

loaders 返回不同的 Rust 类型：

```text
independent::load(...) → Vec<IndependentRequest>
session::load(...)     → SessionPlans
multimodal::load(...)  → Vec<RequestSpec>
```

而 `--input-file-format` 到 runtime 才能确定，所以 `load_workload` 不能在编译时选择其中一个
返回类型。`ReplayWorkload` 只是承载这些可能结果的 sum type：

```rust
enum ReplayWorkload {
    IndependentRequests(Vec<IndependentRequest>),
    Sessions(SessionPlans),
    MultimodalRequests(Vec<RequestSpec>),
}
```

它没有重新解析 rows，也没有增加一种 schema。variant 直接拥有对应 loader 的原始输出；
整份文件只选择一个 variant，runner 最终只进入对应 executor。

这个 branch 本身无法消失，因为执行结构确实不同：independent requests 可以分别
release，multimodal requests 需要预先准备 assets，而 session rounds 必须按 predecessor
顺序 closed-loop 执行。若删除这个 enum，
只能把同一个 `match` 搬进 `runner.rs` 并让两条路径分别承担后续 setup，或者引入更复杂的
trait abstraction。把 session 强行 flatten 成 independent rows 则会丢失 dependency 和
tool-wait 语义。

因此这里保留的是最小的 runtime branch，不是第二套 input model。如果以后两个 format
共享完全相同的执行 shape，这个 enum 才应该合并或删除。

`workload.rs` 随后应用只属于本次运行的操作：

1. 先验证完整文件，再应用 `--max-items`；
2. 计算 trace 中 top-level workload units 的 arrival rate；
3. 如设置 `--rate`，统一缩放 top-level arrival offsets；
4. 生成 `WorkloadSummary`，包括 unit 数、step 数、最大 prompt/output 和 context-limit 信息。

先验证再截断是有意的：`--max-items 1` 不能隐藏文件后半段的坏 row。

session 的 workload unit 是 session，step 是 round；independent workload 中二者都是
request。offered rate 与 delivered step throughput 比较时必须通过
`steps_per_workload_unit` 转换，不能把两种单位直接相比。

## 6. 执行前准备

`runner.rs` 在启动 tasks 前完成一次性准备：

```text
WorkloadSummary / dry-run early return
  ├─ text → tokenizer + synthetic token pool → prefix-cache preflight
  └─ multimodal → 在 run_start 前验证/read/hash/base64 assets
  ↓
construct GenerationClient and protocol adapter
  ↓
create AppState, concurrency gate, log channel, optional timeline channel
```

`--dry-run` 在 token corpus 和网络访问前返回，因此它验证的是输入与 workload shaping，
不是 serving endpoint。

text replay 的 tokenizer 与 synthetic corpus 可由 `CorpusCache` 跨 sweep points 复用，
backend preflight 在正式 replay 前确认 server 能报告所需的 prefix-cache usage。
multimodal replay 没有 corpus/cache preflight；它在 arrival clock 前准备 immutable assets。

## 7. `executor/` 负责 release、dependency 与 admission

runner 对每个 top-level workload unit 启动一个 task，然后按 `ReplayWorkload` variant
进入三条执行路径（text independent、text session、multimodal independent）。

### Independent request

```text
wait for request arrival
  → acquire concurrency slot
  → draw input_len synthetic tokens
  → context-limit check
  → GenerationClient::run_step
  → StepLog
```

每个 independent request 独立 release，也独立持有 concurrency slot。

### Session

```text
wait for session arrival
  → acquire one concurrency slot for the whole session
  → for each round in order:
       build prompt from carried context
       context-limit check
       GenerationClient::run_step
       carry real output token ids into next round
       wait tool_wait_after_ms
  → release slot when the session ends
```

后续 round 不按原始 wall-clock arrival 独立 release；它是 closed-loop chain，必须等待
predecessor 完成及 tool wait。session 在所有 rounds 和 tool waits 期间持有同一个 capacity
slot，这是当前 concurrency contract。

`arrival_mode=trace` 尊重记录的 arrival offset；`arrival_mode=saturated` 忽略该 timeline，
让 units 尽快进入 admission。`--max-concurrency` 只限制 active units，并用 deterministic
admission order 处理竞争。

## 8. `tokens.rs` 把长度声明变成实际 prompt

输入文件保存长度和 prefix 关系，不保存本次 replay 的 concrete token ids。

independent request 从共享 synthetic token pool 取得 `input_len` 个 tokens。session round
由 `PromptBuilder` 构造：

```text
previous realized context[..prefix_len]
+ fresh synthetic tokens[input_len]
= prompt ids sent to server
```

round 完成后，builder 优先携带 server 返回的真实 output token ids，而不是重新生成假
output。`prefix_len` 表示计划中可复用的 prefix，不保证 server 实际命中；真实 cached
prompt tokens 由 backend usage 单独记录。

## 9. `backend/` 统一不同 serving 协议

executor 只提交 backend-neutral 的：

```rust
GenRequest {
    request_id,
    prompt: Prompt::Tokens(...),
    max_tokens,
    ...
}
```

`backend/dialect/` 负责 `wire/` 不覆盖的另一个维度：不是「哪种协议」，而是「哪套词汇」。
一个 [`Dialect`] 是一张 const 表，为某个服务系统声明：媒体如何附加到请求、模型
旋钮嵌套在哪里、流式媒体如何分帧、每个语义旋钮叫什么名字。trace 以模型中立的方式
声明生成参数（`steps`、`guidance`、`sample_rate_hz`），加上一个按 dialect 命名空间
划分的 `model_params`；dialect 将其渲染为该服务器的拼写，对没有对应名称的旋钮直接
丢弃而不是猜测。`dialect/` 不接触时间、并发或测量，只做重命名与重新嵌套。

`backend/wire/` 负责 OpenAI、vLLM native token endpoint 和 SGLang native token endpoint
之间的 JSON 差异。`GenerationClient` 负责共享 async lifecycle：

1. 构造并发送 payload；
2. 持续读取 stream；
3. 将 wire objects 标准化为 `StreamEvent`；
4. 折叠 text、token ids、usage、finish reason 与错误；
5. 检查 prompt echo 和 token accounting；
6. 返回 backend-neutral `GenerationResult`。生成媒体由 `media_client.rs` 统一
   JSON image、chat audio delta 与 raw PCM，记录 first-output、bytes、duration、
   RTF、artifact 和 timeline。

```rust
GenerationResult {
    outcome: GenerationOutcome,
    output_ids: Vec<u32>,
    timeline: Vec<TimelineEvent>,
}
```

这里测量 submit、send、first text/token-id、last token-id 和 response completion 等 clock。
TTFT、TPOT、E2E 与 arrival release lag 都从这些明确 clock 推导；它们是 outcome 的一部分，
不是控制整个架构的中心对象。

## 10. 结果怎样流向文件和 summary

executor 将 source declaration 与 `GenerationOutcome` 合成 `StepLog`：

```text
IndependentRequest / SessionRound
            +
GenerationOutcome
            ↓
         StepLog
```

之后有两条互不阻塞主执行的输出路径：

- log channel：`summary::write_logs` 写 per-step JSONL，同时折叠 replay metrics、prefix-cache
  metrics 和可选 SLO attainment；
- timeline channel：可选地在独立 blocking writer 中编码 per-event Parquet。channel 满时丢
  timeline sample，而不让磁盘或 Arrow encoding 对被测 submission 施加 backpressure。

所有 workload tasks 完成后，runner 等待 writers 收尾并构造：

```rust
RunSummary {
    workload,
    replay,
    client_runtime,
    timeline,
    slo,
}
```

它既作为 library return value 返回，也可写入 `--summary-path`。因此一次 run 的最终产物
不是只有 SLO：它同时报告输入规模、replay outcome、client runtime、timeline 完整度、
throughput、latency、prefix-cache fidelity，以及在有声明时的 SLO attainment。

## 11. Tags 是穿过主线的附加声明

tags 在 schema parsing 时进入 typed row，并随 source record 一直保留到 output：

```text
slo         → ttft_slo_ms, tpot_slo_ms, e2e_slo_ms
priority    → priority
session     → native independent layout 的 session-related columns
speculative → accept_rate
```

它们互不替代。特别地，priority 是 scheduling hint，不属于 SLO；SLO 的三个 metric 也逐
request、逐 metric 独立声明。根目录 `slo.rs` 只在执行后比较 declared bounds 与 measured
timings，并不是 schema loader 或 executor 的 owner。

## 12. Binary 与 library 边界

| 入口 | 作用 |
|---|---|
| `run` / `session_runner` | 执行一次真实 replay |
| `sweep` | 多次调用同一个 `run_once_reusing`，搜索 rate/SLO boundary，并复用 corpus |
| `tracegen` | 从 generator source materialize canonical session input files |
| `selfcheck` | 用受控 stub 验证 release、stream measurement、cache accounting 等 fidelity |

主 binary 与 sweep 都调用同一个 runner，而不是维护两套 replay path。其他 consumer 可以
共享 `schema` 的 format contract 与 loaders，但不会自动继承这个 HTTP client 的
`ReplayWorkload`、token construction 或 execution policy。

## 目录 owner map

| 路径 | 唯一职责 |
|---|---|
| `launcher/` | YAML validation、argv/build/run lifecycle、terminal UI |
| `schema/input_file_schema.rs` | 组合完整 format 与 tags，产生精确文件 contract |
| `schema/format/` | decode、validate 并组织 family-specific typed contents |
| `schema/family/` | family-specific declared value types |
| `schema/tag/` | orthogonal per-row declaration types |
| `workload.rs` | runtime dispatch、truncate、rate scaling、workload summary |
| `runner.rs` | 一次 run 的 wiring 与 lifecycle |
| `executor/` | arrival release、session dependency、admission、request lifecycle |
| `tokens.rs` | concrete token ids 与 session context carry-forward |
| `backend/wire/` | protocol-specific JSON shaping/parsing |
| `backend/dialect/` | 各服务系统的线上词汇：字段名、旋钮位置、媒体分帧 |
| `backend/client.rs` | shared token/text HTTP streaming engine 与 integrity measurements |
| `backend/media_client.rs` | generated image/audio transport 与 modality-neutral measurements |
| `record.rs` | per-step source + outcome JSONL contract |
| `timeline.rs` | optional per-event recording |
| `summary.rs` | replay、runtime、timeline 与 run-level aggregation |
| `slo.rs` | 可选的 declared-vs-measured SLO evaluation |

最短的心智模型是：

```text
Schema 说明文件是什么。
Format loader 把 bytes 变成已验证的 typed contents。
Workload 决定本次怎样 replay。
Executor 把计划按时间和依赖变成实际请求。
Backend 把请求变成 stream outcome。
Record 与 Summary 保存实际发生的事情。
```
