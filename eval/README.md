# RAG 检索实验 · 使用说明

> 对应论文章节：第 5 章（混合检索与精排）、第 7 章（实验与结果分析）。
> 检索实现见 `app/core/rag/`，本目录只放评测数据与结果。

## 1. 文件说明

| 文件 | 说明 |
|---|---|
| `dataset.json` | 评测集（冻结版），每题只标 query → 相关 chunk id |
| `chunks.json` | 由 `tools/list_chunks.py` 导出，供标注；已 gitignore |
| `results/{strategy}_{timestamp}.json` | 评测输出：配置快照、数据集哈希、git 状态、逐题结果 |
| `README.md` | 本文件 |

## 2. 评测集规范与标注

题型 3 类，各 10–15 条：

- `term`：术语 / 编号精确匹配
- `cross_page`：跨段、跨页归纳
- `paraphrase`：同义改写、口语化提问

规则：

- `relevant_chunk_ids` 用 Chroma 原始 id，格式 `{doc_id}_{chunk_index}`
- 每题至少 1 个相关 chunk，允许多个；只标相关性，不写标准答案
- 数据集一旦用于正式对照实验即冻结；增补后必须重新全量跑

标注流程：

```bash
# 1. 导出全部 chunk（当前 108 条）
.venv/Scripts/python.exe tools/list_chunks.py --out eval/chunks.json

# 2. 关键词搜候选（结果写到 D:/cc-rubbish/search_results.txt）
.venv/Scripts/python.exe tools/search_chunks.py --keyword 时效
.venv/Scripts/python.exe tools/search_chunks.py --id 2a1ca15f03db4c54806d572f7eb44a28_3

# 3. 人工确认后，把 chunk id 填进 eval/dataset.json
```

冻结校验：

```bash
sha256sum eval/dataset.json
```

结果 JSON 里的 `dataset.sha256` 必须等于数据集冻结值，否则两次实验不可比。

## 3. 跑评测

```bash
.venv/Scripts/python.exe tools/evaluate.py \
  --dataset eval/dataset.json \
  --strategies vector,hybrid,hybrid_rerank \
  --topk 3,5,10
```

口径：

- 检索深度 `D = max(topk)`；每个 query 在每个策略下只检索一次，K 只是截断
- 每个 query 只取一次 embedding，三个策略复用
- 二值相关性，指标为 **macro 平均**（先算每题，再对题平均）
- `Recall@K = |R ∩ L_K| / |R|`
- `MRR = 1 / 首个相关项名次`（深度 D 内）
- `nDCG@K = DCG@K / IDCG@K`，`rel_i ∈ {0, 1}`

输出：

- 控制台：总体表 + 按题型分组表
- `eval/results/{strategy}_{timestamp}.json`：配置快照、`dataset.sha256`、`git.head`、`git.diff_sha256`、`git.status_porcelain`、耗时、逐题命中与指标

冒烟 / 中间结果不建议入库；正式实验批次可用 `git add -f eval/results/<file>.json` 归档，论文表格从这些 JSON 生成，不手抄。

## 4. 切换策略

改 `.env` 后重启服务：

```env
RAG_STRATEGY=vector          # vector | hybrid | hybrid_rerank
RAG_RECALL_K=50              # 单路召回条数
RRF_K=60                     # 加权 RRF 常数
RRF_WEIGHT_VECTOR=0.5
RRF_WEIGHT_BM25=0.5
RERANK_TOP_N=20              # 送精排的候选数
RERANK_MODEL=gte-rerank-v2
```

| 策略 | 链路 |
|---|---|
| `vector` | embed → Chroma(top_k) |
| `hybrid` | embed → Chroma(RECALL_K) + BM25(RECALL_K) → 加权 RRF → top_k |
| `hybrid_rerank` | hybrid 召回 → RRF 前 RERANK_TOP_N → DashScope 精排 → top_k |

注意：

- `AliQwen_API_KEY` 由进程环境提供，不写进 `.env`
- 精排失败会抛 `BizException`，不会静默降级为纯向量
- 实验期间固定 embedding 模型、chunk 配置与语料；消融只做单因素（`RRF_WEIGHT_BM25` 或 `RRF_K` 二选一）

## 5. 回归

先启动服务，再运行：

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

`tests/test_fusion.py` 是纯函数测试，不需要服务和 API Key。