# Design Template

## Problem

Xây dựng một research assistant nhận câu hỏi nghiên cứu (research query), tìm nguồn, phân
tích, và viết câu trả lời cuối có trích dẫn. So sánh hai cách làm: single-agent baseline
(1 lần gọi LLM, không tool) và multi-agent workflow (Supervisor điều phối Researcher,
Analyst, Writer, Critic), theo latency, cost, quality, citation coverage, failure rate.

## Why multi-agent?

Single-agent baseline chỉ trả lời từ kiến thức nội tại của model, không có tool/search nên
không thể trích dẫn nguồn và dễ trả lời chung chung. Multi-agent xứng đáng ở đây vì task cần
3 loại xử lý thông tin khác nhau: (1) truy xuất bằng chứng từ nguồn ngoài, (2) so sánh/phân
tích nhiều nguồn trước khi kết luận, (3) một bước review độc lập kiểm tra trích dẫn - ba nhu
cầu "information/verification" khác nhau mà một lần generate duy nhất khó làm tốt cả ba.

Benchmark thật (xem `reports/benchmark_report.md`, chạy 2026-08-20) xác nhận: multi-agent
đạt quality 10.0/10 và citation coverage 100%, so với baseline 6.0/10 và không có trích dẫn
nào - đổi lại latency trung bình cao hơn ~3-6 lần (~24-30s so với ~5-10s) và cost cao hơn
~4-7 lần (~$0.0015 so với ~$0.0003 mỗi query).

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Quyết định agent nào chạy tiếp theo; enforce guardrail (`max_iterations`); retry/fallback khi 1 bước fail | Toàn bộ `ResearchState` | Một quyết định route (`researcher`/`analyst`/`writer`/`critic`/`done`), ghi vào `route_history` | Nếu vượt `max_iterations` (mặc định 6) -> dừng ngay (`done`) bất kể trạng thái, tránh loop vô hạn |
| Researcher | Tìm nguồn (Tavily nếu có key, ngược lại offline corpus), tóm tắt `research_notes` kèm trích dẫn `[source_id]` | `request.query` | `sources`, `research_notes` | Search 0 kết quả -> ghi `research_notes` là thông báo "không tìm thấy nguồn" (không tính là lỗi); exception (API lỗi/timeout) -> ghi vào `state.errors`, Supervisor retry tối đa 2 lần |
| Analyst | Trích xuất claim chính, so sánh nguồn, gắn cờ evidence yếu | `research_notes`, `sources` | `analysis_notes` | Lỗi lặp lại 2 lần -> Supervisor coi bước này là optional, **bỏ qua** và cho Writer chạy thẳng từ `research_notes` thay vì kẹt pipeline |
| Writer | Tổng hợp câu trả lời cuối, giữ trích dẫn `[source_id]` | `research_notes`, `analysis_notes` | `final_answer` | Lỗi lặp lại 2 lần -> Supervisor dừng hẳn (`done`), `final_answer` giữ `None`, lỗi được ghi lại để debug |
| Critic (bonus) | Kiểm tra citation coverage, review câu trả lời cuối | `final_answer`, `sources` | Ghi chú review (không sửa `final_answer`) | Citation coverage < 50% -> ghi cảnh báo vào `state.errors` (không chặn pipeline, chỉ để cảnh báo chất lượng) |

## Shared state

`core/state.py: ResearchState` gồm:

- `request`, `iteration`, `route_history` - điều khiển routing.
- `sources`, `research_notes`, `analysis_notes`, `final_answer` - dữ liệu handoff chính giữa
  các agent, đủ để agent sau không cần gọi lại agent trước.
- `agent_results` - lưu `AgentResult` mỗi bước (agent, nội dung, `metadata` gồm token/cost)
  để `evaluation/benchmark.py` tính cost và quality mà không cần agent tự báo cáo riêng.
- `trace` - danh sách span (`name`, `attributes`, `duration_seconds`) cho debug/observability,
  đồng thời là nguồn cho bảng "Pipeline Trace" trong CLI/UI.
- `errors` - vừa là log lỗi, vừa được Supervisor dùng làm **bộ đếm retry** (đếm số lỗi có
  prefix `"<agent_name>: "`) - tận dụng một field cho hai mục đích thay vì thêm state riêng.

## Routing policy

Deterministic, không dùng LLM để quyết định route (tránh tốn thêm 1 lần gọi model chỉ để
routing, và tránh routing không ổn định giữa các lần chạy). Logic trong
`agents/supervisor.py::_decide`:

1. `iteration >= max_iterations` -> `done` (guardrail cứng, luôn kiểm tra đầu tiên).
2. `research_notes is None` -> `researcher` (retry tối đa 2 lần nếu fail, sau đó `done`).
3. `analysis_notes is None` -> `analyst` (retry tối đa 2 lần, sau đó **fallback**: bỏ qua,
   sang thẳng `writer`).
4. `final_answer is None` -> `writer` (retry tối đa 2 lần, sau đó `done`).
5. `"critic" not in route_history` -> `critic` (chạy đúng 1 lần, như quality gate).
6. Ngược lại -> `done`.

## Guardrails

- **Max iterations**: `MAX_ITERATIONS` (mặc định 6, đọc qua `core/config.Settings`) - guard
  cứng trong `SupervisorAgent._decide`, luôn được kiểm tra trước mọi logic khác.
- **Timeout**: `TIMEOUT_SECONDS` (mặc định 60s) áp dụng cho OpenAI client và Tavily HTTP
  request.
- **Retry**: tối đa 2 lần/bước cho `researcher`/`analyst`/`writer`, đếm qua
  `state.errors` với prefix `"<agent>: "`.
- **Fallback**: `analyst` fail liên tục -> bỏ qua, `writer` chạy thẳng từ `research_notes`;
  `researcher`/`writer` fail liên tục -> dừng hẳn, giữ lại phần `state` đã có thay vì crash.
- **Validation**: `ResearchQuery` (Pydantic) validate độ dài query >= 5 ký tự, `max_sources`
  trong khoảng 1-20; `BenchmarkMetrics` validate các score trong khoảng hợp lệ (0-10, 0-1).

## Benchmark plan

3 query cấu hình trong `configs/lab_default.yaml`. Với mỗi query, chạy cả baseline và
multi-agent qua `evaluation/benchmark.run_benchmark`, đo latency (wall-clock), cost (tổng
`cost_usd` từ `agent_results`), citation coverage (tỉ lệ `source_id` xuất hiện trong
`final_answer`), quality (heuristic 0-10 dựa trên coverage/độ dài/không lỗi - không thay thế
rubric người chấm), và failure rate. Kết quả render bằng `evaluation/report.render_markdown_report`
ra `reports/benchmark_report.md`. Chạy qua `malab benchmark`.

Kết quả thật (2026-08-20, xem `reports/benchmark_report.md` để có số liệu mới nhất):

| Run | Quality | Citation cov. | Latency TB | Cost TB |
|---|---:|---:|---:|---:|
| baseline | 6.0/10 | - | ~5-10s | ~$0.0003 |
| multi-agent | 10.0/10 | 100% | ~24-30s | ~$0.0015 |
