# Exit Ticket

## 1. Case nào nên dùng multi-agent? Vì sao?

Câu hỏi nghiên cứu cần tổng hợp nhiều nguồn, đòi hỏi trích dẫn/kiểm chứng, và chất lượng /
độ tin cậy quan trọng hơn tốc độ trả lời. Dẫn chứng từ benchmark thật
(`reports/benchmark_report.md`): multi-agent đạt quality 10/10 và citation coverage 100%,
so với baseline chỉ 6/10 và không có trích dẫn nào (vì baseline không dùng search/tool).
Lý do sâu hơn: multi-agent tách được 3 nhu cầu xử lý thông tin khác nhau - tìm nguồn
(Researcher), so sánh/đánh giá bằng chứng (Analyst), viết có kiểm chứng (Writer) và review
lại (Critic) - mỗi bước cần một loại "context" và tiêu chí đúng/sai khác nhau, khó để một
lần generate duy nhất làm tốt cả bốn việc cùng lúc mà vẫn giữ được trích dẫn chính xác.

## 2. Case nào không nên dùng multi-agent? Vì sao?

Câu hỏi đơn giản, cần trả lời nhanh, không yêu cầu trích dẫn nguồn ngoài, hoặc latency/cost
là ưu tiên hàng đầu (chat thời gian thực, autocomplete, câu hỏi mà kiến thức nội tại của
model đã đủ tốt). Dẫn chứng thật: multi-agent tốn trung bình ~24-30s và ~$0.0015/query, gấp
khoảng 3-6 lần thời gian và 4-7 lần chi phí so với baseline (~5-10s, ~$0.0003/query) - theo
`reports/benchmark_report.md`. Khi câu hỏi không thực sự cần bằng chứng từ nguồn ngoài, phần
overhead điều phối (Supervisor routing, 4 lần gọi LLM thay vì 1) chỉ làm tăng chi phí/độ trễ
mà không cải thiện chất lượng tương ứng - đúng như nguyên tắc "Không thêm agent nếu không có
lý do rõ ràng" nêu trong `docs/lab_guide.md`.
