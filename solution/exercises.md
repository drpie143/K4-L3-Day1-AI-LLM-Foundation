# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Ở temperature 0.0, phản hồi có xu hướng ổn định, có cấu trúc rõ ràng và tập trung vào các thông tin quen thuộc. Khi tăng lên 0.5, 1.0 và 1.5, model thay đổi cách chọn sự thật, ví dụ và cách diễn đạt, nên câu trả lời đa dạng hơn nhưng không nhất thiết chính xác hoặc nhất quán hơn. Trong lần chạy này, các phản hồi đều bị giới hạn ở giữa danh sách vì `max_tokens=256`, còn latency lần lượt khoảng 4.62, 4.31, 3.74 và 3.72 giây.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature khoảng 0.2–0.3 cho chatbot hỗ trợ khách hàng. Mức thấp giúp câu trả lời ổn định, nhất quán và bám sát thông tin sản phẩm, đồng thời giảm nguy cơ model tự tạo thông tin không có trong dữ liệu. Nếu chatbot có một chức năng sáng tạo riêng, chẳng hạn gợi ý cách diễn đạt quảng cáo, có thể dùng temperature cao hơn cho chức năng đó.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Mỗi ngày có khoảng `10.000 × 3 × 350 = 10.500.000` token output, tương đương 10.500 đơn vị 1.000 token. Theo bảng giá của lab, GPT-4o tốn khoảng `10.500 × 0,010 = 105 USD`, còn GPT-4o-mini tốn khoảng `10.500 × 0,0006 = 6,30 USD`, nên GPT-4o đắt hơn khoảng `105 / 6,30 = 16,67 lần`. GPT-4o phù hợp với phân tích phức tạp, câu trả lời chuyên môn hoặc tác vụ có hậu quả cao; mini phù hợp với FAQ, phân loại yêu cầu và các tác vụ lặp lại có số lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phản hồi dành cho giáo viên tiểu học dùng phép so sánh cuốn sổ, các trang và chuỗi khối, với từ vựng đơn giản để trẻ 8 tuổi dễ hình dung. Phản hồi dành cho chuyên gia tài chính dài và kỹ thuật hơn, sử dụng các khái niệm như sổ cái phân tán, phi tập trung, hash, block và giao dịch được xác minh. System prompt định hướng vai trò, đối tượng người đọc, mức độ chuyên sâu, giọng điệu và kiểu ví dụ mà model lựa chọn. Vì vậy cùng một câu hỏi nhưng persona khác nhau tạo ra cách giải thích khác nhau.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Đoạn văn có 95 từ. `count_tokens` trả về 107 token, trong khi cách ước lượng `số từ / 0,75` cho kết quả `95 / 0,75 = 126,67 token`; chênh lệch là 19,67 token, tương đương khoảng 15,53% so với con số ước lượng. Đây chỉ là ước lượng thô vì số token phụ thuộc tokenizer và model chứ không chỉ phụ thuộc số từ. Tiếng Việt có dấu, nhiều âm tiết và cách tách từ khác tiếng Anh, nên một từ hoặc cụm từ tiếng Việt có thể bị chia thành nhiều token hơn; tỷ lệ thực tế còn phụ thuộc nội dung và bộ mã hóa.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng với chatbot, trợ lý học tập và các tác vụ tạo nội dung dài vì người dùng nhìn thấy phần đầu của câu trả lời ngay khi model đang sinh phần còn lại, nhờ đó giảm cảm giác phải chờ. Nó cũng cho phép ứng dụng hiển thị tiến trình và dừng sớm nếu người dùng không cần câu trả lời tiếp tục. Non-streaming phù hợp với câu trả lời ngắn, các tác vụ backend cần nhận đủ dữ liệu rồi mới xử lý, hoặc trường hợp cần kiểm tra và lưu toàn bộ phản hồi trước khi hiển thị.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng thời gian chờ sau mỗi lần lỗi, chẳng hạn 0,1 rồi 0,2 và 0,4 giây, nên client giảm tốc độ gửi request khi server đang quá tải nhưng vẫn thử lại nhanh trong những lỗi đầu tiên. Cách này giúp phân tán các lần retry và cho server thời gian phục hồi. Nếu hàng nghìn client cùng chờ một khoảng cố định rồi retry cùng lúc, request sẽ dồn thành từng đợt, tạo hiệu ứng thundering herd và có thể làm tình trạng quá tải kéo dài. Trong thực tế có thể thêm jitter ngẫu nhiên để các client không retry đúng cùng thời điểm.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona tôi chọn là: “Bạn là trợ giảng của khóa AI thực chiến, trả lời ngắn gọn bằng tiếng Việt và giải thích khái niệm theo từng bước bằng ví dụ dễ hiểu.” Từ “ngắn gọn” giúp kiểm soát độ dài, còn “bằng tiếng Việt” bảo đảm câu trả lời phù hợp với người dùng của khóa học. Yêu cầu giải thích theo từng bước giúp người mới học dễ theo dõi hơn.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là history chỉ giữ ba lượt gần nhất, nên trợ lý có thể quên mục tiêu hoặc thông tin người dùng đã nói ở các lượt cũ. Cách cải thiện cụ thể là tạo một bản tóm tắt hội thoại sau mỗi vài lượt và lưu bản tóm tắt đó ngoài history ngắn hạn. Trước mỗi request, chương trình có thể đưa bản tóm tắt liên quan vào messages cùng với sáu message gần nhất; khi history bị cắt, các thông tin quan trọng vẫn được giữ lại mà không làm số token tăng vô hạn.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
