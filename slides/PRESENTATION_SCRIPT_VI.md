# Script thuyết trình seminar MedRaC

Deck gồm 21 slide nội dung và Slide 22 Q&A. Tổng phần nói mục tiêu là khoảng 18 phút 45 giây; thời gian còn lại dành cho chuyển phần và xử lý phát sinh. Nội dung hiển thị bằng tiếng Anh, phần trình bày bằng tiếng Việt.

## I. Motivation and Research Problem

### Slide 1 — Title (0:20)

Kính chào thầy cô và các bạn. Nhóm K23 xin trình bày paper *From Scores to Steps: Diagnosing and Improving LLM Performance in Evidence-Based Medical Calculations*. Nhóm gồm Dương Tuấn Anh, Nguyễn Gia Bảo và Lại Nguyễn Hồng Thanh, dưới sự hướng dẫn của thầy Nguyễn Tiến Huy và thầy Lê Thanh Tùng.

### Slide 2 — Presentation Outline (0:15)

Bài trình bày gồm sáu phần. Chúng ta bắt đầu từ hạn chế của final-answer accuracy, sau đó đi qua Evaluation Framework, MedRaC Methodology, kết quả paper, live demo ngắn, và phần thảo luận.

### Slide 3 — Medical Calculations Require More Than Arithmetic (0:45)

Medical calculation không chỉ là thay số. Model phải đọc clinical note, chọn đúng calculator, trích xuất đúng biến và tính toán đúng. Vì vậy, creatinine clearance, corrected sodium và Wells score đều đòi hỏi nhiều năng lực hơn arithmetic.

### Slide 4 — Final-Answer Accuracy Can Hide Invalid Reasoning (1:15)

Equation (1) là corrected-sodium formula đúng. Equation (2) dùng formula sai nhưng vẫn cho một con số plausible và có thể được chấp nhận bởi tolerance rộng. Vì vậy, đáp án có vẻ đúng chưa chứng minh reasoning hợp lệ.

### Slide 5 — Research Questions and Contributions (0:35)

Paper hỏi ba câu: đánh giá từng stage thế nào, lỗi bắt đầu ở đâu, và external tools có giảm lỗi không. Ba contribution tương ứng là step-wise evaluation, structured error attribution và MedRaC.

## II. Evaluation Framework

### Slide 6 — Medical Calculation as a Four-Stage Process (1:10)

Figure 1 phân rã bài toán thành Formula, Extraction, Calculation và Answer. Mỗi stage có thể sai độc lập và lỗi phía trước sẽ ảnh hưởng bước sau.

### Slide 7 — Strict Correctness Requires Every Stage to Pass (1:10)

Equation (3) là strict correctness. Một case chỉ đúng khi cả F, E, C và A đều valid. Đây là tiêu chuẩn loại bỏ những đáp án cuối tình cờ đúng.

### Slide 8 — Conditional Correctness Isolates Stage Reliability (1:10)

Equation (4) đo khả năng một stage đúng khi các prerequisite stage đã đúng. Metric này tách reliability thật của stage hiện tại khỏi ảnh hưởng lỗi upstream.

### Slide 9 — First Error Attribution Identifies the Earliest Failure (1:10)

Equation (5) phân bổ một case sai cho stage sai đầu tiên. Nhờ đó, framework chỉ rõ cần ưu tiên cải thiện formula knowledge, extraction hay calculation.

### Slide 10 — Error Taxonomy Turns Failures into Diagnoses (0:45)

Figure 2 gom lỗi thành knowledge, clinical interpretation và numerical execution. Taxonomy giúp biến một accuracy tổng quát thành failure mode có thể can thiệp.

## III. MedRaC Methodology

### Slide 11 — MedRaC Grounds Formulas and Executes Calculations (1:15)

Figure 3 mô tả MedRaC. LLM đọc clinical text; Formula RAG cung cấp formula; pipeline trích xuất values, sinh code và thực thi code để tạo result deterministic.

### Slide 12 — Each MedRaC Component Targets a Distinct Failure Mode (0:55)

Table 1 ánh xạ từng component với một nhóm lỗi. Retrieval nhắm formula error, execution nhắm arithmetic error; clinical interpretation vẫn là điểm khó.

## IV. Paper Experiments and Results

### Slide 13 — Experimental Setup for Calculation and Clinical Scoring (0:25)

Paper dùng 55 calculators và 940 cases sau cleaning, gồm equation-based và rule-based tasks. Các method được đánh giá bằng final-answer và step-wise evaluation.

### Slide 14 — Main Results Across Models and Prompting Methods (0:50)

Đây là bảng main results đầy đủ của paper. Trên các model có kết quả MedRaC được báo cáo, calculation accuracy đều tăng so với One-shot; gain lớn nhất xuất hiện ở các open model nhỏ. Ngược lại, rule-based results còn mixed, cho thấy clinical interpretation vẫn là năng lực giới hạn.

### Slide 15 — Main Result: MedRaC Improves Equation-Based Tasks (1:15)

Slide này phóng to kết luận quan trọng nhất từ bảng trước. MedRaC tăng equation-based accuracy thêm 51.92 points với Phi-4-mini, 49.25 points với LLaMA3.1-8B, và 10.15 points với GPT-4o. Rule-based tasks cho gain mixed vì clinical interpretation vẫn chi phối.

### Slide 16 — Step-Wise Evaluation Aligns Closely with Expert Judgment (0:50)

Table 4 cho thấy LLM Judge có agreement cao với expert ở Formula và Answer. Extraction thấp hơn, phù hợp với việc extraction là một bottleneck khó.

### Slide 17 — Error Reductions Reveal MedRaC's Remaining Boundary (1:05)

Figure 4 cho thấy formula, arithmetic và demographic errors giảm mạnh. Incorrect extraction và clinical misinterpretation giảm ít hơn. MedRaC đặc biệt hiệu quả với tool-addressable errors.

## V. Local Demo and Mini-Experiment

### Slide 18 — Local Live Demo Makes the F--E--C--A Pipeline Observable (0:30)

Tiếp theo, nhóm chạy live demo. Khi chạy, chỉ theo dõi patient input, retrieved formula, extracted values, generated code và final result; không cần giới thiệu UI.

### Slide 19 — Local Mini-Experiment (0:45)

Slide này trình bày hai bảng kết quả trên cùng 10 samples cố định, gồm 7 equation-based và 3 rule-based cases. Bảng bên trái là lượt chạy end-to-end với rule-based evaluator: Plain CoT 9/10 và MedRaC 10/10, cả hai hoàn thành 10/10 cases. Bảng bên phải là đánh giá hậu kỳ LLM F--E--C--A trên các output đã lưu: Plain CoT đạt F/E/C/A là 10/9/10/8 và strict 8/10; MedRaC là 10/10/10/9 và strict 9/10.

## VI. Discussion and Conclusion

### Slide 20 — Discussion (1:00)

Paper còn giới hạn ở structured English cases, chất lượng formula bank, LLM Judge và human validation. Hướng tiếp theo là tăng khả năng clinical interpretation, mở rộng validation và đưa human review vào pipeline.

### Slide 21 — Conclusions (0:45)

Kết luận: final-answer accuracy là chưa đủ; F--E--C--A giúp đo và chẩn đoán reasoning errors; MedRaC nâng formula grounding và arithmetic reliability, còn clinical interpretation là bài toán mở.

### Slide 22 — Questions and Discussion

Cảm ơn thầy cô và các bạn. Nhóm sẵn sàng trao đổi.
