# Script thuyết trình seminar MedRaC

Deck gồm 21 slide nội dung và Slide 22 Q&A. Phần trình bày mục tiêu khoảng 20 phút, chưa tính phần trao đổi. Nội dung trên slide bằng tiếng Anh; phần nói bằng tiếng Việt.

Thông điệp xuyên suốt là: **trong tính toán y khoa, một con số đúng chưa chắc được tạo ra bởi một quá trình đúng**.

## I. Motivation and Research Problem

### Slide 1 — Title (0:25)

Kính chào thầy cô và các bạn. Nhóm K23 xin trình bày paper *From Scores to Steps: Diagnosing and Improving LLM Performance in Evidence-Based Medical Calculations*.

Paper đặt ra một câu hỏi quan trọng: nếu LLM đưa ra đúng con số, liệu chúng ta có chắc mô hình đã chọn đúng công thức, đọc đúng bệnh án và tính toán đúng hay không?

Nhóm gồm Dương Tuấn Anh, Nguyễn Gia Bảo và Lại Nguyễn Hồng Thanh, dưới sự hướng dẫn của thầy Nguyễn Tiến Huy và thầy Lê Thanh Tùng.

### Slide 2 — Presentation Outline (0:20)

Trước hết, nhóm sẽ giải thích medical calculation là gì và vì sao đây không chỉ là số học. Tiếp theo là framework đánh giá từng bước, phương pháp MedRaC, kết quả thực nghiệm, demo cục bộ và phần thảo luận.

Điểm cần theo dõi không chỉ là MedRaC cải thiện bao nhiêu phần trăm, mà là paper đã thay đổi cách định nghĩa một lời giải đúng như thế nào.

### Slide 3 — Medical Calculations Require More Than Arithmetic (1:40)

Medical calculation là việc sử dụng dữ kiện của bệnh nhân để tính một chỉ số hoặc điểm số hỗ trợ quyết định lâm sàng. Paper chia bài toán này thành hai nhóm.

Nhóm thứ nhất là **equation-based**, tức sử dụng một công thức toán học. Ví dụ, **Creatinine Clearance**, viết tắt là CrCl, là chỉ số ước tính khả năng lọc creatinine của thận trong một phút và thường được dùng như một chỉ dấu về chức năng thận.

Để tính CrCl, bác sĩ cần các dữ kiện như tuổi, cân nặng, giới tính và nồng độ creatinine trong máu. Kết quả giúp theo dõi mức độ suy giảm chức năng thận và hỗ trợ điều chỉnh liều của một số loại thuốc được đào thải qua thận. Vì vậy, nếu mô hình lấy sai cân nặng, bỏ hệ số giới tính hoặc dùng sai đơn vị, kết quả có thể ảnh hưởng nghiêm trọng đến đánh giá và sử dụng thuốc.

Nhóm thứ hai là **rule-based**, ví dụ Wells score để đánh giá khả năng thuyên tắc phổi. Hệ thống phải đọc bệnh án, xác định bệnh nhân có thỏa mãn từng tiêu chí hay không, gán điểm rồi cộng lại để phân tầng nguy cơ.

Điểm khó là bệnh án không viết sẵn “tiêu chí này đúng”. Mô hình phải hiểu một mô tả có phải dấu hiệu DVT hay không, hoặc triệu chứng đang xuất hiện hay đã được phủ định.

Như vậy, equation-based chủ yếu yêu cầu đúng công thức, đúng biến và đúng phép tính. Rule-based còn đòi hỏi diễn giải đúng ngữ cảnh lâm sàng. Medical calculation vì thế cần nhiều hơn arithmetic.

### Slide 4 — Final-Answer Accuracy Can Hide Invalid Reasoning (1:10)

Vấn đề xuất hiện khi benchmark chỉ so sánh con số cuối với đáp án chuẩn.

Trong ví dụ corrected sodium của paper, công thức đúng cho kết quả 137,248. Mô hình lại dùng sai hệ số và sai công thức, nhưng vẫn tính ra 135,432.

Do benchmark cũ cho phép sai số ±5%, đáp án này vẫn có thể được đánh dấu đúng. Tuy nhiên, mô hình thực chất chỉ **tính đúng trên một công thức sai**. Với dữ liệu bệnh nhân khác, sai công thức có thể tạo ra chênh lệch lớn hơn.

Vì vậy, final-answer accuracy cho biết con số có gần đáp án hay không, nhưng chưa cho biết quá trình có đáng tin hay không. Đây chính là vấn đề trung tâm mà paper muốn giải quyết.

### Slide 5 — Research Questions and Contributions (0:50)

Từ vấn đề đó, paper đặt ra ba câu hỏi.

Thứ nhất, làm sao đánh giá riêng từng bước của lời giải? Thứ hai, khi một case sai, lỗi bắt đầu ở đâu? Và thứ ba, external tools có thể giảm những lỗi này hay không?

Ba đóng góp tương ứng là: **step-wise evaluation** trên benchmark đã được làm sạch; **structured error attribution** để chẩn đoán lỗi; và **MedRaC**, kết hợp Formula RAG với Python execution.

Có thể tóm tắt logic của paper bằng ba bước: **đánh giá, chẩn đoán và can thiệp**.

## II. Evaluation Framework

### Slide 6 — Medical Calculation as a Four-Stage Process (1:05)

Paper phân rã một lời giải thành bốn stage: Formula, Extraction, Calculation và Answer, viết tắt là F–E–C–A.

**Formula** kiểm tra mô hình có chọn đúng calculator, công thức, hệ số và đơn vị hay không. **Extraction** kiểm tra các dữ kiện cần thiết có được lấy đúng từ bệnh án không. **Calculation** kiểm tra các phép toán trung gian. Cuối cùng, **Answer** so sánh kết quả với ground truth.

Với Creatinine Clearance, Formula là công thức được sử dụng; Extraction là tuổi, cân nặng, giới tính và serum creatinine; Calculation là quá trình thay số; còn Answer là giá trị CrCl cuối cùng.

Nhờ cách phân rã này, thay vì chỉ biết một case sai, chúng ta biết chính xác lời giải đã hỏng ở bước nào.

### Slide 7 — Strict Correctness Requires Every Stage to Pass (0:45)

Từ bốn stage trên, paper định nghĩa strict correctness bằng phép AND. Một case chỉ được xem là đúng khi Formula, Extraction, Calculation và Answer đều đúng.

Chỉ cần một stage sai thì \(\kappa\), tức độ đúng tổng thể, bằng 0. Tiêu chuẩn này loại bỏ những trường hợp mô hình dùng sai công thức hoặc sai dữ kiện nhưng tình cờ tạo ra một đáp án gần đúng.

Nói cách khác, paper không đánh giá sự may mắn của một kết quả, mà đánh giá tính hợp lệ của toàn bộ quá trình.

### Slide 8 — Conditional Correctness Isolates Stage Reliability (0:50)

Strict correctness cho biết một case có đúng toàn bộ hay không. Conditional Correctness lại đo độ tin cậy của từng stage.

Cách hiểu đơn giản là: **nếu tất cả bước trước đã đúng, mô hình có làm đúng bước hiện tại hay không?**

Ví dụ, khi đánh giá Calculation, paper chỉ xét những case đã đúng Formula và Extraction. Nếu Calculation vẫn sai, lỗi thực sự nằm ở numerical execution chứ không phải ở công thức hay dữ kiện đầu vào.

Metric này giúp tách lỗi của stage hiện tại khỏi ảnh hưởng của các lỗi upstream.

### Slide 9 — First Error Attribution Identifies the Earliest Failure (0:50)

First Error Attribution trả lời một câu hỏi khác: **trong những case thất bại, lỗi đầu tiên xuất hiện ở đâu?**

Nếu mô hình chọn sai công thức ngay từ đầu, nguyên nhân gốc là Formula, dù các bước sau cũng sai. Nếu công thức và dữ kiện đúng nhưng phép tính sai, lỗi đầu tiên là Calculation.

Metric này cho biết nên ưu tiên cải thiện capability nào. Formula FE cao gợi ý cần retrieval; Extraction FE cao cho thấy vấn đề đọc bệnh án; còn Calculation FE cao cho thấy code execution có thể hữu ích.

### Slide 10 — Error Taxonomy Turns Failures into Diagnoses (0:50)

Sau khi xác định stage bị lỗi, paper tiếp tục phân loại nguyên nhân.

Nhóm **knowledge** gồm chọn sai hoặc bịa công thức và bỏ hệ số hiệu chỉnh. Nhóm **clinical interpretation** gồm lấy sai dữ kiện, bỏ sót biến hoặc hiểu sai bệnh án. Nhóm **numerical execution** gồm sai đơn vị, sai số học và làm tròn.

Mỗi nhóm cần một giải pháp khác nhau. Sai công thức cần grounding kiến thức; sai số học có thể dùng chương trình; nhưng hiểu sai bệnh án không thể được sửa chỉ bằng một máy tính.

Đây là cơ sở để paper thiết kế MedRaC.

## III. MedRaC Methodology

### Slide 11 — MedRaC Grounds Formulas and Executes Calculations (1:05)

MedRaC là một pipeline training-free, không cần fine-tune lại LLM. Ý tưởng là giao từng phần cho thành phần phù hợp hơn.

Đầu tiên, Formula RAG truy xuất công thức hoặc quy tắc liên quan từ formula bank. LLM dùng thông tin đó để trích xuất dữ kiện cần thiết và sinh mã Python. Cuối cùng, code được thực thi để tạo ra kết quả.

Có thể hiểu sự phân công như sau: LLM phụ trách đọc và kết nối thông tin; retrieval cung cấp kiến thức công thức; Python phụ trách số học.

MedRaC vì vậy bổ sung hai điểm tựa ở những nơi LLM thường thiếu ổn định: **nhớ đúng công thức** và **tính đúng kết quả**.

### Slide 12 — Each MedRaC Component Targets a Distinct Failure Mode (0:50)

Mỗi thành phần của MedRaC nhắm vào một failure mode cụ thể.

Formula RAG giảm lỗi chọn sai hoặc bịa công thức. Value extraction tìm các input cần thiết. Code generation chuyển công thức thành thao tác có thể thực thi, còn Python execution giảm lỗi số học.

Tuy nhiên, với Wells score, retrieval chỉ cung cấp tiêu chí và số điểm. Nếu LLM hiểu sai một mô tả trong bệnh án, Python vẫn sẽ cộng chính xác trên dữ liệu đầu vào sai.

Đây là ranh giới của phương pháp: MedRaC phù hợp hơn với equation-based tasks, trong khi rule-based tasks vẫn phụ thuộc nhiều vào clinical interpretation.

## IV. Paper Experiments and Results

### Slide 13 — Experimental Setup for Calculation and Clinical Scoring (0:40)

Paper đánh giá trên MedCalc-Bench, ban đầu gồm 1.048 case từ 55 medical calculators. Sau khi kiểm tra công thức, quy tắc và ground truth, tác giả loại 108 case bị lỗi hoặc lỗi thời, còn lại 940 case.

Các phương pháp gồm Direct, Chain-of-Thought, One-shot, Self-Refine, MedPrompt và MedRaC, được chạy trên nhiều mô hình mở và đóng.

Việc làm sạch benchmark cũng rất quan trọng: nếu ground truth sai, kết luận về mô hình và phương pháp cũng có thể sai.

### Slide 14 — Main Results Across Models and Prompting Methods (0:50)

Thay vì đọc từng ô trong bảng, chúng ta tập trung vào hai xu hướng.

Trên equation-based tasks, MedRaC cải thiện khá nhất quán, đặc biệt với các open model nhỏ. Điều này phù hợp với thiết kế: công thức được retrieval và số học được giao cho Python.

Trên rule-based tasks, kết quả không ổn định. One-shot đôi khi tốt hơn vì cung cấp một ví dụ đầy đủ về cách ánh xạ bệnh án sang tiêu chí. Formula RAG chỉ cung cấp quy tắc, trong khi mô hình vẫn phải tự diễn giải ngữ cảnh.

Kết quả này đồng thời cho thấy điểm mạnh và giới hạn của MedRaC.

### Slide 15 — Main Result: MedRaC Improves Equation-Based Tasks (0:50)

Trên equation-based tasks, so với One-shot, Phi-4-mini tăng 51,92 điểm phần trăm, LLaMA3.1-8B tăng 49,25 điểm và GPT-4o tăng 10,15 điểm.

Mức tăng lớn ở các mô hình nhỏ cho thấy tool augmentation có thể bù đắp một phần hạn chế về kiến thức công thức và số học. Với mô hình mạnh, baseline đã cao hơn nên khoảng cải thiện nhỏ hơn.

Tuy nhiên, kết luận nên được giới hạn ở những task có công thức rõ ràng và input có thể chuyển thành biến. Paper chưa giải quyết hoàn toàn các bài toán cần diễn giải lâm sàng phức tạp.

### Slide 16 — Step-Wise Evaluation Aligns Closely with Expert Judgment (0:50)

Vì framework dùng LLM Judge, paper cần kiểm tra độ tin cậy của evaluator.

Trên 46 sample, agreement giữa LLM và chuyên gia đạt 90,2% ở Formula, 88,1% ở Calculation và 97,8% ở Answer. Extraction thấp hơn, ở mức 78,3%.

Kết quả cho thấy LLM Judge tương đối phù hợp ở những bước có tiêu chuẩn rõ. Tuy nhiên, tập validation còn nhỏ và paper chỉ dùng percent agreement, nên chưa thể xem Judge là hoàn toàn chính xác.

### Slide 17 — Error Reductions Reveal MedRaC's Remaining Boundary (0:55)

Figure này cho biết MedRaC cải thiện nhờ giảm loại lỗi nào.

So với CoT trên LLaMA3.1-8B, formula error giảm 77,5%, arithmetic error giảm 82,6% và demographic adjustment error giảm 70,9%. Đây là các lỗi mà retrieval và code execution có thể tác động trực tiếp.

Ngược lại, incorrect extraction chỉ giảm 21,3% và clinical misinterpretation giảm 26,9%. Nếu mô hình hiểu sai bệnh án, Python chỉ thực hiện chính xác trên đầu vào sai.

Vì vậy, MedRaC xử lý tốt các **tool-addressable errors**, còn clinical interpretation vẫn là bottleneck.

## V. Local Demo and Mini-Experiment

### Slide 18 — Local Live Demo Makes the F--E--C--A Pipeline Observable (0:30)

Sau phần kết quả paper, nhóm sử dụng code được công bố để xây dựng một demo nhỏ.

Thay vì chỉ hiển thị final answer, demo cho phép theo dõi patient note, công thức được retrieval, dữ kiện được trích xuất, code được sinh và kết quả sau execution.

Mục tiêu là giúp F–E–C–A trở thành một quá trình có thể quan sát và kiểm tra.

### Slide 19 — Local Mini-Experiment (0:50)

Nhóm chạy mini-experiment trên cùng 10 sample cố định, gồm 7 equation-based và 3 rule-based case. Quy mô này chỉ nhằm minh họa pipeline, không dùng để khẳng định lại kết quả paper.

Theo final-answer evaluation, Plain CoT đúng 9 trên 10 và MedRaC đúng 10 trên 10. Nhưng khi đánh giá F–E–C–A, strict correctness tương ứng chỉ còn 8 trên 10 và 9 trên 10.

Như vậy, một số đáp án cuối vượt qua cách chấm số học nhưng quá trình vẫn có bước không hợp lệ. Kết quả nhỏ này minh họa đúng thông điệp trung tâm của paper.

## VI. Discussion and Conclusion

### Slide 20 — Discussion (0:55)

Paper vẫn có một số giới hạn. Dataset chủ yếu gồm các bài toán tiếng Anh có cấu trúc, chưa phản ánh đầy đủ bệnh án dài và nhiễu. MedRaC phụ thuộc vào formula bank chính xác; còn LLM Judge cũng có thể đánh giá sai và mới được kiểm chứng trên một tập nhỏ.

Quan trọng hơn, retrieval và Python chỉ hỗ trợ tốt sau khi bệnh án đã được chuyển thành công thức và dữ kiện đúng. Bước diễn giải ngôn ngữ lâm sàng vẫn phụ thuộc nhiều vào LLM.

Vì vậy, hướng tiếp theo cần tập trung vào clinical interpretation, dữ liệu thực tế đa dạng hơn và vai trò kiểm tra của chuyên gia. Đây là hệ thống nghiên cứu, không phải công cụ chẩn đoán tự động.

### Slide 21 — Conclusions (0:50)

Quay lại câu hỏi đầu tiên: một con số đúng có đủ để tin tưởng LLM hay không? Câu trả lời của paper là chưa đủ.

Paper đóng góp framework F–E–C–A để đánh giá từng bước, error attribution để xác định nguyên nhân và MedRaC để giảm lỗi công thức cùng lỗi số học.

Tuy nhiên, công cụ chủ yếu giúp mô hình **nhớ đúng** và **tính đúng**. Việc **hiểu đúng bệnh án**, đặc biệt trong các bài toán rule-based như Wells score, vẫn là thách thức.

Thông điệp cuối cùng là: với AI trong y tế, chúng ta không chỉ cần **correct outputs**, mà còn cần **verifiable processes**.

### Slide 22 — Questions and Discussion

Cảm ơn thầy cô và các bạn đã lắng nghe. Nhóm xin kết thúc phần trình bày và sẵn sàng trao đổi về framework đánh giá, phương pháp MedRaC, kết quả cũng như những giới hạn của paper.
