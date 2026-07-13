# Script thuyết trình seminar MedRaC

Script này được soạn cho 32 slide chính. Các slide 33-39 thuộc `Appendix` và chỉ sử dụng khi cần giải thích thêm trong phần hỏi đáp. Nội dung tiếng Việt được viết theo văn phong nói học thuật; các thuật ngữ chuyên ngành, tên phương pháp, model, dataset và metric được giữ nguyên bằng tiếng Anh.

## Slide 1 - From Scores to Steps

Kính chào quý thầy và các bạn. Nhóm K23 xin trình bày paper **“From Scores to Steps: Diagnosing and Improving LLM Performance in Evidence-Based Medical Calculations”**, được công bố tại `EMNLP 2025 Main Conference` dưới hình thức `Oral Presentation`.

Paper nghiên cứu một vấn đề quan trọng trong ứng dụng `Large Language Model` cho y khoa: một model có thể đưa ra kết quả số gần đúng nhưng vẫn sử dụng sai công thức, trích xuất sai dữ kiện hoặc thực hiện sai các bước trung gian. Vì vậy, paper đề xuất chuyển trọng tâm đánh giá từ `final-answer accuracy` sang kiểm tra toàn bộ quá trình thực hiện `medical calculation`.

Trong bài trình bày, nhóm sẽ tập trung vào `Step-wise Evaluation Framework`, phương pháp `MedRaC`, kết quả thực nghiệm của paper và một `local demo` quy mô nhỏ.

## Slide 2 - Presentation Outline

Bài trình bày gồm sáu phần. Trước tiên, nhóm giới thiệu động cơ nghiên cứu và hạn chế của cách đánh giá chỉ dựa trên đáp án cuối. Phần hai trình bày `Step-wise Evaluation Framework` với bốn stage `Formula`, `Extraction`, `Calculation` và `Answer`.

Tiếp theo, nhóm phân tích methodology của `MedRaC`, sau đó trình bày experimental setup và các kết quả chính. Phần năm minh họa pipeline bằng `local demo` và `mini-experiment`. Cuối cùng, nhóm thảo luận limitations, kết luận và các câu hỏi nghiên cứu mở.

## Slide 3 - Evidence-Based Medical Calculation as a Structured Task

`Evidence-based medical calculation` không chỉ là một phép toán số học. Quy trình bắt đầu từ `Clinical Note` và câu hỏi, sau đó model phải chọn đúng formula hoặc scoring rule, trích xuất đúng patient variables, thực hiện calculation và báo cáo kết quả.

Paper phân biệt hai nhóm calculator. Với `Equation-based calculator`, đầu ra được tính bằng một hàm số rõ ràng, chẳng hạn BMI, corrected sodium hoặc Cockcroft-Gault creatinine clearance. Với `Rule-based calculator`, model phải ánh xạ từng clinical criterion sang điểm số rồi cộng lại, như Wells score hoặc CHA2DS2-VASc.

Điểm quan trọng là mỗi stage đòi hỏi một năng lực khác nhau: medical knowledge, information extraction, clinical interpretation và arithmetic. Sai ở bất kỳ stage nào cũng có thể làm kết quả cuối không còn đáng tin cậy.

## Slide 4 - Case Study: A Correct Score from an Incorrect Formula

Slide này minh họa trực tiếp hạn chế của `final-answer scoring`. Bệnh nhân có measured sodium bằng 127 mEq/L và glucose bằng 527 mg/dL. Bài toán yêu cầu dùng Hillier equation.

Formula đúng là: corrected sodium bằng measured sodium cộng 0.024 nhân với glucose trừ 100. Thay số vào, reference answer là 137.248.

Tuy nhiên, model sử dụng hệ số 0.016 và không trừ baseline 100, nên tính ra 135.432. Giá trị này vẫn nằm trong khoảng sai số cộng trừ 5 phần trăm của benchmark cũ, từ 130.39 đến 144.11, nên được đánh dấu `Correct`.

Với `Step-wise Evaluation`, `Formula` sai, `Extraction` đúng, `Calculation` đúng theo formula sai, và `Answer` sai theo reference chuẩn. Ví dụ này cho thấy tính đúng một công thức sai vẫn phải được xem là một lời giải sai.

## Slide 5 - Research Gap and Questions

Từ case study vừa rồi, paper xác định bốn hạn chế của cách đánh giá trước đây. Thứ nhất, `final-answer scoring` không kiểm tra model đã chọn đúng medical formula hay chưa. Thứ hai, numerical tolerance rộng có thể chấp nhận một computation path không hợp lệ. Thứ ba, aggregate accuracy không cho biết model bắt đầu sai tại stage nào. Cuối cùng, một accuracy duy nhất đang gộp chung knowledge error, extraction error, clinical interpretation error và arithmetic error.

Do đó, paper đặt ra ba research question: đánh giá từng stage như thế nào; stage nào thường là first failure; và liệu `Formula RAG` kết hợp `code execution` có giảm được các lỗi này hay không.

## Slide 6 - Main Contributions

Paper có ba contribution chính.

Thứ nhất, tác giả curate lại `MedCalc-Bench` và xây dựng `Step-wise Evaluation` cho bốn stage `Formula`, `Extraction`, `Calculation` và `Answer`.

Thứ hai, paper sử dụng các `LLM Judge` chuyên biệt cho từng nhiệm vụ, đồng thời đề xuất `Conditional Correctness`, `First Error Attribution` và error taxonomy gồm tám nhóm.

Thứ ba, paper đề xuất `MedRaC`, một training-free pipeline kết hợp `Formula RAG` với `Python code execution`. Mức cải thiện được báo cáo đạt tối đa 53.19 percentage points trong một số thiết lập.

Đáng chú ý, accuracy của GPT-4o giảm từ 62.7 phần trăm khi chấm theo đáp án cuối xuống 43.6 phần trăm khi áp dụng `Step-wise Evaluation`. Khoảng chênh lệch này chính là phần lỗi mà final-answer metric đã che giấu.

## Slide 7 - Medical Calculation as a Multi-Stage Process

Đây là framework đánh giá trung tâm của paper. Từ `Patient Note` và `Question`, lời giải được phân rã thành bốn stage: `Formula`, `Extraction`, `Calculation` và `Answer`, viết tắt là `F-E-C-A`.

Mỗi stage nhận một binary correctness judgment riêng. Tuy nhiên, các stage không độc lập về mặt quy trình. `Extraction` phụ thuộc vào formula đã chọn; `Calculation` phụ thuộc vào formula và extracted values; còn `Answer` phụ thuộc vào toàn bộ các bước trước.

Cách phân rã này không chỉ trả lời model có đúng hay không, mà còn cho biết model sai ở đâu và module nào có thể được sử dụng để khắc phục lỗi đó.

## Slide 8 - Formula Selection and Value Extraction

Stage đầu tiên là `Formula Selection`, ký hiệu `F`. Model phải xác định đúng calculator hoặc scoring rule, sử dụng đúng coefficients, units và boundary conditions. Formula của model được so sánh với canonical formula trong reference library gồm 55 calculator.

Stage thứ hai là `Value Extraction`, ký hiệu `E`. Model phải trích xuất đầy đủ cả numerical variables và categorical variables, đồng thời giữ đúng units, demographic attributes và clinical meaning.

Tiêu chuẩn ở stage `E` khá nghiêm ngặt. Chỉ cần thiếu một required variable, thêm một variable không có trong bệnh án hoặc gán sai clinical label thì toàn bộ stage được xem là incorrect. Vì vậy, `Extraction` không chỉ là nhận diện con số mà còn bao gồm việc gắn con số đó với đúng vai trò lâm sàng.

## Slide 9 - Calculation Correctness Is Not Answer Correctness

Paper phân biệt rõ `Calculation correctness`, ký hiệu `C`, và `Answer correctness`, ký hiệu `A`.

`Calculation` kiểm tra các phép toán có hợp lệ đối với formula và values mà model thực sự sử dụng hay không. Vì vậy, model có thể đạt `V(C)=1` dù formula ban đầu sai. Ngược lại, `Answer` so sánh kết quả cuối với ground truth, có xét valid unit conversion và precision-aware tolerance.

Tolerance phụ thuộc vào số chữ số thập phân của reference. Ví dụ, giá trị 10.7 có epsilon bằng 0.05, còn 10.65 có epsilon bằng 0.005. Nếu output có nhiều hơn hai chữ số thập phân, hệ thống làm tròn đến tối đa hai chữ số để đánh giá.

Như vậy, một phép tính internally consistent chưa đủ để chứng minh lời giải đúng về mặt y khoa.

## Slide 10 - Strict Overall Correctness

Paper định nghĩa `Strict Overall Correctness` bằng phép AND của bốn stage. Một case chỉ được xem là correct khi `Formula`, `Extraction`, `Calculation` và `Answer` đều correct.

Ký hiệu kappa ở đây không phải Cohen's kappa. Đây là biến correctness của một case, được xác định bằng `V(F) AND V(E) AND V(C) AND V(A)`.

Bảng minh họa cho thấy chỉ dòng đầu tiên, khi cả bốn giá trị đều true, mới được đánh dấu correct. Nếu calculation sai, hoặc formula sai dù các stage khác đúng, strict correctness vẫn bằng false.

Metric này loại bỏ những trường hợp model tình cờ đưa ra đáp án chấp nhận được bằng một reasoning path không hợp lệ.

## Slide 11 - Conditional Correctness

`Conditional Correctness`, viết tắt là `CC`, đo xác suất một stage đúng với điều kiện tất cả preceding stages đều đúng.

Ví dụ, `Calculation CC` là xác suất `Calculation` đúng khi cả `Formula` và `Extraction` đã đúng. Metric này khác với unconditional calculation accuracy, vì nó loại ảnh hưởng của lỗi formula và lỗi input ra khỏi việc đánh giá khả năng tính toán.

Do đó, `CC` giúp trả lời một câu hỏi cụ thể hơn: khi model đã đi đến stage hiện tại với đầu vào hợp lệ, model thực hiện stage này đáng tin cậy đến mức nào.

## Slide 12 - First Error Attribution

`First Error Attribution`, viết tắt là `FE`, phân bổ mỗi failed case cho stage incorrect đầu tiên trong chuỗi `F-E-C-A`.

Ví dụ, nếu formula sai thì case được tính vào `Formula FE`, kể cả khi các stage sau cũng sai. Nếu formula đúng nhưng extraction sai, case được tính vào `Extraction FE`.

Khác biệt giữa `CC` và `FE` là: `CC` đo reliability tại một stage sau khi các stage trước đã thành công, còn `FE` xác định nguồn gốc đầu tiên của các failed cases. Vì vậy, `FE` đặc biệt hữu ích để quyết định nên cải thiện retrieval, extraction hay numerical execution.

## Slide 13 - Structured Error Taxonomy

Ngoài binary judgment cho từng stage, paper định nghĩa tám error category có cấu trúc.

Nhóm knowledge và formula gồm `Formula Misselection or Hallucination`. Nhóm extraction và interpretation gồm `Incorrect Variable Extraction`, `Clinical Misinterpretation`, `Missing Variables` và `Demographic Adjustment Failure`. Nhóm numerical execution gồm `Unit Conversion Error`, `Arithmetic Error` và `Rounding or Precision Error`.

Một response có thể chứa nhiều error label cùng lúc. Vì vậy, paper sử dụng `Jaccard similarity` để so sánh hai tập nhãn, bằng kích thước intersection chia cho kích thước union.

Error taxonomy giúp chuyển kết quả đánh giá từ một accuracy tổng quát sang các failure mode có thể phân tích và can thiệp cụ thể.

## Slide 14 - Equation-Based and Rule-Based Tasks

`Equation-based task` sử dụng một mathematical transformation rõ ràng. Các lỗi chính thường liên quan đến coefficient, unit, arithmetic và rounding. Khi formula và values đã đúng, loại bài này phù hợp với deterministic code execution.

`Rule-based task` yêu cầu model đánh giá từng clinical criterion rồi gán điểm hoặc category. Ở đây, model không chỉ xử lý số mà còn phải quyết định một finding có thỏa mãn điều kiện chấm điểm hay không.

Do đó, `Python execution` có thể giải quyết tốt phần arithmetic của equation-based task, nhưng chỉ hỗ trợ một phần đối với rule-based task. Bottleneck chính của rule-based task thường nằm ở clinical interpretation.

## Slide 15 - MedRaC Overview

Sau khi xác định các failure mode, paper đề xuất `MedRaC` để can thiệp trực tiếp vào hai nhóm lỗi lớn: formula knowledge và arithmetic.

Pipeline gồm bốn thành phần chính: `Formula RAG`, `Value Extraction`, `Code Generation` và `Python Execution`. `Formula RAG` cung cấp medical formula từ external knowledge base. LLM tiếp tục trích xuất values, chuyển formula thành Python code và giao phép tính cho execution engine.

MedRaC là `training-free` và `tool-augmented`. Phương pháp không đề xuất một neural architecture mới và không yêu cầu fine-tuning. Phần Deep Learning nằm ở khả năng reasoning của LLM, embedding-based retrieval và tool-augmented inference.

## Slide 16 - Formula Retrieval

Trong `Formula Retrieval`, câu hỏi về calculator được encode thành query embedding. Hệ thống so sánh query embedding với các formula-document embedding trong formula bank và lấy document có similarity cao nhất.

Ký hiệu `d*` trên slide là formula document được chọn bằng phép `argmax` trên similarity score. Retrieved document cung cấp formula, required variables, coefficients và các điều kiện áp dụng cần thiết cho những stage sau.

Mục tiêu của bước này là giảm `Formula Misselection` và `Formula Hallucination` bằng cách externalize medical knowledge thay vì yêu cầu LLM nhớ chính xác toàn bộ formula từ model parameters.

## Slide 17 - Value Extraction and Code Generation

Sau retrieval, `Patient Note`, `Question` và `Retrieved Formula` được kết hợp để tạo `Structured Values`.

Vai trò của từng input khác nhau. `Patient Note` cung cấp clinical evidence; `Question` xác định calculator và output cần trả lời; còn `Retrieved Formula` xác định những variables nào phải được trích xuất và chúng được sử dụng như thế nào.

Tiếp theo, model kết hợp `Retrieved Formula`, `Structured Values` và `Question` để sinh `Python Code`. Việc tách extraction khỏi code generation giúp pipeline thể hiện rõ input nào được dùng và phép tính nào sẽ được thực thi.

## Slide 18 - Python-Assisted Calculation

Trong `Python-Assisted Calculation`, LLM vẫn chịu trách nhiệm hiểu clinical text, xác định required patient values và chuyển grounded formula thành code.

Ví dụ trên slide sử dụng corrected sodium: measured sodium bằng 127, glucose bằng 527, và code thực hiện đúng Hillier equation. Sau khi code được sinh, Python thực hiện calculation một cách deterministic và trả numerical result.

Thành phần này nhằm giảm arithmetic error, sai operation order và một phần rounding error. Tuy nhiên, code execution chỉ bảo đảm phép tính đúng theo code đã sinh; nó không thể tự sửa một formula sai hoặc một extracted value sai.

## Slide 19 - How MedRaC Targets Errors

Bảng này ánh xạ từng component của MedRaC với error type mà component đó trực tiếp xử lý.

`Formula RAG` nhắm vào formula selection và hallucination. `LLM Value Extraction` xác định required patient variables nhưng vẫn phụ thuộc vào clinical interpretation. `Code Generation` chuyển formula thành program, còn `Python Execution` giảm arithmetic và operation-order error.

Điểm cần nhấn mạnh là MedRaC externalize hai phần: formula knowledge và arithmetic. Extraction và clinical interpretation vẫn do LLM thực hiện, nên đây tiếp tục là các residual bottleneck của hệ thống.

## Slide 20 - Dataset and Benchmark Cleanup

Thực nghiệm sử dụng `MedCalc-Bench`, gồm các case từ 55 medical calculator trên MDCalc và bao gồm cả equation-based lẫn rule-based task.

Theo Section 4 và Appendix A của paper, tập ban đầu có 1,048 case. Sau quá trình curation, 108 case bị loại và còn lại 940 case hợp lệ.

Các lỗi được phát hiện không chỉ là lỗi định dạng. Chúng bao gồm gestational-age equation bị gõ sai, APACHE II threshold không đúng, lower và upper limits bị đảo khi đáp án âm, QTc unit mismatch và Caprini scoring không hợp lệ.

Việc làm sạch benchmark là quan trọng vì ground truth sai sẽ làm sai cả đánh giá model và đánh giá phương pháp.

## Slide 21 - Experimental Setup

Paper đánh giá tám model, bao gồm Phi-4-mini, LLaMA, Qwen và GPT-4o family. Sáu method được so sánh là `Direct`, `Zero-shot CoT`, `One-shot`, `Self-Refine`, `MedPrompt` và `MedRaC`.

`Self-Refine` cho phép tối đa năm vòng revision, còn `MedPrompt` truy xuất ba example gần nhất. Tất cả thí nghiệm đều là inference-only, không fine-tuning.

`DeepSeek-chat` được sử dụng làm `Step-wise Judge`, còn `DeepSeek-reasoner` đánh giá error type. Các open-source model chạy trên hai NVIDIA RTX A6000 với temperature 0.6 và top-p 0.95. Các GPT model sử dụng default inference setting.

## Slide 22 - Main Results

Table 1 trình bày accuracy theo model, method và task type. Có hai kết luận chính.

Thứ nhất, MedRaC cải thiện rõ rệt calculation-based accuracy trên các model được đánh giá. Thứ hai, kết quả trên rule-based task không nhất quán; trong một số model mạnh, One-shot tốt hơn MedRaC.

Cần lưu ý một caveat quan trọng khi đọc bảng. `Direct` chỉ được chấm theo final answer, trong khi các reasoning-based method được chấm bằng step-wise automatic evaluation. Vì vậy, không nên so sánh trực tiếp cột `Direct` với CoT hoặc MedRaC như thể chúng sử dụng cùng một metric.

Ở slide tiếp theo, nhóm tách riêng một số model để làm rõ khác biệt giữa hai task type.

## Slide 23 - Equation-Based and Rule-Based Performance

Với calculation-based task, MedRaC tăng mạnh accuracy. Chẳng hạn, Phi-4-mini tăng từ 16.47 lên 68.39 phần trăm, LLaMA3.1-8B tăng từ 20.97 lên 70.22 phần trăm, và GPT-4o tăng từ 54.24 lên 64.39 phần trăm so với One-shot.

Ngược lại, với rule-based task, GPT-4o giảm từ 62.24 xuống 51.03 phần trăm, và Qwen3-14B giảm từ 60.77 xuống 50.44 phần trăm.

Một cách giải thích của paper là One-shot example không chỉ cung cấp scoring rule mà còn minh họa cách ánh xạ clinical findings sang criteria. Formula RAG cung cấp rule chính xác nhưng không thay thế được clinical interpretation cần thiết cho rule-based task.

## Slide 24 - Error Reduction

Figure này so sánh error counts của LLaMA3.1-8B-Instruct giữa `Zero-shot CoT` và MedRaC.

MedRaC giảm formula error từ 757 xuống 170, tương đương 77.5 phần trăm. Arithmetic error giảm từ 426 xuống 74, tương đương 82.6 phần trăm. Demographic adjustment error giảm 70.9 phần trăm.

Đây là ba error type mà retrieval và execution có thể can thiệp trực tiếp. Trong khi đó, incorrect extraction chỉ giảm 21.3 phần trăm và clinical misinterpretation giảm 26.9 phần trăm.

Kết quả này phù hợp với thiết kế của MedRaC: hệ thống rất hiệu quả đối với tool-addressable error, nhưng cải thiện hạn chế đối với lỗi cần hiểu sâu clinical context.

## Slide 25 - Ablations Isolate the Roles of Retrieval and Execution

`Ablation Study` cho thấy hai component có vai trò bổ sung cho nhau.

Khi bỏ `Formula RAG`, accuracy giảm từ 64.68 xuống 25.64 phần trăm, tức giảm 39.04 percentage points. `Formula FE` tăng từ 20.78 lên 71.96 phần trăm, cho thấy formula trở thành first failure phổ biến nhất.

Khi bỏ `Code Execution`, accuracy giảm từ 64.68 xuống 53.09 phần trăm, tức giảm 11.59 percentage points. `Calculation FE` tăng từ 3.23 lên 31.88 phần trăm.

Vì vậy, RAG có tác động lớn hơn lên overall accuracy, còn code execution tác động trực tiếp hơn lên arithmetic reliability. Hai thành phần giải quyết hai failure mode khác nhau chứ không thay thế nhau.

## Slide 26 - LLM-as-Judge Validation

Để kiểm tra độ tin cậy của `LLM-as-Judge`, paper sử dụng 46 validated samples từ năm calculator và so sánh judgment giữa Expert, Non-Expert và LLM.

`LLM-Expert agreement` lần lượt là 90.2 phần trăm cho Formula, 78.3 cho Extraction, 88.1 cho Calculation và 97.8 cho Answer. Extraction là stage có agreement thấp nhất và chỉ cao hơn `Expert-Non-Expert agreement` 0.2 percentage points.

Kết quả cho thấy LLM Judge có thể hỗ trợ scalable evaluation, nhưng chưa thể được xem là ground truth tuyệt đối. Ngoài ra, paper báo cáo `percent agreement`, không sử dụng chance-corrected reliability metric như Cohen's kappa hoặc Krippendorff's alpha.

## Slide 27 - Local Demo Objective and Pipeline

Phần tiếp theo trình bày `local demo` của nhóm. Mục tiêu không phải tái tạo toàn bộ kết quả paper mà là minh họa một pipeline có thể quan sát được.

Input gồm `Calculator ID`, `Patient Note` và `Question`. Pipeline thực hiện formula retrieval, value extraction, code generation, safe execution và trả final answer. Nếu bật LLM evaluation, hệ thống có thể đánh giá theo `F-E-C-A`. Các structured output được hiển thị trên Streamlit để kiểm tra từng stage.

Demo sử dụng fixed benchmark rows để bảo đảm reproducibility. Code được chạy qua seminar-specific safety layer, nhưng nhóm không xem đây là một secure sandbox cho production.

## Slide 28 - Streamlit Demo and Representative Example

Đây là giao diện Streamlit ở chế độ replay cho calculator `Creatinine Clearance`, sử dụng Cockcroft-Gault Equation với Calculator ID 2.

Ảnh bên trái hiển thị các input có thể kiểm tra trực tiếp, gồm Calculator ID, Patient Note và Question, cùng các tab tương ứng với từng pipeline stage.

Ảnh bên phải minh họa retrieval result. Query chỉ sử dụng Question, sau đó hệ thống hiển thị retrieval status, similarity score, rank và Cockcroft-Gault formula được truy xuất.

Điểm quan trọng của demo là người xem có thể theo dõi intermediate artifacts thay vì chỉ nhìn final answer. Điều này phản ánh đúng tinh thần process-level evaluation của paper.

## Slide 29 - Local Mini-Experiment Setup and Results

`Mini-experiment` so sánh baseline `Plain CoT` với `MedRaC + RAG` trên cùng 10 fixed sample, gồm 7 equation-based case và 3 rule-based case.

Hai phương pháp sử dụng cùng generator là `openai/gpt-4.1`. MedRaC sử dụng `text-embedding-3-small`, temperature bằng 0 và seed bằng 42.

Kết quả final-answer accuracy là 9 trên 10 cho Plain và 10 trên 10 cho MedRaC. Không có failed hoặc skipped sample.

Tuy nhiên, `Step-wise Evaluation` đã bị tắt trong run này. Vì vậy, nhóm không báo cáo Formula, Extraction, Calculation, Answer accuracy, strict correctness, CC hoặc FE. Kết quả 10 sample chỉ mang tính descriptive, không phải paper reproduction và không hỗ trợ statistical inference.

## Slide 30 - Scope, Limitations, and Evaluation Caveats

Nhóm tách limitations của paper và limitations của local experiment.

Về paper, benchmark chủ yếu gồm structured, single-turn task với curated English note, chưa đại diện cho noisy EHR, multilingual data hoặc multi-turn clinical workflow. Human validation của LLM Judge chỉ có 46 case và dùng percent agreement. Các model comparison chủ yếu báo cáo point estimate, không có confidence interval hoặc significance test. MedRaC cũng phụ thuộc vào một formula bank chính xác và cập nhật.

Về local experiment, quy mô chỉ có 10 fixed case, sử dụng model substitution qua GitHub Models, một retrieval index riêng và không có LLM step-wise evaluation.

Cả paper lẫn local demo đều phục vụ research evaluation, không phải clinical deployment.

## Slide 31 - Conclusions

Nhóm rút ra ba kết luận chính.

Thứ nhất, `final-answer accuracy` không đủ để đánh giá độ tin cậy của medical calculation.

Thứ hai, decomposition `F-E-C-A`, cùng `Conditional Correctness` và `First Error Attribution`, giúp xác định conditional reliability và earliest failed stage.

Thứ ba, MedRaC giảm hiệu quả formula error và arithmetic error bằng retrieval và deterministic execution. Tuy nhiên, extraction và clinical interpretation vẫn là unresolved bottleneck.

Thông điệp tổng quát của paper là end-task scoring cần được bổ sung bằng `domain-grounded process evaluation`, đặc biệt trong những lĩnh vực có yêu cầu cao về tính chính xác và khả năng kiểm chứng như y khoa.

## Slide 32 - Questions and Discussion

Nhóm xin kết thúc phần trình bày bằng ba câu hỏi thảo luận.

Thứ nhất, decomposition `F-E-C-A` có đủ cho các clinical reasoning workflow phức tạp hơn hay không?

Thứ hai, một `LLM-as-Judge` cần được calibration và human validation như thế nào trước khi sử dụng cho high-stakes evaluation?

Thứ ba, làm thế nào để formula bank luôn được versioning, cập nhật theo guideline và có thể clinical audit?

Nhóm K23 xin cảm ơn quý thầy và các bạn đã lắng nghe, và xin tiếp nhận câu hỏi.

# Appendix - Script sử dụng khi Q&A

## Slide 33 - Appendix: Complete Main-Results Table

Slide này cung cấp toàn bộ Table 1 của paper. Khi trả lời câu hỏi, cần đọc theo ba chiều: model, prompting method và task type. Cột `Rule` tương ứng rule-based task, còn `Calc` tương ứng calculation-based task.

Không nên dùng cột `Direct` để kết luận method nào tốt hơn vì `Direct` chỉ được chấm final answer, trong khi các reasoning-based method dùng step-wise automatic evaluation. Bảng phù hợp để tra cứu một model cụ thể, không nên đọc tuần tự toàn bộ trong phần trình bày chính.

## Slide 34 - Appendix: Complete Metric Definitions

Slide này tổng hợp ba metric chính. `Strict Correctness` yêu cầu cả F, E, C và A đều đúng. `Conditional Correctness` đo xác suất stage hiện tại đúng khi tất cả preceding stage đúng. `First Error Attribution Rate` đo tỷ lệ failed case có stage hiện tại là first error.

Khi giải thích, cần phân biệt `CC` là reliability có điều kiện, còn `FE` là phân bổ nguyên nhân đầu tiên của failed case.

## Slide 35 - Appendix: Paper Discrepancies

Paper có hai numerical inconsistency cần được công khai.

Thứ nhất, Section 2 ghi 1,047 vignette, trong khi Section 4 và Appendix A ghi 1,048. Deck sử dụng 1,048 vì đây là con số nhất quán với 108 case bị loại và 940 case được giữ lại.

Thứ hai, phần prose của Section 5 ghi `Formula CC` khi bỏ RAG là 7.34 phần trăm, nhưng Table 5 ghi 46.49 phần trăm. Deck sử dụng giá trị in trong Table 5 và không tự suy diễn một con số thay thế.

## Slide 36 - Appendix: Formula-Bank Scaling

Paper mở rộng formula bank từ 55 lên 785 document, tương đương khoảng 14 lần. Với `ada-002`, Top-1 và Top-2 retrieval accuracy đều đạt 100 phần trăm. Hai model `text-embedding-3-large` và `text-embedding-3-small` có Top-1 lần lượt là 96.36 và 98.18 phần trăm; Top-2 đều đạt 100 phần trăm.

Kết quả cho thấy retrieval ổn định trong thí nghiệm này, nhưng chưa đủ để khẳng định khả năng mở rộng đến toàn bộ medical knowledge hoặc các guideline có phiên bản và ngữ cảnh phức tạp.

## Slide 37 - Appendix: Local Experiment

Slide này cung cấp execution count và metric availability của local experiment. Plain sử dụng 10 chat attempt. MedRaC sử dụng 20 chat attempt và 10 embedding attempt. Không có failed hoặc skipped sample.

Chỉ final-answer accuracy khả dụng. Formula, Extraction, Calculation, Answer step accuracy, strict correctness, CC và FE đều không được đánh giá vì LLM step-wise evaluation đã bị tắt. Do đó, không được suy ra các metric này từ final answer.

## Slide 38 - Appendix: Safe-Execution Boundary

Seminar demo sử dụng AST allowlist, cấm import, filesystem, network, shell và user input. Code chạy trong spawned child process với strict timeout, đồng thời kiểm tra result type và magnitude.

Các safeguard này giảm rủi ro khi thực thi generated arithmetic code nhưng không tạo thành secure sandbox. Hệ thống không có kernel isolation, syscall filtering hoặc multi-tenant security guarantee. Phạm vi sử dụng chỉ giới hạn trong seminar demonstration.

## Slide 39 - References

Nguồn chính của bài trình bày là paper của Wang và cộng sự, bản arXiv v2 ngày 31 tháng 1 năm 2026. Các số liệu paper-reported, formula, metric và limitation trong phần trình bày đều được đối chiếu với nguồn này.
