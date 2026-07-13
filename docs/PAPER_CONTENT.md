# From Scores to Steps: Diagnosing and Improving LLM Performance in Evidence-Based Medical Calculations

## 1. Thông tin chung và tư tưởng trung tâm

Paper được công bố trên arXiv, bản hiện tại là v2 ngày 31/01/2026, và được chấp nhận trình bày Oral tại hội nghị chính EMNLP 2025.

### Thông điệp cốt lõi

Paper cho rằng:
Đánh giá LLM trong bài toán tính toán y khoa chỉ dựa trên đáp án cuối cùng là không đủ và có thể tạo ra cảm giác an toàn giả tạo.

Một LLM có thể đưa ra con số gần đúng, nhưng quá trình suy luận lại chứa những lỗi nghiêm trọng như:
- Chọn sai công thức y khoa.
- Trích xuất sai dữ kiện bệnh nhân.
- Bỏ qua hệ số hiệu chỉnh theo giới tính, tuổi hoặc tình trạng thai kỳ.
- Dùng sai đơn vị.
- Tính toán sai nhưng tình cờ ra kết quả nằm trong khoảng sai số cho phép.

Paper đề xuất chuyển từ **“Scores”** — chỉ chấm điểm đáp án cuối sang **“Steps”** — kiểm tra từng bước suy luận. Đồng thời, tác giả xây dựng một hệ thống agent tên là **MedRaC**, kết hợp truy xuất công thức y khoa bằng RAG và thực thi Python để giảm lỗi.

## 2. Abstract — Tóm tắt paper

### Vấn đề nghiên cứu

LLM đã đạt kết quả tốt trên nhiều benchmark y khoa, nhưng năng lực thực hiện **medical calculations** (tính toán y khoa) vẫn chưa được đánh giá đúng mức.

Các benchmark trước thường:
- Chỉ xem đáp án cuối cùng.
- Cho phép khoảng sai số khá rộng, ví dụ ±5%.
- Không kiểm tra xem công thức, biến đầu vào và các phép tính trung gian có đúng hay không.

Điều này đặc biệt nguy hiểm vì một kết quả số có thể ảnh hưởng đến quyết định điều trị, đánh giá nguy cơ hoặc liều thuốc.

### Ba đóng góp chính

Paper đưa ra ba đóng góp:

1. **Làm sạch MedCalc-Bench và xây dựng đánh giá từng bước**
   LLM được đánh giá độc lập ở các bước:
   `Formula → Value Extraction → Calculation → Final Answer`
   Khi dùng cách đánh giá nghiêm ngặt này, độ chính xác của GPT-4o được báo cáo giảm từ 62,7% xuống 43,6%, cho thấy nhiều lỗi trước đây bị che giấu bởi cách chấm đáp án cuối.

2. **Xây dựng hệ thống tự động phân loại lỗi**
   Một LLM Judge kiểm tra từng bước và gán lỗi vào các nhóm có cấu trúc như:
   - Sai công thức.
   - Sai biến đầu vào.
   - Sai đơn vị.
   - Sai số học.
   - Sai diễn giải lâm sàng.

3. **Đề xuất MedRaC**
   MedRaC kết hợp:
   `Formula RAG + Python Code Execution`
   Hệ thống không cần fine-tuning nhưng giúp tăng độ chính xác của nhiều LLM, với mức cải thiện tuyệt đối được báo cáo lên tới 53,19 điểm phần trăm trong một số thiết lập.

## 3. Section 1 — Introduction

### 3.1. Medical calculation là gì?

Medical calculation là việc sử dụng dữ kiện của bệnh nhân để tính một đại lượng hoặc điểm số có ý nghĩa lâm sàng, chẳng hạn:
- BMI.
- Độ thanh thải creatinine.
- eGFR.
- Nguy cơ tim mạch.
- Wells score.
- APACHE II.
- CHA₂DS₂-VASc.
- Corrected sodium.

Bài toán này không chỉ là phép tính số học. Nó thường gồm bốn khả năng khác nhau:
1. **Medical knowledge**: Biết đúng công thức
2. **Information extraction**: Đọc đúng bệnh án
3. **Clinical reasoning**: Hiểu đúng ý nghĩa lâm sàng
4. **Arithmetic**: Tính đúng

Đây là lý do một mô hình có khả năng làm toán tốt vẫn có thể thất bại trong medical calculation.

### 3.2. Vấn đề của MedCalc-Bench cũ

MedCalc-Bench ban đầu chứa các vignette lâm sàng được xây dựng từ 55 calculator phổ biến trên MDCalc. Tuy nhiên, benchmark chỉ kiểm tra xem đáp án cuối có nằm trong khoảng ±5% so với đáp án chuẩn hay không.

Giả sử đáp án chuẩn là: `137.25`
Với sai số ±5%, một câu trả lời khoảng 130–144 vẫn có thể được xem là đúng. Trong thực tế, mô hình có thể:
- Dùng sai hệ số.
- Bỏ mất một hạng tử.
- Dùng sai công thức.
- Nhưng vẫn tình cờ tạo ra kết quả 135.

Benchmark cũ sẽ chấp nhận, trong khi về mặt lâm sàng quá trình đó không đáng tin cậy.

### 3.3. Ví dụ trong Appendix G

Paper đưa ra bài toán hiệu chỉnh sodium khi tăng đường huyết.

Công thức Hillier đúng là:
$Na_{corrected} = Na_{measured} + 0.024(glucose - 100)$

Với:
$Na_{measured} = 127, glucose = 527$

Ta có:
$Na_{corrected} = 127 + 0.024(527 - 100) = 137.248$

LLM lại sử dụng:
$127 + 0.016 \times 527 = 135.432$

Con số 135.432 vẫn nằm trong khoảng ±5% nên benchmark cũ đánh dấu đúng. Tuy nhiên, hệ thống mới phát hiện rằng:
- **Formula**: sai.
- **Extraction**: đúng.
- **Arithmetic**: đúng theo công thức sai.
- **Final answer**: sai theo tiêu chuẩn nghiêm ngặt.

Ví dụ này thể hiện rất rõ luận điểm của paper: **tính đúng một công thức sai vẫn là sai**.

## 4. Section 2 — Background and Related Work

Phần này chia nghiên cứu liên quan thành ba hướng.

### 4.1. Benchmark y khoa dựa trên đáp án cuối

Các benchmark như MedQA, PubMedQA và MedMCQA chủ yếu đánh giá:
- Khả năng nhớ kiến thức.
- Trả lời câu hỏi trắc nghiệm.
- Chọn chẩn đoán hoặc phương án đúng.

Chúng không đánh giá đầy đủ quá trình suy luận định lượng.

MedCalc-Bench tiến thêm một bước khi đưa vào các bài toán tính toán thực tế. Tuy nhiên, việc chỉ đánh giá con số cuối vẫn chưa phản ánh được độ tin cậy lâm sàng.

### 4.2. LLM-as-a-Judge

LLM-as-a-Judge là việc dùng một LLM mạnh để đánh giá câu trả lời của LLM khác.

Trong paper:
- **LLM Test Taker**: mô hình làm bài.
- **LLM Judge**: mô hình đánh giá.
- **Ground-truth explanation**: lời giải và thông tin chuẩn.

LLM Judge không chỉ hỏi “đáp án có đúng không?” mà được giao từng nhiệm vụ hẹp:
- Công thức có đúng không?
- Các biến có được trích xuất đầy đủ không?
- Phép tính có hợp lệ không?
- Có lỗi đơn vị hay không?

Việc tách nhỏ nhiệm vụ giúp quá trình đánh giá có cấu trúc và dễ giải thích hơn.

### 4.3. RAG và tool use

Paper kết hợp hai hướng đã phổ biến trong nghiên cứu LLM:

**Retrieval-Augmented Generation (RAG)**
RAG truy xuất kiến thức đáng tin cậy từ một cơ sở dữ liệu bên ngoài rồi cung cấp cho LLM.
Trong bài toán này, RAG truy xuất:
- Công thức y khoa.
- Điều kiện áp dụng.
- Các biến cần thiết.
- Hệ số hiệu chỉnh.
- Quy tắc chấm điểm.

**Program-aided reasoning**
Thay vì để LLM tự tính số học bằng token generation, LLM sinh mã Python và giao phép tính cho trình thực thi.
Ý tưởng là:
- LLM phụ trách hiểu ngôn ngữ.
- Python phụ trách tính toán xác định.

MedRaC hợp nhất hai hướng này thành một pipeline dành riêng cho medical calculation.

## 5. Section 3 — Methods

### 5.1. Step-wise Evaluation

Paper phân rã quá trình tính toán thành bốn bước.

**Bước 1: Formula Selection — Chọn công thức**
Mô hình phải:
- Chọn đúng calculator.
- Trình bày đúng công thức.
- Dùng đúng hệ số.
- Dùng đúng đơn vị.
- Xét đúng điều kiện biên.
- Không thêm hoặc bỏ các thành phần.

Công thức do mô hình tạo ra được so sánh với một canonical formula trong formula library chứa 55 calculator. Kết quả là nhãn nhị phân:
$V(F) \in \{0, 1\}$
Trong đó $F$ là Formula.

**Bước 2: Value Extraction — Trích xuất dữ kiện**
Hệ thống trích xuất các biến số và biến phân loại từ:
- Clinical vignette.
- Câu trả lời của LLM.

Sau đó so sánh với annotation chuẩn. Tiêu chuẩn ở bước này rất nghiêm ngặt: chỉ cần thiếu một biến, bịa thêm một biến hoặc gán sai nhãn thì cả bước bị xem là sai.
$V(E) \in \{0, 1\}$

Ví dụ, bệnh án có:
- Tuổi: 87.
- Cân nặng: 62 kg.
- Creatinine: 1.4 mg/dL.
- Giới tính: nữ.

Nếu mô hình đọc thành 72 kg hoặc bỏ qua giới tính thì $V(E) = 0$.

**Bước 3: Mathematical Calculation — Tính toán**
Judge kiểm tra từng phép toán dựa trên:
- Công thức mà mô hình sử dụng.
- Các giá trị mô hình đã trích xuất.

Khác với sai số ±5% của benchmark cũ, paper sử dụng mức tolerance phụ thuộc vào số chữ số thập phân.
Ví dụ:
- $10.7 \Rightarrow \epsilon = 0.05$
- $10.65 \Rightarrow \epsilon = 0.005$

Nếu mô hình đưa ra nhiều hơn hai chữ số, kết quả được làm tròn để đánh giá tối đa hai chữ số thập phân.

**Bước 4: Final Answer**
Đáp án cuối được so sánh với ground truth và cho phép các phép chuyển đổi đơn vị hợp lệ.
$V(A) \in \{0, 1\}$

Ở đây cần phân biệt:
- **Calculation correctness**: các phép toán được thực hiện đúng hay không.
- **Final-answer correctness**: con số cuối có khớp ground truth hay không.

Một mô hình có thể tính đúng theo một công thức sai. Khi đó:
$V(C) = 1, V(F) = 0, V(A) = 0$

### 5.2. Các công thức đánh giá quan trọng

**Công thức phụ thuộc giữa các bước**
Paper ký hiệu kết quả tại bước $i$ là $S_i$:
$S_i = f(S_{i-1}, \dots, S_1)$

Điều này thể hiện rằng các bước sau phụ thuộc vào những bước trước. Paper viết:
$\lozenge V(S_i) \iff V(S_{i-1})$

Ý định của tác giả là: bước $S_i$ chỉ có khả năng đúng khi bước trước đó đúng.

> **Ghi chú phản biện**
> Ký hiệu này hơi khó hiểu. Nếu bỏ toán tử khả năng $\lozenge$ và đọc như logic cổ điển, phép tương đương hai chiều có thể bị hiểu rằng:
> *Bước trước đúng thì bước sau bắt buộc đúng.*
> Điều đó không hợp lý. Biểu diễn rõ hơn có thể là:
> $V(S_i) \Rightarrow V(S_{i-1})$
> Tức là: nếu bước sau đúng một cách hợp lệ thì điều kiện tiên quyết là bước trước phải đúng.

**Overall correctness**
Một case chỉ được xem là đúng khi cả bốn bước đều đúng:
$\kappa = V(F) \land V(E) \land V(C) \land V(A)$
Trong đó:
- $F$: formula.
- $E$: extraction.
- $C$: calculation.
- $A$: answer.
- $\kappa$: độ đúng tổng thể của một case.

Đây là phép AND nghiêm ngặt. Chỉ cần một bước sai thì toàn bộ case sai.

**Conditional Correctness**
$CC_i = P(V(S_i) \mid V(S_1) \land \dots \land V(S_{i-1}))$

Conditional Correctness (CC) trả lời câu hỏi:
*Khi tất cả bước trước đã đúng, xác suất bước hiện tại đúng là bao nhiêu?*

Ví dụ:
$CC_{calculation} = P(\text{calculation đúng} \mid \text{formula và extraction đúng})$
Metric này giúp tách lỗi tính toán khỏi lỗi công thức và lỗi dữ liệu đầu vào.

**First Error Attribution Rate**
$FE_i = P(V(S_1) \land \dots \land V(S_{i-1}) \land \neg V(S_i) \mid \neg \kappa)$

First Error Attribution Rate (FE) đo tỷ lệ các case sai mà bước $i$ là bước sai đầu tiên.
Ví dụ, FE cao ở Formula nghĩa là phần lớn lời giải đã thất bại ngay từ việc chọn công thức.

CC và FE có ý nghĩa khác nhau:
| Metric | Câu hỏi được trả lời |
| --- | --- |
| CC | Khi đến được bước này với đầu vào đúng, mô hình làm đúng bao nhiêu lần? |
| FE | Trong những case sai, bước nào là nguồn lỗi đầu tiên? |

### 5.3. Structured Error Attribution

Paper định nghĩa tám nhóm lỗi chính.

| Nhóm lỗi | Ý nghĩa |
| --- | --- |
| Formula Misselection/Hallucination | Chọn sai công thức hoặc bịa, bỏ, đặt sai thành phần công thức |
| Incorrect Variable Extraction | Lấy sai giá trị, thời điểm hoặc đơn vị từ bệnh án |
| Clinical Misinterpretation | Đọc đúng thông tin nhưng hiểu sai ý nghĩa lâm sàng |
| Missing Variables | Bỏ qua biến bắt buộc |
| Demographic Adjustment Failure | Bỏ hoặc dùng sai hệ số tuổi, giới, thai kỳ, BSA… |
| Unit Conversion Error | Không đổi đơn vị hoặc đổi sai |
| Arithmetic Error | Sai phép tính, thứ tự phép toán hoặc cộng điểm |
| Rounding/Precision Error | Làm tròn không phù hợp hoặc vượt tolerance |

**Equation-based và rule-based**
Paper phân biệt hai loại calculator:

1. **Equation-based calculation**: Sử dụng công thức số học trực tiếp:
   $y = f(x_1, x_2, \dots, x_n)$
   Ví dụ BMI, creatinine clearance hoặc corrected sodium.

2. **Rule-based calculation**: Gán điểm dựa trên từng tiêu chí rồi cộng lại:
   $\text{Score} = \sum_{j=1}^m p_j$
   Ví dụ Wells score hoặc APACHE II.
   Rule-based thường đòi hỏi nhiều clinical interpretation hơn. Mô hình không chỉ tìm con số mà phải xác định một triệu chứng hoặc tình trạng có thỏa mãn tiêu chí chấm điểm hay không.

### 5.4. MedRaC

MedRaC là pipeline agentic, training-free, bao gồm hai thành phần cốt lõi.

**Thành phần 1: Formula RAG**
Các công thức và mô tả từ MDCalc được:
- Chuyển thành embedding.
- Lưu trong vector index.
- Truy xuất dựa trên câu hỏi.
- Đưa vào prompt cho LLM.

Luồng xử lý có thể hiểu là:
$q \xrightarrow{\text{embedding}} e_q$
$\text{Retrieve top-k} = \arg\max_{d \in D} \text{sim}(e_q, e_d)$
Trong đó $D$ là formula bank.

Thành phần này nhắm trực tiếp vào:
- Formula hallucination.
- Chọn sai calculator.
- Bỏ hệ số hiệu chỉnh.
- Quên điều kiện sử dụng công thức.

**Thành phần 2: Python Code Execution**
Sau khi có công thức và biến đầu vào, LLM được yêu cầu sinh mã Python biểu diễn phép tính.
Ví dụ:
```python
measured_na = 127
glucose = 527

corrected_na = measured_na + 0.024 * (glucose - 100)
print(corrected_na)
```
Code được thực thi để lấy kết quả cuối.
Thành phần này nhắm vào:
- Arithmetic error.
- Sai thứ tự phép toán.
- Sai phép lũy thừa.
- Một phần lỗi làm tròn.

MedRaC không yêu cầu huấn luyện lại mô hình, nên có thể được gắn lên các LLM có sẵn thông qua API.

## 6. Section 4 — Experiments

### 6.1. Dataset

Paper cho biết dữ liệu thực nghiệm ban đầu có 1.048 case. Sau khi kiểm tra, các tác giả loại 108 case lỗi hoặc lỗi thời, giữ lại:
$1048 - 108 = 940$ case hợp lệ.

Các lỗi dữ liệu được phát hiện gồm:
- Công thức estimated due date bị gõ sai.
- Quy tắc APACHE II sai ngưỡng.
- Ground truth sai.
- Lower và upper limit bị đảo khi đáp án âm.
- Không thống nhất đơn vị giây và mili giây ở QTc.
- Caprini score cộng điểm cho mọi bệnh nhân nữ thay vì chỉ các tình trạng phù hợp.

Việc làm sạch benchmark là một đóng góp đáng chú ý vì benchmark lỗi sẽ làm sai cả đánh giá mô hình lẫn đánh giá phương pháp.

### 6.2. Các baseline

Paper so sánh sáu thiết lập:
- **Direct**: Mô hình chỉ trả về đáp án cuối.
- **Zero-shot CoT**: Mô hình trình bày chuỗi suy luận mà không có ví dụ.
- **One-shot**: Prompt có một lời giải mẫu thuộc cùng calculator.
- **Self-Refine**: Mô hình tự phê bình và sửa câu trả lời, tối đa năm vòng.
- **MedPrompt**: Truy xuất $k=3$ ví dụ gần nhất để đưa vào prompt.
- **MedRaC**: Truy xuất công thức và thực thi code.

Các mô hình gồm Phi-4-mini, LLaMA 3.2-3B, Qwen3-4B/8B/14B, LLaMA 3.1-8B, GPT-4o-mini và GPT-4o.

### 6.3. Môi trường chạy

Các thí nghiệm đều là inference-only, không fine-tuning. Mô hình mã nguồn mở được chạy trên hai GPU NVIDIA RTX A6000, với:
$T=0.6, \text{top-p}=0.95$

DeepSeek-chat được dùng cho đánh giá từng bước và DeepSeek-reasoner được dùng để kiểm tra nhóm lỗi.

## 7. Section 4.1 — Kết quả chính

### 7.1. Equation-based tasks

MedRaC cải thiện mạnh các bài toán dựa trên công thức.
Một số ví dụ khi so với One-shot:

| Model | One-shot Calc | MedRaC Calc |
| --- | --- | --- |
| Phi-4-mini | 16,47% | 68,39% |
| LLaMA3.1-8B | 20,97% | 70,22% |
| Qwen3-8B | 62,90% | 74,54% |
| GPT-4o | 54,24% | 64,39% |

Điều này cho thấy việc cung cấp đúng công thức và giao số học cho Python đặc biệt hữu ích với equation-based calculation.

### 7.2. Rule-based tasks

Kết quả trên rule-based task không hoàn toàn nhất quán.
Ví dụ:
- GPT-4o One-shot: 62,24%.
- GPT-4o MedRaC: 51,03%.
- Qwen3-14B One-shot: 60,77%.
- Qwen3-14B MedRaC: 50,44%.

Tác giả giải thích rằng các mô hình mạnh đã có kiến thức y khoa nội tại tốt. Ngoài ra, One-shot cung cấp một ví dụ hoàn chỉnh về cách ánh xạ bệnh án sang tiêu chí chấm điểm, trong khi Formula RAG chủ yếu cung cấp quy tắc. Với rule-based task, biết quy tắc chưa đủ; mô hình còn phải hiểu ngữ cảnh lâm sàng.

> **Lưu ý về cách đọc Table 1**
> Cột Direct chỉ được chấm theo đáp án cuối, trong khi các phương pháp có reasoning được chấm theo tiêu chuẩn step-wise nghiêm ngặt.
> Do đó, không nên so sánh Direct với CoT hoặc MedRaC như thể chúng sử dụng cùng một metric. Đây là một điểm thiết kế thực nghiệm cần nêu khi phản biện paper.

## 8. Section 4.2 — Kiểm chứng LLM Judge

Paper lấy 46 câu hỏi thuộc năm calculator và yêu cầu:
- Hai chuyên gia y khoa.
- Hai người học trong lĩnh vực liên quan đến y khoa.
- LLM Judge.
đánh giá từng bước.

**Percent agreement**
Công thức:
$\text{Agreement}(a, b) = \frac{1}{n} \sum_{i=1}^n I[a_i = b_i]$
Trong đó:
- $a_i, b_i$: nhãn nhị phân của hai evaluator.
- $I[\cdot]$: indicator function.

Kết quả LLM–Expert:
| Bước | Agreement |
| --- | --- |
| Formula | 90,2% |
| Extraction | 78,3% |
| Calculation | 88,1% |
| Answer | 97,8% |

LLM Judge có agreement với chuyên gia cao hơn Expert–Non-Expert ở Formula, Calculation và Answer, nhưng không cao hơn ở Extraction.

> **Nhận xét phản biện**
> Percent agreement đơn giản, dễ hiểu, nhưng không hiệu chỉnh agreement xảy ra do ngẫu nhiên. Các metric như Cohen’s $\kappa$, Fleiss’ $\kappa$ hoặc Krippendorff’s $\alpha$ có thể cung cấp đánh giá chặt chẽ hơn.
> Ngoài ra, 46 case là quy mô khá nhỏ so với 940 case của benchmark.

## 9. Section 4.3 — Error Type Experiments

Vì một câu trả lời có thể chứa nhiều loại lỗi, paper sử dụng Jaccard similarity để so sánh tập lỗi do hai evaluator gán:
$J(A, B) = \frac{|A \cup B|}{|A \cap B|}$
Trong đó $A$ và $B$ là hai tập nhãn lỗi.

**Kết quả error reduction**
Với LLaMA3.1-8B, khi chuyển từ CoT sang MedRaC:
| Loại lỗi | CoT | MedRaC | Thay đổi |
| --- | --- | --- | --- |
| Formula | 757 | 170 | giảm 77,5% |
| Arithmetic | 426 | 74 | giảm 82,6% |
| Demographic adjustment | 148 | 43 | giảm 70,9% |
| Incorrect extraction | 301 | 237 | giảm 21,3% |
| Clinical misinterpretation | 305 | 223 | giảm 26,9% |

**Ý nghĩa**
MedRaC xử lý rất tốt các lỗi có thể được giải quyết bằng công cụ:
- Formula error $\xrightarrow{\text{RAG}}$ $\downarrow$
- Arithmetic error $\xrightarrow{\text{Python}}$ $\downarrow$

Nhưng cải thiện ít hơn ở:
- Incorrect variable extraction.
- Clinical misinterpretation.
Hai nhóm này đòi hỏi hiểu bệnh án, kiến thức y khoa và suy luận ngữ cảnh. Đây là bottleneck còn lại của hệ thống.

## 10. Section 5 — Ablation Studies

Ablation study trả lời câu hỏi: *Từng thành phần của MedRaC đóng góp bao nhiêu?*

### 10.1. Loại bỏ Formula RAG

| Metric | MedRaC | Không RAG |
| --- | --- | --- |
| Accuracy | 64,68% | 25,64% |
| Formula FE | 20,78% | 71,96% |
| Formula CC | 92,66% | 46,49% |

Khi bỏ retrieval, accuracy giảm gần 39 điểm phần trăm. Formula trở thành nguồn lỗi đầu tiên trong phần lớn case.

> **Một lỗi trình bày trong paper**
> Phần văn bản nói Formula CC giảm xuống 7,34%, nhưng Table 5 ghi 46,49%. Bảng và phần mô tả không nhất quán. Khi trình bày seminar, nên sử dụng giá trị trong bảng và ghi chú đây có thể là lỗi đánh máy.

### 10.2. Loại bỏ Code Execution

| Metric | MedRaC | Không code |
| --- | --- | --- |
| Accuracy | 64,68% | 53,09% |
| Calculation FE | 3,23% | 31,88% |
| Calculation CC | 97,82% | 76,52% |

Code execution làm giảm mạnh xác suất calculation là lỗi đầu tiên.
Có thể kết luận:
- **RAG** là thành phần có tác động lớn hơn đến **overall accuracy**.
- **Code execution** tác động trực tiếp hơn đến **arithmetic reliability**.

### 10.3. Memory Scaling

Formula bank được tăng từ 55 $\rightarrow$ 785 công thức.

Kết quả retrieval:
| Embedding | Top-1 với 785 formula | Top-2 |
| --- | --- | --- |
| ada-002 | 100% | 100% |
| text-embedding-3-large | 96,36% | 100% |
| text-embedding-3-small | 98,18% | 100% |

Các tác giả lập luận rằng công thức y khoa có cấu trúc và ngữ nghĩa tương đối phân biệt, nên retrieval vẫn ổn định khi knowledge base mở rộng khoảng 14 lần.
Tuy nhiên, kết quả này mới chứng minh khả năng mở rộng từ 55 lên 785 formula trong tập dữ liệu cụ thể, chưa đủ để khẳng định khả năng mở rộng cho toàn bộ tri thức y khoa.

## 11. Section 6 — Discussion and Conclusion

Paper kết luận rằng medical calculation phải được xem như một quá trình suy luận có cấu trúc, không phải bài toán dự đoán một số duy nhất.

Ba kết luận quan trọng là:
- Final-answer accuracy có thể che giấu lỗi nghiêm trọng.
- Đánh giá từng bước tạo ra phản hồi minh bạch và có khả năng hành động.
- RAG và code execution có thể cải thiện độ tin cậy mà không cần fine-tuning.

Paper cũng đề xuất một thay đổi rộng hơn trong đánh giá AI y tế:
`End-task score` $\longrightarrow$ `Domain-grounded process evaluation`

Tức là không chỉ hỏi mô hình có trả lời đúng không, mà còn hỏi:
- Dựa trên kiến thức nào?
- Trích xuất thông tin nào?
- Sai ở đâu?
- Lỗi đó có ý nghĩa lâm sàng như thế nào?
- Có thể dùng module nào để khắc phục?

## 12. Section 7 — Limitations

Paper thừa nhận bốn giới hạn chính.

**1. Structured single-turn tasks**
Các case đều là bài toán tương đối rõ ràng và có công thức xác định. Chúng chưa thể hiện đầy đủ:
- Bệnh án dài.
- Thông tin mâu thuẫn.
- Chuyển đổi ngữ cảnh.
- Ngoại lệ trong guideline.
- Trao đổi nhiều lượt với bác sĩ hoặc bệnh nhân.

**2. Chỉ sử dụng tiếng Anh và clinical note đã được tuyển chọn**
Kết quả chưa được kiểm chứng trên:
- EHR nhiễu.
- Văn bản viết tắt.
- Đa ngôn ngữ.
- Hội thoại với bệnh nhân.
- Bệnh án tiếng Việt.

**3. Phụ thuộc vào LLM-as-a-Judge**
LLM Judge cũng có thể:
- Hiểu sai vấn đề lâm sàng.
- Bị ảnh hưởng bởi cách trình bày của câu trả lời.
- Tạo ra lỗi đánh giá.
- Lan truyền lỗi từ ground truth.

**4. Phụ thuộc vào formula bank đúng**
MedRaC giả định công thức, guideline và các hệ số trong knowledge base đều chính xác và cập nhật. Trong triển khai thực tế, công thức y khoa có thể có nhiều phiên bản hoặc thay đổi theo guideline.

## 13. Section 8 — Ethics Statement

Dữ liệu được sử dụng là dữ liệu công khai và đã ẩn danh. Hệ thống chỉ nhằm mục đích đánh giá và nghiên cứu, không được thiết kế để chẩn đoán hoặc triển khai trực tiếp trong lâm sàng.

Tác giả nhấn mạnh:
- Không vượt qua benchmark thì không nên dùng cho medical calculation thực tế.
- Vượt qua benchmark chỉ là điều kiện cần, không phải điều kiện đủ.
- Kết quả phải được chuyên gia y tế kiểm tra.

Hai chuyên gia y khoa tham gia annotation được trả 40 USD mỗi giờ.

## 14. Thuật ngữ quan trọng cần ghi nhớ

| Thuật ngữ | Giải thích |
| --- | --- |
| Medical calculation | Tính toán các đại lượng hoặc điểm số hỗ trợ quyết định lâm sàng |
| Clinical vignette | Mô tả ngắn về bệnh nhân và tình trạng lâm sàng |
| Calculator | Công thức hoặc hệ thống chấm điểm y khoa |
| Equation-based | Calculator dựa trên công thức toán học |
| Rule-based | Calculator gán điểm theo tiêu chí lâm sàng |
| MedCalc-Bench | Benchmark đánh giá LLM trên medical calculation |
| MDCalc | Nguồn calculator và công thức y khoa được sử dụng |
| LLM Test Taker | LLM thực hiện bài toán |
| LLM-as-a-Judge | Dùng LLM để đánh giá đầu ra của LLM khác |
| Step-wise evaluation | Đánh giá độc lập từng bước suy luận |
| Error attribution | Xác định và phân loại nguồn gây lỗi |
| Formula hallucination | LLM bịa hoặc làm biến dạng công thức |
| RAG | Truy xuất kiến thức bên ngoài trước khi sinh đáp án |
| Formula bank | Kho công thức dùng cho retrieval |
| Embedding | Vector biểu diễn ý nghĩa của câu hỏi hoặc tài liệu |
| Code execution | Sinh và thực thi chương trình để tính kết quả |
| CoT | Chain-of-Thought, trình bày suy luận từng bước |
| One-shot prompting | Cung cấp một ví dụ mẫu trong prompt |
| Self-Refine | Mô hình tự phê bình và sửa câu trả lời |
| MedPrompt | Phương pháp truy xuất các ví dụ tương tự để hỗ trợ reasoning |
| Ablation study | Loại bỏ từng thành phần để đo đóng góp |
| Conditional Correctness | Xác suất một bước đúng khi các bước trước đều đúng |
| First Error Attribution | Tỷ lệ một bước là lỗi đầu tiên trong các case thất bại |
| Jaccard similarity | Độ tương đồng giữa hai tập nhãn lỗi |
| EHR | Electronic Health Record — hồ sơ sức khỏe điện tử |
| BSA | Body Surface Area — diện tích bề mặt cơ thể |

## 15. Đánh giá paper dưới góc nhìn AI Researcher

### Điểm mạnh
Paper có một câu hỏi nghiên cứu rất thực tế: độ chính xác số học không đồng nghĩa với độ tin cậy lâm sàng.
Cách phân rã lỗi thành `Formula – Extraction – Calculation – Answer` vừa dễ hiểu, vừa cho phép thiết kế giải pháp nhắm đúng nguyên nhân. MedRaC cũng có tính module rõ ràng:
- `Knowledge error` $\rightarrow$ `RAG`
- `Arithmetic error` $\rightarrow$ `Code`

Việc làm sạch benchmark và kiểm chứng LLM Judge bằng chuyên gia giúp paper thuyết phục hơn.

### Điểm hạn chế đáng thảo luận trong seminar
- Chỉ có 46 case trong human evaluation.
- Chưa báo cáo confidence interval hoặc kiểm định ý nghĩa thống kê.
- Dùng percent agreement thay vì các chỉ số hiệu chỉnh theo ngẫu nhiên.
- Direct và reasoning methods không được đánh giá bằng cùng tiêu chuẩn.
- LLM Judge vẫn có thể sai và phụ thuộc vào model cụ thể.
- MedRaC chưa giải quyết tốt clinical interpretation.
- Code execution có thể xử lý số học nhưng không sửa được dữ liệu đầu vào sai.
- Formula bank phải chính xác, cập nhật và phù hợp guideline.
- Paper có một số điểm không nhất quán giữa văn bản và bảng, ví dụ Formula CC trong ablation.
- Đây chủ yếu là paper về LLM evaluation, RAG và agentic tool use, không đề xuất kiến trúc mạng neural hoặc phương pháp training mới.

> **Điểm cuối đặc biệt quan trọng đối với môn Học Sâu**: khi báo cáo, bạn nên liên hệ paper với deep learning thông qua năng lực reasoning của LLM, embedding retrieval, hallucination, in-context learning và tool-augmented inference; không nên trình bày MedRaC như một kiến trúc deep neural network mới.
