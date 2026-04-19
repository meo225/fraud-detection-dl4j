# Phát hiện gian lận thẻ tín dụng bằng Deeplearning4j

Dự án này triển khai mô hình mạng nơ-ron Autoencoder nhằm phát hiện các giao dịch bất thường trong dữ liệu thẻ tín dụng. Hệ thống được xây dựng trên ngôn ngữ Java, sử dụng framework Deeplearning4j cho các tính toán học sâu.

---

## 1. Dữ liệu đầu vào

Dự án sử dụng bộ dữ liệu thực tế từ Kaggle. Trước tiên cần chuẩn bị dữ liệu theo các bước sau:

*   **Nguồn dữ liệu:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
*   **Cài đặt:** Tải tệp `creditcard.csv` từ đường dẫn trên và lưu trữ tại thư mục: `src/main/resources/data/`

---

## 2. Hướng dẫn

Để khởi chạy hệ thống, vui lòng thực thi các lớp Java theo trình tự dưới đây:

### Bước 1: Tiền xử lý dữ liệu
Thực thi tệp **`DataPrepUtils.java`**. Mục tiêu của bước này là phân tách tệp dữ liệu gốc thành hai tập tin `train_clean.csv` (dữ liệu bình thường để huấn luyện) và `test_mixed.csv` (dữ liệu hỗn hợp để kiểm thử).

### Bước 2: Huấn luyện mô hình
Thực thi tệp **`FraudDetectionTrain.java`**. Quá trình huấn luyện sẽ học các đặc trưng của giao dịch bình thường. Thông số huấn luyện có thể được theo dõi trực tuyến tại địa chỉ `http://localhost:9000` trong quá trình thực thi.

### Bước 3: Đánh giá
Thực thi tệp **`FraudDetectionInference.java`**. Hệ thống sẽ sử dụng mô hình đã lưu để dự báo các giao dịch gian lận trên tập dữ liệu kiểm thử và xuất ra các chỉ số báo cáo hiệu năng.

---

## 3. Công nghệ sử dụng
*   Ngôn ngữ: Java 11 hoặc mới hơn.
*   Framework Deep Learning: Deeplearning4j.
*   Backend tính toán: ND4J (Native-platform).
*   Quản lý dự án: Maven.
