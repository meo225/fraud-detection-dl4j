package com.demo;

// --- NHÓM 1: MÔ HÌNH VÀ DỰ BÁO (DL4J CORE) ---
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

// --- NHÓM 2: ĐÁNH GIÁ CHỈ SỐ AI (EVALUATION) ---
import org.nd4j.evaluation.classification.Evaluation;

// --- NHÓM 3: TOÁN HỌC MA TRẬN & CHUẨN HÓA (ND4J) ---
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;

// --- NHÓM 4: ĐỌC DỮ LIỆU KIỂM THỬ (DATAVEC) ---
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;

import java.io.File;

/**
 * Chương trình Kiểm chứng hiệu năng mô hình trên tập dữ liệu chưa biết.
 */
public class FraudDetectionInference {

    public static void main(String[] args) {
        try {
            // 1. Khoi phuc Model va Bo chuan hoa (Normalizer) da hoc tu tap Train
            String modelPath = "src/main/resources/data/fraud_autoencoder.zip";
            String normPath = "src/main/resources/data/normalizer.bin";
            
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File(modelPath));
            NormalizerMinMaxScaler normalizer = NormalizerSerializer.getDefault().restore(new File(normPath));
            
            System.out.println("[INFO] Khoi phuc thanh cong Model va Bo chuan hoa STT 12.");

            // 2. Thiet lap luong doc du lieu Test
            RecordReader rr = new CSVRecordReader(1, ',');
            rr.initialize(new FileSplit(new File("src/main/resources/data/test_mixed.csv")));

            DataSetIterator testIterator = new RecordReaderDataSetIterator.Builder(rr, 1)
                    .classification(30, 2)
                    .build();

            // Ap dung bo chuan hoa da lưu de dam bao tinh dong nhat du lieu
            testIterator.setPreProcessor(normalizer);

            double threshold = 0.046; // Nguong MSE chan van hanh
            Evaluation eval = new Evaluation(2);

            System.out.println("[INFO] Dang tien hanh kiem tra tung giao dich...");

            while (testIterator.hasNext()) {
                DataSet ds = testIterator.next();
                INDArray features = ds.getFeatures();
                INDArray actualLabel = ds.getLabels();

                // Tinh toan sai so tai cau truc (Score)
                double score = model.score(new DataSet(features, features));
                
                // Phan loai bat thuong dua tren nguong Threshold
                int predictedClass = (score > threshold) ? 1 : 0;
                int actualClass = actualLabel.argMax(1).getInt(0);

                eval.eval(predictedClass, actualClass);
            }

            // 3. In ket qua thong ke (Recall, Precision, F1)
            System.out.println("\n--- BAO CAO HIEU NANG AI (STT 12) ---");
            System.out.println(eval.stats());
            System.out.println("Nguong MSE Threshold su dung: " + threshold);

        } catch (Exception e) {
            System.err.println("[LOI] Khong the thuc thi kiem chung: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
