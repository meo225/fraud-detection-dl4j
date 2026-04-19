package com.demo;

// --- CẤU TRÚC MẠNG NƠ-RON  ---
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

// --- XỬ LÝ DỮ LIỆU & ETL (DATAVEC) ---
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;

// --- ĐỘNG CƠ TOÁN HỌC & CHUẨN HÓA (ND4J) ---
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;

// --- GIÁM SÁT & Traning UI ---
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.model.stats.StatsListener;

import java.io.File;

/**
 * Chương trình huấn luyện Mô hình Phát hiện gian lận (Kiến trúc Champion STT
 * 12).
 */
public class FraudDetectionTrain {

    public static MultiLayerConfiguration buildAutoencoder() {
        return new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(1e-3))
                .l2(4e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(30).nOut(18).activation(Activation.LEAKYRELU).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new DenseLayer.Builder().nIn(18).nOut(4).activation(Activation.LEAKYRELU).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new DenseLayer.Builder().nIn(4).nOut(18).activation(Activation.LEAKYRELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(18).nOut(30)
                        .activation(Activation.IDENTITY)
                        .build())
                .build();
    }

    public static void main(String[] args) {
        MultiLayerNetwork model = new MultiLayerNetwork(buildAutoencoder());
        model.init();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage, 10));

        try {
            int batchSize = 128;
            RecordReader recordReader = new CSVRecordReader(1, ',');
            recordReader.initialize(new FileSplit(new File("src/main/resources/data/train_clean.csv")));

            DataSetIterator iterator = new RecordReaderDataSetIterator.Builder(recordReader, batchSize)
                    .classification(30, 2)
                    .build();

            // QUAN TRỌNG: Tính toán bộ chuẩn hóa dựa trên tập huấn luyện
            NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
            normalizer.fit(iterator);
            iterator.setPreProcessor(normalizer);

            System.out.println("[INFO] Bat dau huan luyen Mo hinh Champion STT 12 (18-BN-4)...");
            int epochs = 15;
            for (int i = 0; i < epochs; i++) {
                iterator.reset();
                while (iterator.hasNext()) {
                    DataSet ds = iterator.next();
                    ds.setLabels(ds.getFeatures());
                    model.fit(ds);
                }
                System.out.println(" - Hoan tat Epoch " + (i + 1) + "/" + epochs);
            }

            String modelPath = "src/main/resources/data/fraud_autoencoder.zip";
            String normPath = "src/main/resources/data/normalizer.bin";

            ModelSerializer.writeModel(model, new File(modelPath), true);
            NormalizerSerializer.getDefault().write(normalizer, new File(normPath));

            System.out.println("[SUCCESS] Da luu Model va Normalizer tai src/main/resources/data/");
            System.out.println("\n[UI SERVER] Truy cap: http://localhost:9000 de xem do thi.");
            System.out.println("Nhan Enter de ket thuc chuong trinh...");
            new java.util.Scanner(System.in).nextLine();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
