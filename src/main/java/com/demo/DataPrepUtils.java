package com.demo;

// --- NHÓM 1: TIỆN ÍCH HỆ THỐNG TỆP TIN ---
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.io.File;

/**
 * Bo cong cu so che du lieu (Preprocessing Tools).
 * Thuc hien phan tach du lieu goc thanh tap Train (Sach) va Test (Hon hop).
 */
public class DataPrepUtils {

    /**
     * Chia du lieu Kaggle theo chien luoc Hoc ban giam sat (Semi-supervised).
     * @param inputPath Duong dan file creditcard.csv goc
     */
    public static void splitKaggleData(String inputPath) {
        String trainPath = "src/main/resources/data/train_clean.csv";
        String testPath = "src/main/resources/data/test_mixed.csv";

        try (BufferedReader br = new BufferedReader(new FileReader(inputPath));
             PrintWriter trainWriter = new PrintWriter(trainPath);
             PrintWriter testWriter = new PrintWriter(testPath)) {

            String header = br.readLine();
            trainWriter.println(header);
            testWriter.println(header);

            String line;
            int normalCount = 0;
            int fraudCount = 0;

            System.out.println("[INFO] Dang phan tach du lieu... Vui long cho.");

            while ((line = br.readLine()) != null) {
                // Kiem tra nhãn (Class) o cuoi dong
                if (line.trim().endsWith("\"1\"") || line.trim().endsWith(",1")) {
                    // Tat ca giao dich gian lận duoc dua vao tap Test
                    testWriter.println(line);
                    fraudCount++;
                } else {
                    // Lay 200,000 giao dich binh thuong dau tien de Train
                    if (normalCount < 200000) {
                        trainWriter.println(line);
                    } else {
                        testWriter.println(line);
                    }
                    normalCount++;
                }
            }
            System.out.println("[SUCCESS] Da chia du lieu:");
            System.out.println(" - Tap Train (Normal only): " + 200000);
            System.out.println(" - Tap Test (Mixed): " + (normalCount - 200000 + fraudCount));

        } catch (Exception e) {
            System.err.println("[LOI] Khong the phan tach du lieu: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        // Luu y: Ban can co file creditcard.csv tai thu muc resources/data/
        String path = "src/main/resources/data/creditcard.csv";
        File file = new File(path);
        if (file.exists()) {
            splitKaggleData(path);
        } else {
            System.err.println("[LOI] Khong tim thay file goc tai: " + path);
        }
    }
}
