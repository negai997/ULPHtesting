#pragma once
#include<iostream>

const int MAX_SIZE = 100;

void displayMatrix(double matrix[][MAX_SIZE], int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void inverseMatrix(double matrix[][MAX_SIZE], double inverse[][MAX_SIZE], int size) {
    // �����������
    double augmented[MAX_SIZE][2 * MAX_SIZE];
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            augmented[i][j] = matrix[i][j];
            augmented[i][j + size] = (i == j) ? 1 : 0;
        }
    }

    // �Ծ�����г����б任������������ɵ�λ����
    for (int i = 0; i < size; i++) {
        // ����i�еĵ�i��Ԫ�����ŵ�1
        double pivot = augmented[i][i];
        for (int j = 0; j < size * 2; j++) {
            augmented[i][j] /= pivot;
        }

        // �������е�i��Ԫ�ر�Ϊ0
        for (int j = 0; j < size; j++) {
            if (j != i) {
                double factor = augmented[j][i];
                for (int k = 0; k < size * 2; k++) {
                    augmented[j][k] -= factor * augmented[i][k];
                }
            }
        }
    }

    // �������������ȡ�����
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            inverse[i][j] = augmented[i][j + size];
        }
    }
}
