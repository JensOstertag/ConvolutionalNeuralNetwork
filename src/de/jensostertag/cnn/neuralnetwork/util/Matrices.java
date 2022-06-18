package de.jensostertag.cnn.neuralnetwork.util;

import java.util.Arrays;

public class Matrices {
    public static int[] getSize(double[][] matrix) {
        int height = matrix.length;
        int width = 0;
    
        for(int i = 0; i < matrix.length; i++) {
            if(i != 0) {
                if(matrix[i].length != width)
                    throw new IllegalArgumentException("Matrix is not of correct Size");
            } else
                width = matrix[i].length;
        }
        
        return new int[] {height, width};
    }
    
    public static int[] getSize(double[][][] matrix) {
        int depth = matrix.length;
        int height = 0;
        int width = 0;
        
        for(int i = 0; i < matrix.length; i++) {
            for(int j = 0; j < matrix[i].length; j++) {
                if(i != 0 && j != 0) {
                    if(matrix[i].length != height || matrix[i][j].length != width)
                        throw new IllegalArgumentException("Matrix is not of correct Size");
                } else {
                    height = matrix[i].length;
                    width = matrix[i][j].length;
                }
            }
        }
        
        return new int[] {depth, height, width};
    }
    
    public static boolean validateSize(double[] matrix, int height) {
        return matrix.length == height;
    }
    
    public static boolean validateSize(double[][] matrix, int height, int width) {
        int[] dimensions = getSize(matrix);
        return Arrays.equals(dimensions, new int[]{height, width});
    }
    
    public static boolean validateSize(double[][][] matrix, int depth, int height, int width) {
        int[] dimensions = getSize(matrix);
        return Arrays.equals(dimensions, new int[]{depth, height, width});
    }
    
    public static int[] getSameSize(double[] vector1, double[] vector2) {
        int size1 = vector1.length;
        int size2 = vector2.length;
        
        if(size1 == size2)
            return new int[]{size1};
        else
            throw new IllegalArgumentException("Vector1 and Vector2 are not of same Size");
    }
    
    public static int[] getSameSize(double[][] matrix1, double[][] matrix2) {
        int[] dimensions1 = getSize(matrix1);
        int[] dimensions2 = getSize(matrix2);
        
        if(Arrays.equals(dimensions1, dimensions2))
            return dimensions1;
        else
            throw new IllegalArgumentException("Matrix1 and Matrix2 are not of same Size");
    }
    
    public static int[] highestValue(double[][] matrix) {
        int[] dimensions = getSize(matrix);
        int[] output = new int[2];
        double highestValue = Double.NEGATIVE_INFINITY;
        for(int i = 0; i < matrix.length; i++) {
            for(int j = 0; j < matrix[i].length; j++) {
                if(matrix[i][j] > highestValue) {
                    highestValue = matrix[i][j];
                    output[0] = i;
                    output[1] = j;
                }
            }
        }
        
        return output;
    }
    
    public static double sum(double[][] matrix) {
        int[] dimensions = getSize(matrix);
        double sum = 0;
        for(int i = 0; i < matrix.length; i++)
            for(int j = 0; j < matrix[i].length; j++)
                sum += matrix[i][j];
        return sum;
    }
    
    public static double[] singleValueMatrix(int height, double value) {
        double[] output = new double[height];
        Arrays.fill(output, value);
        return output;
    }
    
    public static double[][] singleValueMatrix(int height, int width, double value) {
        double[][] output = new double[height][width];
        for(double[] vectors : output)
            Arrays.fill(vectors, value);
        
        return output;
    }
    
    public static double[][] randomMatrix(int height, int width, double min, double max) {
        double[][] output = new double[height][width];
        for(int i = 0; i < output.length; i++)
            for(int j = 0; j < output[i].length; j++)
                output[i][j] = min + Math.random() * (max - min);
    
        return output;
    }
    
    public static double[][] center(double[][] matrix, int height, int width) {
        int[] dimensions = getSize(matrix);
        if(dimensions[0] <= height && dimensions[1] <= width) {
            if((height - dimensions[0]) % 2 == 0 && (width - dimensions[1]) % 2 == 0) {
                double[][] output = new double[height][width];
                int heightOffset = (height - dimensions[0]) / 2;
                int widthOffset = (width - dimensions[1]) / 2;
                for(int i = 0; i < matrix.length; i++)
                    for(int j = 0; j < matrix[i].length; j++)
                        output[i + heightOffset][j + widthOffset] = matrix[i][j];
                return output;
            } else
                throw new IllegalArgumentException("The given Matrix cannot be centered within the expected Output Matrix");
        } else
            throw new IllegalArgumentException("Cannot center a Matrix within a smaller Matrix");
    }
    
    public static double[] add(double[] vector1, double[] vector2) {
        int[] dimensions = getSameSize(vector1, vector2);
        double[] output = new double[dimensions[0]];
        for(int i = 0; i < output.length; i++)
            output[i] = vector1[i] + vector2[i];
        
        return output;
    }
    
    public static double[][] add(double[][] matrix1, double[][] matrix2) {
        int[] dimensions = getSameSize(matrix1, matrix2);
        double[][] output = new double[dimensions[0]][dimensions[1]];
        for(int i = 0; i < output.length; i++)
            for(int j = 0; j < output[i].length; j++)
                output[i][j] = matrix1[i][j] + matrix2[i][j];
        
        return output;
    }
    
    public static double[][] multiply(double[][] matrix1, double[][] matrix2) {
        int[] dimensions1 = getSize(matrix1);
        int[] dimensions2 = getSize(matrix2);
        if(dimensions1[1] == dimensions2[0]) {
            double[][] output = new double[dimensions1[0]][dimensions2[1]];
            for(int i = 0; i < output.length; i++)
                for(int j = 0; j < output[i].length; j++)
                    for(int k = 0; k < dimensions1[1]; k++)
                        output[i][j] += matrix1[i][k] * matrix2[k][j];
            
            return output;
        } else
            throw new IllegalArgumentException("Cannot calculate the Product of these Matrices");
    }
    
    public static double[][] dotProduct(double[][] matrix1, double[][] matrix2) {
        int[] dimensions = getSameSize(matrix1, matrix2);
        double[][] output = new double[dimensions[0]][dimensions[1]];
        for(int i = 0; i < output.length; i++)
            for(int j = 0; j < output[i].length; j++)
                output[i][j] = matrix1[i][j] * matrix2[i][j];
        return output;
    }
    
    public static double[] multiplyConstant(double[] vector, double constant) {
        double[] output = new double[vector.length];
        for(int i = 0; i < vector.length; i++)
            output[i] = vector[i] * constant;
        
        return output;
    }
    
    public static double[][] multiplyConstant(double[][] matrix, double constant) {
        int[] dimensions = getSize(matrix);
        double[][] output = new double[dimensions[0]][dimensions[1]];
        for(int i = 0; i < matrix.length; i++)
            for(int j = 0; j < matrix[i].length; j++)
                output[i][j] = matrix[i][j] * constant;
    
        return output;
    }
    
    public static double[][] transpose(double[][] matrix) {
        int[] dimensions = getSize(matrix);
        double[][] output = new double[dimensions[1]][dimensions[0]];
        for(int i = 0; i < output.length; i++)
            for(int j = 0; j < output[i].length; j++)
                output[i][j] = matrix[j][i];
        
        return output;
    }
    
    public static double[][] rotate(double[][] matrix) {
        int[] dimensions = getSize(matrix);
        double[][] output = new double[dimensions[0]][dimensions[1]];
        for(int i = output.length - 1; i >= 0; i--)
            for(int j = output[i].length - 1; j >= 0; j--)
                output[i][j] = matrix[dimensions[0] - 1 - i][dimensions[1] - 1 - j];
        return output;
    }
    
    public static double[][] subMatrix(double[][] matrix, int heightOffset, int widthOffset, int heightSize, int widthSize) {
        double[][] output = new double[heightSize][widthSize];
        for(int i = 0; i < output.length; i++)
            for(int j = 0; j < output[i].length; j++)
                output[i][j] = matrix[i + heightOffset][j + widthOffset];
        
        return output;
    }
    
    public static double[][] asMatrix(double[] vector) {
        double[][] output = new double[1][vector.length];
        for(int i = 0; i < vector.length; i++)
            output[0][i] = vector[i];
        return output;
    }
    
    public static double[] asVector(double[][] matrix) {
        int[] dimensions = getSize(matrix);
        if(dimensions[0] == 1)
            return matrix[0];
        else
            throw new IllegalArgumentException("Matrix is not of a Vector Format");
    }
    
    public static double[] flatten(double[][] matrix) {
        int[] dimensions = getSize(matrix);
        double[] output = new double[dimensions[0] * dimensions[1]];
        
        for(int i = 0; i < matrix.length; i++)
            for(int j = 0; j < matrix[i].length; j++)
                output[i * matrix[i].length + j] = matrix[i][j];
        
        return output;
    }
    
    public static double[] flatten(double[][][] matrix) {
        int[] dimensions = getSize(matrix);
        double[] output = new double[dimensions[0] * dimensions[1] * dimensions[2]];
        
        for(int i = 0; i < matrix.length; i++)
            for(int j = 0; j < matrix[i].length; j++)
                for(int k = 0; k < matrix[i][j].length; k++)
                    output[i * (matrix[i].length * matrix[i][j].length) + j * matrix[i][j].length + k] = matrix[i][j][k];
        
        return output;
    }
    
    public static double[][] convolve(double[][] matrix, double[][] kernel) {
        int[] matrixDimensions = getSize(matrix);
        int[] kernelDimensions = getSize(kernel);
    
        double[][] output = new double[matrixDimensions[0] - kernelDimensions[0] + 1][matrixDimensions[1] - kernelDimensions[1] + 1];
        for(int i = 0; i < output.length; i++) {
            for(int j = 0; j < output[i].length; j++) {
                double[][] subMatrix = subMatrix(matrix, i, j, kernelDimensions[0], kernelDimensions[1]);
                output[i][j] = sum(dotProduct(subMatrix, kernel));
            }
        }
        return output;
    }
}
