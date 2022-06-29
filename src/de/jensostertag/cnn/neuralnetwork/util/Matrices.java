package de.jensostertag.cnn.neuralnetwork.util;

import java.util.Arrays;

public class Matrices {
    /**
     *  Get the Size of a 2-Dimensional Matrix
     *  @param matrix Matrix to get the Size
     *  @return Array with two Values: Height and Width
     */
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
    
    /**
     * Get the Size of a 3-Dimensional Matrix
     * @param matrix Matrix to get the Size
     * @return Array with three Values: Depth, Height and Width
     */
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
    
    /**
     * Validate the Size of a Vector
     * @param vector Vector to validate the Size
     * @param height Required Height
     * @return Vector is of correct Size
     */
    public static boolean validateSize(double[] vector, int height) {
        return vector.length == height;
    }
    
    /**
     * Validate the Size of a 2-Dimensional Matrix
     * @param matrix Matrix to validate the Size
     * @param height Required Height
     * @param width Required Width
     * @return Matrix is of correct Size
     */
    public static boolean validateSize(double[][] matrix, int height, int width) {
        int[] dimensions = getSize(matrix);
        return Arrays.equals(dimensions, new int[]{height, width});
    }
    
    /**
     * Validate the Size of a 3-Dimensional Matrix
     * @param matrix Matrix to validate the Size
     * @param depth Required Depth
     * @param height Required Width
     * @param width Required Height
     * @return Matrix is of correct Size
     */
    public static boolean validateSize(double[][][] matrix, int depth, int height, int width) {
        int[] dimensions = getSize(matrix);
        return Arrays.equals(dimensions, new int[]{depth, height, width});
    }
    
    /**
     * Get the common Size of two Vectors
     * @param vector1 First Vector
     * @param vector2 Second Vector
     * @return Array with one Value: Common Height
     */
    public static int[] getSameSize(double[] vector1, double[] vector2) {
        int size1 = vector1.length;
        int size2 = vector2.length;
        
        if(size1 == size2)
            return new int[]{size1};
        else
            throw new IllegalArgumentException("Vector1 and Vector2 are not of same Size");
    }
    
    /**
     * Get the common Size of two 2-Dimensional Matrices
     * @param matrix1 First Matrix
     * @param matrix2 Second Matrix
     * @return Array with two Values: Common Width and Common Height
     */
    public static int[] getSameSize(double[][] matrix1, double[][] matrix2) {
        int[] dimensions1 = getSize(matrix1);
        int[] dimensions2 = getSize(matrix2);
        
        if(Arrays.equals(dimensions1, dimensions2))
            return dimensions1;
        else
            throw new IllegalArgumentException("Matrix1 and Matrix2 are not of same Size");
    }
    
    /**
     * Get the Position of the highest Value within a 2-Dimensional Matrix
     * @param matrix Matrix
     * @return Array with two Values: Height Coordinate and Width Coordinate
     */
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
    
    /**
     * Get the Sum of all Values within a Matrix
     * @param matrix Matrix to calculate the Sum
     * @return Sum of all Values
     */
    public static double sum(double[][] matrix) {
        int[] dimensions = getSize(matrix);
        double sum = 0;
        for(int i = 0; i < matrix.length; i++)
            for(int j = 0; j < matrix[i].length; j++)
                sum += matrix[i][j];
        return sum;
    }
    
    /**
     * Generate a Vector and fill it with one Value
     * @param height Vector Height
     * @param value Value to fill the Vector
     * @return Generated Vector
     */
    public static double[] singleValueMatrix(int height, double value) {
        double[] output = new double[height];
        Arrays.fill(output, value);
        return output;
    }
    
    /**
     * Generate a 2-Dimensional Matrix and fill it with one Value
     * @param height Matrix Height
     * @param width Matrix Width
     * @param value Value to fill the Matrix
     * @return Generated Matrix
     */
    public static double[][] singleValueMatrix(int height, int width, double value) {
        double[][] output = new double[height][width];
        for(double[] vectors : output)
            Arrays.fill(vectors, value);
        
        return output;
    }
    
    /**
     * Generate a 2-Dimensional Matrix and fill it with random Values
     * @param height Matrix Height
     * @param width Matrix Width
     * @param min Minimum Value
     * @param max Maximum Value
     * @return Generated Matrix
     */
    public static double[][] randomMatrix(int height, int width, double min, double max) {
        double[][] output = new double[height][width];
        for(int i = 0; i < output.length; i++)
            for(int j = 0; j < output[i].length; j++)
                output[i][j] = min + Math.random() * (max - min);
    
        return output;
    }
    
    /**
     * Center a Matrix within another Matrix
     * @param matrix Matrix to center
     * @param height Outer Matrix Height
     * @param width Outer Matrix Width
     * @return Outer Matrix
     */
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
    
    /**
     * Add two Vectors
     * @param vector1 First Vector
     * @param vector2 Second Vector
     * @return Added Vectors
     */
    public static double[] add(double[] vector1, double[] vector2) {
        int[] dimensions = getSameSize(vector1, vector2);
        double[] output = new double[dimensions[0]];
        for(int i = 0; i < output.length; i++)
            output[i] = vector1[i] + vector2[i];
        
        return output;
    }
    
    /**
     * Add two 2-Dimensional Matrices
     * @param matrix1 First Matrix
     * @param matrix2 Second Matrix
     * @return Added Matrices
     */
    public static double[][] add(double[][] matrix1, double[][] matrix2) {
        int[] dimensions = getSameSize(matrix1, matrix2);
        double[][] output = new double[dimensions[0]][dimensions[1]];
        for(int i = 0; i < output.length; i++)
            for(int j = 0; j < output[i].length; j++)
                output[i][j] = matrix1[i][j] + matrix2[i][j];
        
        return output;
    }
    
    /**
     * Multiply two 2-Dimensional Matrices
     * @param matrix1 First Matrix
     * @param matrix2 Second Matrix
     * @return Multiplied Matrix
     */
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
    
    /**
     * Multiply two 2-Dimensional Matrices (Value-wise)
     * @param matrix1 First Matrix
     * @param matrix2 Second Matrix
     * @return Multiplied Matrices
     */
    public static double[][] dotProduct(double[][] matrix1, double[][] matrix2) {
        int[] dimensions = getSameSize(matrix1, matrix2);
        double[][] output = new double[dimensions[0]][dimensions[1]];
        for(int i = 0; i < output.length; i++)
            for(int j = 0; j < output[i].length; j++)
                output[i][j] = matrix1[i][j] * matrix2[i][j];
        return output;
    }
    
    /**
     * Multiply a Vector with a Constant
     * @param vector Vector to multiply
     * @param constant Constant
     * @return Multiplied Vector
     */
    public static double[] multiplyConstant(double[] vector, double constant) {
        double[] output = new double[vector.length];
        for(int i = 0; i < vector.length; i++)
            output[i] = vector[i] * constant;
        
        return output;
    }
    
    /**
     * Multiply a 2-Dimensional Matrix with a Constant
     * @param matrix Matrix to multiply
     * @param constant Constant
     * @return Multiplied Matrix
     */
    public static double[][] multiplyConstant(double[][] matrix, double constant) {
        int[] dimensions = getSize(matrix);
        double[][] output = new double[dimensions[0]][dimensions[1]];
        for(int i = 0; i < matrix.length; i++)
            for(int j = 0; j < matrix[i].length; j++)
                output[i][j] = matrix[i][j] * constant;
    
        return output;
    }
    
    /**
     * Transpose a 2-Dimensional Matrix
     * @param matrix Matrix to transpose
     * @return Transposed Matrix
     */
    public static double[][] transpose(double[][] matrix) {
        int[] dimensions = getSize(matrix);
        double[][] output = new double[dimensions[1]][dimensions[0]];
        for(int i = 0; i < output.length; i++)
            for(int j = 0; j < output[i].length; j++)
                output[i][j] = matrix[j][i];
        
        return output;
    }
    
    /**
     * Rotate a 2-Dimensional Matrix by 180°
     * @param matrix Matrix to rotate
     * @return Rotated Matrix
     */
    public static double[][] rotate(double[][] matrix) {
        int[] dimensions = getSize(matrix);
        double[][] output = new double[dimensions[0]][dimensions[1]];
        for(int i = output.length - 1; i >= 0; i--)
            for(int j = output[i].length - 1; j >= 0; j--)
                output[i][j] = matrix[dimensions[0] - 1 - i][dimensions[1] - 1 - j];
        return output;
    }
    
    /**
     * Generate a 2-Dimensional Sub-Matrix from a 2-Dimensional Matrix
     * @param matrix Matrix
     * @param heightOffset Height Offset
     * @param widthOffset Width Offset
     * @param heightSize Height of the Sub-Matrix
     * @param widthSize Width of the Sub-Matrix
     * @return Sub-Matrix
     */
    public static double[][] subMatrix(double[][] matrix, int heightOffset, int widthOffset, int heightSize, int widthSize) {
        double[][] output = new double[heightSize][widthSize];
        for(int i = 0; i < output.length; i++)
            for(int j = 0; j < output[i].length; j++)
                output[i][j] = matrix[i + heightOffset][j + widthOffset];
        
        return output;
    }
    
    /**
     * Convert a Vector to a 2-Dimensional Matrix with Height 1
     * @param vector Vector to convert
     * @return Matrix
     */
    public static double[][] asMatrix(double[] vector) {
        double[][] output = new double[1][vector.length];
        for(int i = 0; i < vector.length; i++)
            output[0][i] = vector[i];
        return output;
    }
    
    /**
     * Convert a 2-Dimensional Matrix with Height 1 to a Vector
     * @param matrix Matrix to convert
     * @return Vector
     */
    public static double[] asVector(double[][] matrix) {
        int[] dimensions = getSize(matrix);
        if(dimensions[0] == 1)
            return matrix[0];
        else
            throw new IllegalArgumentException("Matrix is not of a Vector Format");
    }
    
    /**
     * Flatten a 2-Dimensional Matrix
     * @param matrix Matrix to flatten
     * @return Flattened Vector
     */
    public static double[] flatten(double[][] matrix) {
        int[] dimensions = getSize(matrix);
        double[] output = new double[dimensions[0] * dimensions[1]];
        
        for(int i = 0; i < matrix.length; i++)
            for(int j = 0; j < matrix[i].length; j++)
                output[i * matrix[i].length + j] = matrix[i][j];
        
        return output;
    }
    
    /**
     * Flatten a 3-Dimensional Matrix
     * @param matrix Matrix to flatten
     * @return Flattened Vector
     */
    public static double[] flatten(double[][][] matrix) {
        int[] dimensions = getSize(matrix);
        double[] output = new double[dimensions[0] * dimensions[1] * dimensions[2]];
        
        for(int i = 0; i < matrix.length; i++)
            for(int j = 0; j < matrix[i].length; j++)
                for(int k = 0; k < matrix[i][j].length; k++)
                    output[i * (matrix[i].length * matrix[i][j].length) + j * matrix[i][j].length + k] = matrix[i][j][k];
        
        return output;
    }
    
    /**
     * Convolve a 2-Dimensional Matrix
     * @param matrix Matrix to Convolve
     * @param kernel Kernel Matrix
     * @return Convolved Matrix
     */
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
