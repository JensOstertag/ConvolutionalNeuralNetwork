package de.jensostertag.cnn.examples;

import de.jensostertag.cnn.activationfunctions.ActivationFunction;
import de.jensostertag.cnn.dataset.ImageDataset;
import de.jensostertag.cnn.lossfunctions.LossFunction;
import de.jensostertag.cnn.neuralnetwork.CNN;
import de.jensostertag.cnn.neuralnetwork.layers.ConvolutionalLayer;
import de.jensostertag.cnn.neuralnetwork.layers.FlatteningLayer;
import de.jensostertag.cnn.neuralnetwork.layers.FullyConnectedLayer;
import de.jensostertag.cnn.neuralnetwork.layers.PoolingLayer;
import de.jensostertag.cnn.neuralnetwork.util.padding.PaddingType;
import de.jensostertag.cnn.neuralnetwork.util.pooling.Pooling;
import de.jensostertag.cnn.neuralnetwork.util.pooling.PoolingType;

public class MNIST {
    public static void main(String[] args) {
        ImageDataset trainingDataset = buildTrainingDataset();
        ImageDataset testingDataset = buildTestingDataset();
        
        CNN cnn = buildCNN();
        cnn.train(trainingDataset, testingDataset, LossFunction.lossMeanSquaredError, 100);
    }
    
    public static CNN buildCNN() {
        CNN cnn = new CNN();
        cnn.addLayer(new ConvolutionalLayer(1, 6, 28, 28, 3, PaddingType.SAME, ActivationFunction.activationStep));
        cnn.addLayer(new PoolingLayer(6, 28, 28, new Pooling(PoolingType.MAX, 2)));
        cnn.addLayer(new ConvolutionalLayer(6, 12, 14, 14, 3, PaddingType.SAME, ActivationFunction.activationStep));
        cnn.addLayer(new PoolingLayer(12, 14, 14, new Pooling(PoolingType.MAX, 2)));
        cnn.addLayer(new FlatteningLayer(12, 7, 7));
        cnn.addLayer(new FullyConnectedLayer(12*7*7, 6*7*7, ActivationFunction.activationStep));
        cnn.addLayer(new FullyConnectedLayer(6*7*7, 10, ActivationFunction.activationSigmoid));
        
        return cnn;
    }
    
    public static ImageDataset buildTrainingDataset() {
        ImageDataset trainingDataset = new ImageDataset();
        trainingDataset.addFolder("mnist-dataset/training/0/", new double[]{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, true);
        trainingDataset.addFolder("mnist-dataset/training/1/", new double[]{0, 1, 0, 0, 0, 0, 0, 0, 0, 0}, true);
        trainingDataset.addFolder("mnist-dataset/training/2/", new double[]{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, true);
        trainingDataset.addFolder("mnist-dataset/training/3/", new double[]{0, 0, 0, 1, 0, 0, 0, 0, 0, 0}, true);
        trainingDataset.addFolder("mnist-dataset/training/4/", new double[]{0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, true);
        trainingDataset.addFolder("mnist-dataset/training/5/", new double[]{0, 0, 0, 0, 0, 1, 0, 0, 0, 0}, true);
        trainingDataset.addFolder("mnist-dataset/training/6/", new double[]{0, 0, 0, 0, 0, 0, 1, 0, 0, 0}, true);
        trainingDataset.addFolder("mnist-dataset/training/7/", new double[]{0, 0, 0, 0, 0, 0, 0, 1, 0, 0}, true);
        trainingDataset.addFolder("mnist-dataset/training/8/", new double[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 0}, true);
        trainingDataset.addFolder("mnist-dataset/training/9/", new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 1}, true);
        trainingDataset.validateDataset();
        
        return trainingDataset;
    }
    
    public static ImageDataset buildTestingDataset() {
        ImageDataset testingDataset = new ImageDataset();
        testingDataset.addFolder("mnist-dataset/testing/0/", new double[]{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, true);
        testingDataset.addFolder("mnist-dataset/testing/1/", new double[]{0, 1, 0, 0, 0, 0, 0, 0, 0, 0}, true);
        testingDataset.addFolder("mnist-dataset/testing/2/", new double[]{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, true);
        testingDataset.addFolder("mnist-dataset/testing/3/", new double[]{0, 0, 0, 1, 0, 0, 0, 0, 0, 0}, true);
        testingDataset.addFolder("mnist-dataset/testing/4/", new double[]{0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, true);
        testingDataset.addFolder("mnist-dataset/testing/5/", new double[]{0, 0, 0, 0, 0, 1, 0, 0, 0, 0}, true);
        testingDataset.addFolder("mnist-dataset/testing/6/", new double[]{0, 0, 0, 0, 0, 0, 1, 0, 0, 0}, true);
        testingDataset.addFolder("mnist-dataset/testing/7/", new double[]{0, 0, 0, 0, 0, 0, 0, 1, 0, 0}, true);
        testingDataset.addFolder("mnist-dataset/testing/8/", new double[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 0}, true);
        testingDataset.addFolder("mnist-dataset/testing/9/", new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 1}, true);
        testingDataset.validateDataset();
        
        return testingDataset;
    }
}
