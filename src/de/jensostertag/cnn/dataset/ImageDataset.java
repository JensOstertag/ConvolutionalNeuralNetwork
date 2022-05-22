package de.jensostertag.cnn.dataset;

import de.jensostertag.cnn.imageprocessing.ColoredImage;
import de.jensostertag.cnn.imageprocessing.GrayscaleImage;

import java.io.File;

public class ImageDataset extends Dataset {
    public void addFolder(String folderPath, Object[] expectedOutput, boolean grayScale) {
        if(!(super.isValid())) {
            File folder = new File(folderPath);
            if(folder.exists() && folder.isDirectory()) {
                for(final File file : folder.listFiles()) {
                    if(!(file.isDirectory())) {
                        if(grayScale) {
                            GrayscaleImage image = new GrayscaleImage(new ColoredImage(file.getPath()));
    
                            int[][][] pixels = image.getPixels();
                            double[][][] input = new double[1][image.getHeight()][image.getWidth()];
                            
                            for(int i = 0; i < input.length; i++)
                                for(int j = 0; j < input[i].length; j++)
                                    for(int k = 0; k < input[i][j].length; k++)
                                        input[i][j][k] = pixels[i][j][k] / 255d;
    
                            super.insertData(input, expectedOutput.clone());
                        } else {
                            ColoredImage image = new ColoredImage(file.getPath());
    
                            int[][][] pixels = image.getPixels();
                            double[][][] input = new double[3][image.getHeight()][image.getWidth()];
    
                            for(int i = 0; i < input.length; i++)
                                for(int j = 0; j < input[i].length; j++)
                                    for(int k = 0; k < input[i][j].length; k++)
                                        input[i][j][k] = pixels[i][j][k] / 255d;
                            
                            super.insertData(input, expectedOutput.clone());
                        }
                    }
                }
            } else
                throw new IllegalArgumentException("Given Path is not a Folder");
        } else
            throw new IllegalStateException("Cannot insert Data to a validated Dataset");
    }
}
