package de.jensostertag.cnn.imageprocessing;

import java.awt.image.BufferedImage;

public class GrayscaleImage extends Image {
    private static BufferedImage grayscaleImage(BufferedImage coloredImage) {
        BufferedImage grayImage = new BufferedImage(coloredImage.getHeight(), coloredImage.getWidth(), coloredImage.getType());

        for(int i = 0; i < coloredImage.getHeight(); i++) {
            for(int j = 0; j < coloredImage.getWidth(); j++) {
                int rgb = coloredImage.getRGB(i, j);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                double rr = Math.pow(r / 255d, 2.2);
                double gg = Math.pow(g / 255d, 2.2);
                double bb = Math.pow(b / 255d, 2.2);

                double lum = .2126 * rr + .7152 * gg + .0722 * bb;

                int grayLevel = (int) (255d * Math.pow(lum, 1d / 2.2));
                int gray = (grayLevel << 16) + (grayLevel << 8) + grayLevel;
                grayImage.setRGB(i, j, gray);
            }
        }

        return grayImage;
    }

    public GrayscaleImage(ColoredImage coloredImage) {
        super(grayscaleImage(coloredImage.getImage()));
    }

    public int[][][] getPixels() {
        int[][][] pixels = new int[1][this.image.getHeight()][this.image.getWidth()];

        for(int i = 0; i < this.image.getHeight(); i++) {
            for(int j = 0; j < this.image.getWidth(); j++) {
                int rgb = this.image.getRGB(i, j);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                double rr = Math.pow(r / 255d, 2.2);
                double gg = Math.pow(g / 255d, 2.2);
                double bb = Math.pow(b / 255d, 2.2);

                double lum = .2126 * rr + .7152 * gg + .0722 * bb;

                int grayLevel = (int) (255d * Math.pow(lum, 1d / 2.2));
                pixels[0][i][j] = grayLevel;
            }
        }

        return pixels;
    }
}
