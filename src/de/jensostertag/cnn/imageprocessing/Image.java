package de.jensostertag.cnn.imageprocessing;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Image {
    protected final String path;
    protected final BufferedImage image;
    public Image(String path) {
        this.path = path;
        BufferedImage image = null;

        try {
            image = ImageIO.read(new File(this.path));
        } catch(IOException e) {
            e.printStackTrace();
        }

        this.image = image;
    }

    public Image(BufferedImage image) {
        this.path = null;
        this.image = image;
    }

    public BufferedImage getImage() {
        return this.image;
    }

    public int getWidth() {
        return this.image.getWidth();
    }

    public int getHeight() {
        return this.image.getHeight();
    }

    public int[][][] getPixels() {
        int[][][] pixels = new int[3][this.image.getHeight()][this.image.getWidth()];

        for(int i = 0; i < this.image.getHeight(); i++) {
            for(int j = 0; j < this.image.getWidth(); j++) {
                int rgb = this.image.getRGB(i, j);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                pixels[0][i][j] = r;
                pixels[1][i][j] = g;
                pixels[2][i][j] = b;
            }
        }

        return pixels;
    }

    public void save(String path) {
        File file = new File(path);

        try {
            ImageIO.write(this.image, "jpg", file);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
