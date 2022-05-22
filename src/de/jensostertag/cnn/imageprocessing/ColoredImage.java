package de.jensostertag.cnn.imageprocessing;

import java.awt.image.BufferedImage;

public class ColoredImage extends Image {
    public ColoredImage(String path) {
        super(path);
    }

    public ColoredImage(BufferedImage image) {
        super(image);
    }
}
