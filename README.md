# deepSegmentationTool
Easy Image Segmentation Tool based on Fully convolutional network

**Tool has 2 modes:**
1) **Training of new deep neural network** (train_flag = True).
2) **Testing of existing (trained) deep neural network** (train_flag = False).

For training mode you need two folders:
1) Training folder with images
2) Markup folder with grayscale masked images, txt-files or xml-files

Files in training folder and markup folder must have the same names, for example
        
        /train_img/img1.jpg
        /train_markup/img1.xml

**Fragment of masked image (255 - pixel of object, 0 - background pixel)**
        
        0  255 255 255
        0  0   0   0
        0  0   255 255
     
**Example of txt-file content:**
        
        False
        205 144 40 40
        417 149 38 38
        38 130 48 48
        246 134 25 25

**Example of xml-file content - pascal VOC format:**
        
        <annotation>
            <folder>Светофор</folder>
            <filename>00fcf1d745d76a3696f2ca99678f2052.jpg</filename>
            <path>E:\Светофор\00fcf1d745d76a3696f2ca99678f2052.jpg</path>
            <source>
                <database>Unknown</database>
            </source>
            <size>
                <width>455</width>
                <height>256</height>
                <depth>3</depth>
            </size>
            <segmented>0</segmented>
            <object>
                <name>Trafficlight</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>248</xmin>
                    <ymin>106</ymin>
                    <xmax>256</xmax>
                    <ymax>125</ymax>
                </bndbox>
            </object>
            <object>
                <name>Trafficlight</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>350</xmin>
                    <ymin>98</ymin>
                    <xmax>358</xmax>
                    <ymax>117</ymax>
                </bndbox>
            </object>
        </annotation>


***Description of main modules:***

***deepSegmentationTool.py*** - main module for training and testing of deep neural network for image segmentation task
