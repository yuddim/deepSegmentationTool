import sys
import time
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
import os
import numpy as np
from shutil import copyfile
from keras.callbacks import TensorBoard
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from PIL import ImageDraw
from xml.etree import ElementTree

smooth = 1.

#special metrics for FCN training on small blobs - Dice
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

#loss metrics for FCN training on base of Dice
def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

#clustering of grayscale mask - result of segmentation
def cluster_image(y_show, treshold, bottom_border = 0, scale = 1.0, k_means_flag = True):
    """
    Clustering of grayscale mask - result of segmentation
    :param y_show: input image
    :param treshold: threshold of binarization
    :param bottom_border: maximum y value to crop bottom of image
    :param scale: scale of image to increase clustering speed
    :param k_means_flag: usage of k-means method
    :return: rects - bounding boxes [rect_x1*scale, rect_y1*scale, rect_x2*scale, rect_y2*scale, rect_n_points*scale*scale, rect_ratio]
    """
    if bottom_border!=0:
        bottom_y = y_show.shape[0] - bottom_border
        y_show = y_show[0:bottom_y,:]
    if scale != 1.0:
        sizes = y_show.shape
        im_h = sizes[0]
        im_w = sizes[1]
        im_h_new = im_h // scale
        im_w_new = im_w // scale
        y_show = y_show.reshape(im_h_new,im_h//im_h_new, im_w_new, im_w//im_w_new).mean(axis=3).mean(axis=1)

        arg_X_two = np.argwhere(y_show > treshold)

    else:
        arg_X = np.argwhere(y_show > treshold)
        arg_X_two = np.delete(arg_X,2,1)

    start = time.time()
    db = DBSCAN(eps=2, min_samples=10).fit(arg_X_two)
    stop = time.time()
    sec = stop - start
    print("\r\nClustering time = %.4f sec" % sec, end=' ')
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d \r\n' % n_clusters_)

    unique_labels = set(labels)
    rects = []

    for k in unique_labels:
        if k >= 0:
            class_member_mask = (labels == k)

            xy = arg_X_two[class_member_mask] #& core_samples_mask]

            rect_x1 = min(xy[:, 1])
            rect_y1 = min(xy[:, 0])
            rect_x2 = max(xy[:, 1])
            rect_y2 = max(xy[:, 0])
            rect_n_points = len(xy)
            rect_ratio = rect_n_points / ((rect_x2 - rect_x1 + 1) * (rect_y2 - rect_y1 + 1))

            if k_means_flag:
                if rect_ratio <= 0.55 and rect_n_points > 100:
                    n_clusters_k_means = 2
                    max_clusters_n = 10
                    k_means = KMeans(init='k-means++', n_clusters=n_clusters_k_means, n_init=3)
                    start = time.time()
                    k_means.fit(xy)
                    stop = time.time()
                    sec = stop - start
                    print("KMeans time = %.4f sec" % sec, end=' ')
                    mid_inertia = k_means.inertia_ / len(k_means.labels_)
                    mid_inertia_new = mid_inertia
                    inertia_ratio = 0.5
                    k_means_new = k_means
                    while not(inertia_ratio > 0.8 or n_clusters_k_means > max_clusters_n):
                        mid_inertia = mid_inertia_new
                        k_means = k_means_new

                        n_clusters_k_means += 1
                        k_means_new = KMeans(init='k-means++', n_clusters=n_clusters_k_means, n_init=3)
                        start = time.time()
                        k_means_new.fit(xy)
                        stop = time.time()
                        sec = stop - start
                        print("KMeans time = %.4f sec" % sec, end=' ')
                        mid_inertia_new = k_means_new.inertia_ / len(k_means_new.labels_)
                        inertia_ratio = mid_inertia_new/mid_inertia
                     #after cycle we select previous k_means
                    n_clusters_k_means -= 1
                    print("NClusters = %d " % n_clusters_k_means, end=' ')
                    for k_detail in range(n_clusters_k_means):
                        detail_X_mask = k_means.labels_ == k_detail
                        X_KMeans = xy[detail_X_mask]
                        d_rect_x1 = min(X_KMeans[:, 1])
                        d_rect_y1 = min(X_KMeans[:, 0])
                        d_rect_x2 = max(X_KMeans[:, 1])
                        d_rect_y2 = max(X_KMeans[:, 0])
                        d_rect_n_points = len(X_KMeans)
                        d_rect_ratio = d_rect_n_points / ((d_rect_x2 - d_rect_x1 + 1) * (d_rect_y2 - d_rect_y1 + 1))
                        rects.append([d_rect_x1 * scale, d_rect_y1 * scale, d_rect_x2 * scale, d_rect_y2 * scale,
                                      d_rect_n_points * scale * scale, d_rect_ratio])
                else:
                    rects.append([rect_x1*scale, rect_y1*scale, rect_x2*scale, rect_y2*scale, rect_n_points*scale*scale, rect_ratio])
            else:
                rects.append([rect_x1 * scale, rect_y1 * scale, rect_x2 * scale, rect_y2 * scale, rect_n_points * scale * scale, rect_ratio])

    return rects

def generate_image_from_txt(out_shape, markup_path, image_filename, n_cars):
    """  generation of output array from txt file
    Example of file content:
    False
    205 144 40 40
    417 149 38 38
    38 130 48 48
    246 134 25 25"""

    out_arr = np.zeros(out_shape, dtype=K.floatx())

    with open(os.path.join(markup_path, image_filename + '.txt')) as f:
        content = f.read().splitlines()
    if (len(content) >= 2):
        counter = 0
        for content_item in content:
            if counter == 0:
                counter += 1
                continue
            n_cars += 1
            coords = content_item.split(' ')
            obj_x = int(coords[0])
            obj_y = int(coords[1])
            obj_w = int(coords[2])
            obj_h = int(coords[3])

            r_center_x = int(obj_x + obj_w / 2)
            r_center_y = int(obj_y + obj_h / 2)
            r_diameter = min(obj_w, obj_h)
            point_scale = 0.7
            point_size = int(r_diameter * point_scale)
            out_layer = 0
            i_min = r_center_y - int(point_size / 2)
            j_min = r_center_x - int(point_size / 2)
            i_max = i_min + point_size + 1
            j_max = j_min + point_size + 1
            if K.image_data_format() == 'channels_first':
                for i in range(i_min, i_max):
                    for j in range(j_min, j_max):
                        out_arr[out_layer][i][j] = 1
            else:
                for i in range(i_min, i_max):
                    for j in range(j_min, j_max):
                        out_arr[i][j][out_layer] = 1


    return out_arr, n_cars


def generate_image_from_xml(out_shape, markup_path, image_filename, obj_name, n_objs, obj_scale=1):
    """  generation of output array from pascal VOC xml file
    Example of file content:
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
    """

    out_arr = np.zeros(out_shape, dtype=K.floatx())

    res_fname, res_extension = os.path.splitext(image_filename)

    bounding_boxes = []  # массив координат
    tree = ElementTree.parse(os.path.join(markup_path, res_fname + '.xml'))
    root = tree.getroot()
    for object_tree in root.findall('object'):
        for bounding_box in object_tree.iter('bndbox'):
            xmin_o = float(bounding_box.find('xmin').text)
            ymin_o = float(bounding_box.find('ymin').text)
            xmax_o = float(bounding_box.find('xmax').text)
            ymax_o = float(bounding_box.find('ymax').text)

        class_name = object_tree.find('name').text
        bounding_box = [class_name, xmin_o, ymin_o, xmax_o, ymax_o]
        bounding_boxes.append(bounding_box)


    if (len(bounding_boxes) > 0):
        for content_item in bounding_boxes:
            if(content_item[0]==obj_name):
                n_objs += 1
                obj_x = content_item[1]
                obj_y = content_item[2]
                obj_w = content_item[3]-content_item[1]
                obj_h = content_item[4]-content_item[2]

                r_center_x = int(obj_x + obj_w / 2)
                r_center_y = int(obj_y + obj_h / 2)
                r_diameter = min(obj_w, obj_h)

                out_layer = 0
                i_min = r_center_y - int(obj_h*obj_scale / 2)
                j_min = r_center_x - int(obj_w*obj_scale / 2)
                i_max = i_min + int(obj_h*obj_scale)
                j_max = j_min + int(obj_w*obj_scale)
                if K.image_data_format() == 'channels_first':
                    for i in range(i_min, i_max):
                        for j in range(j_min, j_max):
                            out_arr[out_layer][i][j] = 1
                else:
                    for i in range(i_min, i_max):
                        for j in range(j_min, j_max):
                            out_arr[i][j][out_layer] = 1

    return out_arr, n_objs


def generate_image_fromimgfile(img_path, image_filename, img_height, img_width, gray_flag=False):
    """
    Loading of normalized image
    """

    img = load_img(os.path.join(img_path, image_filename), target_size=(img_height, img_width),grayscale=gray_flag)
    img_arr = img_to_array(img)  # this is a Numpy array with shape (3, img_height, img_width)
    img_arr = img_arr / 255

    return img_arr

# Main function
def prepare_detection_output_from_path(imp_path, markup_path, result_path,img_width,img_height,train_flag=False,save_y=False):
    """
    Main Function
    :param imp_path: folder with images for segmentation
    :param markup_path: folder with markup
    :param result_path: folder for segmentaion result
    :param img_width: width of each image
    :param img_height:  height of each image
    :param train_flag: mode of algorithm - True if you need to train Network and False if you need to test it
    :param save_y: save flag if source data have xml or txt markup in will be converted to masked images
    :return: none
    """

    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
    filenames = []
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(result_path+'_im', exist_ok=True)

    #os.makedirs(imp_path+'_intensive', exist_ok=True)

    #read filenames
    for filename in sorted(os.listdir(imp_path)):
        is_valid = False
        for extension in white_list_formats:
            if filename.lower().endswith('.' + extension):
                is_valid = True
                break
        if is_valid:
            filenames.append(filename)

    #read images
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_height, img_width)
        out_shape = (1, img_height, img_width)
    else:
        input_shape = (img_height, img_width, 3)
        out_shape = (img_height, img_width, 1)
    #n_images = len(filenames)

    """
    for numpy array size (float32 - 4 bytes)
    image size 480*270*4
    number of images = 10000

    518400 bytes - monochrome image
    1555200 bytes - color image

    input value = 15,552,000,000 bytes
    output value = 5,184,000,000 bytes"""
    if (train_flag):
        #in training mode
        n_images = len(filenames)
        n_cars = 0
        x = np.zeros((n_images,)+input_shape, dtype=K.floatx())
        y = np.zeros((n_images,) + out_shape, dtype=K.floatx())
        image_index = 0
        load_percentage = 0
        one_percent = n_images/100
        for image_filename in filenames:
            if (image_index < n_images):
                #generation of input array
                x[image_index] = generate_image_fromimgfile(imp_path, image_filename, img_height, img_width)

                #out_arr, n_cars = generate_image_from_txt(out_shape, markup_path, image_filename, n_cars)
                out_arr, n_cars = generate_image_from_xml(out_shape, markup_path, image_filename, 'Trafficlight', n_cars)
                #y[image_index] = out_arr #/ 150
                #out_arr = generate_image_fromimgfile(markup_path, image_filename, img_height, img_width,gray_flag=True)
                y[image_index] = out_arr
                if save_y:
                    img = array_to_img(y[image_index], K.image_data_format(), scale=True)
                    save_path = result_path+'_im'+ '\\'+image_filename
                    img.save(save_path)
                    print(str(image_index)+' '+str(n_cars)+' '+save_path)
                if (image_index >= one_percent * load_percentage):
                    print('images loading ' + str(load_percentage)+'%')
                    load_percentage += 10
                image_index += 1

    else:
        # in testing mode
        n_images = len(filenames)
        x = np.zeros((n_images,) + input_shape, dtype=K.floatx())
        image_index = 0
        for image_filename in filenames:
            if (image_index < n_images):
                # generation of input array
                x[image_index] = generate_image_fromimgfile(imp_path, image_filename, img_height, img_width)
                image_index += 1
    print(str(len(filenames))+' images are loaded')
	

    #creation of model
    epochs = 30
    batch_size = 1
    weights_path = 'tf_segmentation_30_1_fcn1.h5'

    #model - easy autoncoder - Fully convolutional network
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape,padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))

    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    """"""

    model.add(UpSampling2D(size=(2, 2)))
    model.add(ZeroPadding2D(padding=((0,0),(1,2)))) #Zero padding for fitting output layers shape
    model.add(Conv2D(32, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(1, (1, 1)))
    model.add(Activation('sigmoid'))

    if (train_flag):
        # in training mode
        tensorboard = TensorBoard(log_dir='./logs_full', histogram_freq=0,
                                  write_graph=True, write_images=False)

        adam = optimizers.Adam(lr=1e-5)
        #sgd = optimizers.SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=True)
        model.compile(loss=dice_coef_loss,
                      optimizer=adam,
                      metrics=[dice_coef])
        model.summary()
        start = time.time()
        hist = model.fit(x,y,epochs=epochs,batch_size=batch_size,validation_split=0.05, callbacks=[tensorboard])
        stop = time.time()
        sec = stop - start
        print("ConvNet is trained! Training time = %.4f sec" % sec, end=' ')
        print(hist.history)

        model.save_weights(weights_path)
        K.clear_session()
    else:
        # in testing mode
        model.load_weights(weights_path)
        adam = optimizers.Adam(lr=1e-5)
        # sgd = optimizers.SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=True)
        model.compile(loss=dice_coef_loss,
                      optimizer=adam,
                      metrics=[dice_coef])
        model.summary()

        start = time.time()
        y = model.predict(x, batch_size=batch_size)
        stop = time.time()
        sec = stop - start
        print("ConvNet is loaded! Predicting time = %.4f sec" % sec, end=' ')
        K.clear_session()
        # generation of output array
        ind = 0
        for y_img in y:

            y_show = y_img*255
            img = array_to_img(y_show, K.image_data_format(), scale=False)
            """
            #image clustering - for bounding box searching on segmentation result
            rects = cluster_image(y_show, 50, bottom_border = 0, scale=2,k_means_flag=False)  # 50 is 20% of max brightness 255
            drawable = ImageDraw.Draw(img)

            x_show = x[ind] * 255
            img_x = array_to_img(x_show, K.image_data_format(), scale=False)
            drawable_x = ImageDraw.Draw(img_x)

            for rect in rects:
                #rect_ratio
                #rect_ratio = rect[4]/((rect[2]-rect[0]+1)*(rect[3]-rect[1]+1))
                if(rect[4] >= 120): #minimum size in pixel is 200 and percent of fill more than 0.55
                    drawable.rectangle(((rect[0],rect[1]),(rect[2],rect[3])), outline="white")
                    drawable.text((rect[0],rect[1]-10),'car '+str(rect[4])+" {0:.3f}".format(rect[5]),fill="white")
                    drawable_x.rectangle(((rect[0], rect[1]), (rect[2], rect[3])), outline="white")
                    drawable_x.text((rect[0], rect[1] - 10), 'car ' + str(rect[4]) + " {0:.3f}".format(rect[5]),
                                  fill="white")
            del drawable
            del drawable_x
            img.save(os.path.join(result_path, 'ed_'+filenames[ind]))
            img_x.save(os.path.join(result_path+'_im', 'ed_' + filenames[ind]))
            """

            img.save(os.path.join(result_path, 'ed_' + filenames[ind]))

            ind += 1




if __name__ == '__main__':
    """
    imp_path = 'C:/Users/m232P/Desktop/Киляжев(Курсовой)/train-images'
    markup_path = 'C:/Users/m232P/Desktop/Киляжев(Курсовой)/train-markup'
    result_path = 'C:/Users/m232P/Desktop/Киляжев(Курсовой)/result_train'

    test_path = 'C:/Users/m232P/Desktop/Киляжев(Курсовой)/test-images'
    result_test_path = 'C:/Users/m232P/Desktop/Киляжев(Курсовой)/result_test'
    """

    imp_path = 'C:\\Users\\m232P\\Desktop\\s_traff_light\\s_traff_light\\500_img'
    markup_path = 'C:\\Users\\m232P\\Desktop\\s_traff_light\\s_traff_light\\500_xml'
    result_path = 'C:\\Users\\m232P\\Desktop\\s_traff_light\\s_traff_light\\500_img_train_mkp'

    test_path = 'C:\\Users\\m232P\\Desktop\\s_traff_light\\s_traff_light\\107_img'
    result_test_path = 'C:\\Users\\m232P\\Desktop\\s_traff_light\\s_traff_light\\result_107_img'

    #prepare_detection_output_from_path(imp_path, markup_path, result_path, 1241, 376, True)

    #prepare_detection_output_from_path(imp_path, markup_path, result_path, 620, 188, True)

    #prepare_detection_output_from_path(imp_path, markup_path, result_path, 480, 270, True)

    #prepare_detection_output_from_path(test_path, markup_path, result_test_path, 620, 188, False)

    #prepare_detection_output_from_path(imp_path, markup_path, result_test_path, 620, 188, False)

    prepare_detection_output_from_path(test_path, markup_path, result_test_path, 455, 256, train_flag=False,save_y=False)