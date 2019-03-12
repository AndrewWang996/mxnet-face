#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# this file is coming from openface: https://github.com/cmusatyalab/openface, with some changes
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import cv2
import dlib
import numpy as np
import os,errno
import random
import shutil
import tensorflow as tf

file_dir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(file_dir, '..', 'model')
dlib_model_dir = os.path.join(modelDir, 'dlib')

"""Module for dlib-based alignment."""
TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)    # Clip everything to [0,1]




def mkdirP(path):
    """
    Create a directory and don't error if the path already exists.

    If the directory already exists, don't do anything.

    :param path: The directory to create.
    :type path: str
    """
    assert path is not None

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Image:
    """Object containing image metadata."""

    def __init__(self, cls, name, path):
        """
        Instantiate an 'Image' object.

        :param cls: The image's class; the name of the person.
        :type cls: str
        :param name: The image's name.
        :type name: str
        :param path: Path to the image on disk.
        :type path: str
        """
        assert cls is not None
        assert name is not None
        assert path is not None

        self.cls = cls
        self.name = name
        self.path = path

    def getBGR(self):
        """
        Load the image from disk in BGR format.

        :return: BGR image. Shape: (height, width, 3)
        :rtype: numpy.ndarray
        """
        try:
            bgr = cv2.imread(self.path)
        except:
            bgr = None
        return bgr

    def getRGB(self):
        """
        Load the image from disk in RGB format.

        :return: RGB image. Shape: (height, width, 3)
        :rtype: numpy.ndarray
        """
        bgr = self.getBGR()
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = None
        return rgb

    def __repr__(self):
        """String representation for printing."""
        return "({}, {})".format(self.cls, self.name)


def iterImgs(directory):
    u"""
    Iterate through the images in a directory.

    The images should be organized in subdirectories
    named by the image's class (who the person is)::

       $ tree directory
       person-1
       ── image-1.jpg
       ├── image-2.png
       ...
       └── image-p.png

       ...

       person-m
       ├── image-1.png
       ├── image-2.jpg
       ...
       └── image-q.png


    :param directory: The directory to iterate through.
    :type directory: str
    :return: An iterator over Image objects.
    """
    assert directory is not None

    exts = [".jpg", ".png"]

    for subdir, dirs, files in os.walk(directory):
        for path in files:
            (imageClass, fName) = (os.path.basename(subdir), path)
            (imageName, ext) = os.path.splitext(fName)
            if ext in exts:
                yield Image(imageClass, imageName, os.path.join(subdir, fName))


class AlignDlib:
    """
    Use `dlib's landmark estimation <http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html>`_ to align faces.

    The alignment preprocess faces for input into a neural network.
    Faces are resized to the same size (such as 96x96) and transformed
    to make landmarks (such as the eyes and nose) appear at the same
    location on every image.

    Normalized landmarks:

    .. image:: ../images/dlib-landmark-mean.png
    """

    #: Landmark indices corresponding to the inner eyes and bottom lip.
    INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]

    #: Landmark indices corresponding to the inner eyes and nose.
    INNER_EYES_AND_NOSE = [39, 42, 33]    

    #: Landmark indices corresponding to the outer eyes and nose.
    OUTER_EYES_AND_NOSE = [36, 45, 33]

    #: Landmark indices corresponding to the outer eyes and bottom lip.
    OUTER_EYES_AND_BOTTOM_LIP = [36, 45, 57]

    #: Landmark indices corresponding to left eye
    LEFT_EYES = [36, 37, 38, 39, 40, 41]

    #: Landmark indices corresponding to right eye
    RIGHT_EYES = [42, 43, 44, 45, 46, 47]

    def __init__(self, facePredictor):
        """
        Instantiate an 'AlignDlib' object.
    
        :param facePredictor: The path to dlib's
        :type facePredictor: str
        """
        assert facePredictor is not None

        detector_path = os.path.join(dlib_model_dir, 'mmod_human_face_detector.dat')
        self.detector = dlib.cnn_face_detection_model_v1(detector_path)
        self.predictor = dlib.shape_predictor(facePredictor)

    def getAllFaceBoundingBoxes(self, rgbImg):
        """
        Find all face bounding boxes in an image.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :return: All face bounding boxes in an image.
        :rtype: dlib.rectangles
        """
        assert rgbImg is not None

        try:
            rectangles = self.detector(rgbImg, 1)
            if isinstance(rectangles, dlib.mmod_rectangles):
                return [mmod.rect for mmod in rectangles]
            return self.detector(rgbImg, 1)
        except Exception as e:
            print("Warning: {}".format(e))
            # In rare cases, exceptions are thrown.
            return []

    def getLargestFaceBoundingBox(self, rgbImg):
        """
        Find the largest face bounding box in an image.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :return: The largest face bounding box in an image, or None.
        :rtype: dlib.rectangle
        """
        assert rgbImg is not None

        faces = self.getAllFaceBoundingBoxes(rgbImg)
        if len(faces) > 0:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None

    def findLandmarks(self, rgbImg, bb):
        """
        Find the landmarks of a face.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param bb: Bounding box around the face to find landmarks for.
        :type bb: dlib.rectangle
        :return: Detected landmark locations.
        :rtype: list of (x,y) tuples
        """
        assert rgbImg is not None
        assert bb is not None

        points = self.predictor(rgbImg, bb)
        return list(map(lambda p: (p.x, p.y), points.parts()))

    def align(self, imgDim, rgbImg, bb=None, pad=None, ts=None,
              landmarks=None, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP,
              opencv_det=False, opencv_model="../model/opencv/cascade.xml",
              only_crop=False, face_img_ratio=0.8):
        r"""align(imgDim, rgbImg, bb=None, landmarks=None, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP)

        Transform and align a face in an image.

        :param imgDim: The edge length in pixels of the square the image is resized to.
        :type imgDim: int
        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param bb: Bounding box around the face to align. \
                   Defaults to the largest face.
        :type bb: dlib.rectangle
        :param pad: padding bb by left, top, right, bottom
        :type pad: list
        :param landmarks: Detected landmark locations. \
                          Landmarks found on `bb` if not provided.
        :type landmarks: list of (x,y) tuples
        :param landmarkIndices: The indices to transform to.
        :type landmarkIndices: list of ints
        :return: The aligned RGB image. Shape: (imgDim, imgDim, 3)
        :rtype: numpy.ndarray
        """
        assert imgDim is not None
        assert rgbImg is not None
        assert landmarkIndices is not None

        if landmarks is None:
            if bb is None:
                if opencv_det:
                    face_cascade = cv2.CascadeClassifier(opencv_model)
                    faces = face_cascade.detectMultiScale(rgbImg, 1.1, 2, minSize=(30, 30))
                    dlib_rects = []
                    for (x,y,w,h) in faces:
                        dlib_rects.append(dlib.rectangle(int(x), int(y), int(x+w), int(y+h)))
                        if len(faces) > 0:
                            bb = max(dlib_rects, key=lambda rect: rect.width() * rect.height())
                        else:
                            bb = None
                else:
                    bb = self.getLargestFaceBoundingBox(rgbImg)
                if bb is None:
                    return
                if pad is not None:

                    left = int(max(0, bb.left() - bb.width()*float(pad[0])))
                    top = int(max(0, bb.top() - bb.height()*float(pad[1])))
                    right = int(min(rgbImg.shape[1], bb.right() + bb.width()*float(pad[2])))
                    bottom = int(min(rgbImg.shape[0], bb.bottom()+bb.height()*float(pad[3])))
                    bb = dlib.rectangle(left, top, right, bottom)

            landmarks = self.findLandmarks(rgbImg, bb)

        npLandmarks = np.float32(landmarks)
        npLandmarkIndices = np.array(landmarkIndices)

        landmark_locs = MINMAX_TEMPLATE[npLandmarkIndices]
        landmark_locs = (landmark_locs - 0.5) * face_img_ratio + 0.5
        dstLandmarks = imgDim * landmark_locs
        if ts is not None:
            # reserve more area of forehead on a face
            dstLandmarks[(0,1),1] = dstLandmarks[(0,1),1] + imgDim * float(ts)
            dstLandmarks[2,1] = dstLandmarks[2,1] + imgDim * float(ts) / 2
        if not only_crop:
            H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],dstLandmarks)
            return cv2.warpAffine(rgbImg, H, (imgDim, imgDim), borderMode=cv2.BORDER_TRANSPARENT)
        else:
            rgbImg = rgbImg[top:bottom, left:right] # crop is rgbImg[y: y + h, x: x + w]
            return cv2.imresize(rgbImg, (imgDim, imgDim))

 
    def align_frontalized_img(self, imgDim, rgbImg, landmarks, face_img_ratio=0.75):
        r"""
        Transform and align a frontal face in an image.
        IMPORTANT ASSUMPTION: Face is frontalized

        :param imgDim: The edge length in pixels of the square the image is resized to.
        :type imgDim: int
        :param rgbImg: RGB tensor to process. Shape: (160, 160, 3)
        :type rgbImg: numpy.ndarray
        :param landmarks: Detected landmark locations. \
                          Landmarks found on `bb` if not provided.
        :type landmarks: list of (x,y) tuples
        :return: The aligned RGB image. Shape: (imgDim, imgDim, 3)
        :rtype: numpy.ndarray
        """
        assert imgDim is not None
        assert rgbImg is not None
        
        # Pad input image with 160 border (adds 320 to height + width)
        paddings = tf.constant([[imgDim, imgDim], [imgDim, imgDim], [0,0]])
        rgbImg = tf.pad(rgbImg, paddings)
    
        # Update landmark locations correspondingly
        landmarks = imgDim + landmarks        

        # Calculate center of mass for left and right eyes
        eye_L = tf.reduce_mean(
            tf.gather(landmarks, AlignDlib.LEFT_EYES),
            axis=0
        )
        eye_R = tf.reduce_mean(
            tf.gather(landmarks, AlignDlib.RIGHT_EYES),
            axis=0
        )
        
        # Calculate face center & height based on landmark locations
        landmarks_max = tf.reduce_max(landmarks, axis=0)
        landmarks_min = tf.reduce_min(landmarks, axis=0)
        face_height = landmarks_max[1] - landmarks_min[1] 
        
        # Translate center of eyes to center of image
        eye_center = (eye_L + eye_R) / 2.0 
        img_center = (tf.cast(tf.shape(rgbImg)[:2], tf.float32) - 1.) / 2.0 
        rgbImg = tf.contrib.image.translate(rgbImg, img_center - eye_center)

        # Rotate so that left and right eye have same height
        theta = tf.math.atan( - (eye_R[1] - eye_L[1]) / (eye_R[0] - eye_L[0]) )
        rgbImg = tf.contrib.image.rotate(rgbImg, -theta, interpolation='BILINEAR')

        # Scale so that distance between top & bottom-most landmarks
        # are as desired
        minmax_template = (tf.constant(MINMAX_TEMPLATE) - 0.5) * face_img_ratio + 0.5
        desired_face_height = imgDim * (
            tf.reduce_max(minmax_template[:,1]) - 
            tf.reduce_min(minmax_template[:,1])
        )
        scale = face_img_ratio * desired_face_height / face_height
        rgbImg = tf.image.resize_images(
            rgbImg,
            tf.cast(scale * tf.cast(rgbImg.shape[:2], tf.float32), tf.int32)
        )
        new_img_center = tf.dtypes.cast(scale * img_center, tf.int32) 

        # Crop to the relevant part of the image (imgDim x imgDim x 3)
        rgbImg = tf.slice(
            rgbImg,
            begin = tf.concat([new_img_center - imgDim // 2, [0]], 0),
            size = [imgDim, imgDim, 3]
        )
        return rgbImg

def write(vals, fName):
    if os.path.isfile(fName):
        print("{} exists. Backing up.".format(fName))
        os.rename(fName, "{}.bak".format(fName))
    with open(fName, 'w') as f:
        for p in vals:
            f.write(",".join(str(x) for x in p))
            f.write("\n")


def computeMeanMain(args):
    align = AlignDlib(args.dlibFacePredictor)

    imgs = list(iterImgs(args.inputDir))
    if args.numImages > 0:
        imgs = random.sample(imgs, args.numImages)

    facePoints = []
    for img in imgs:
        rgb = img.getRGB()
        bb = align.getLargestFaceBoundingBox(rgb)
        alignedPoints = align.align(rgb, bb)
        if alignedPoints:
            facePoints.append(alignedPoints)

    facePointsNp = np.array(facePoints)
    mean = np.mean(facePointsNp, axis=0)
    std = np.std(facePointsNp, axis=0)

    write(mean, "{}/mean.csv".format(args.modelDir))
    write(std, "{}/std.csv".format(args.modelDir))

    # Only import in this mode.
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(mean[:, 0], -mean[:, 1], color='k')
    ax.axis('equal')
    for i, p in enumerate(mean):
        ax.annotate(str(i), (p[0] + 0.005, -p[1] + 0.005), fontsize=8)
    plt.savefig("{}/mean.png".format(args.modelDir))


def alignMain(args):
    mkdirP(args.outputDir)

    imgs = list(iterImgs(args.inputDir))

    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)

    if args.landmarks == 'outerEyesAndNose':
        landmarkIndices = AlignDlib.OUTER_EYES_AND_NOSE
    elif args.landmarks == 'innerEyesAndBottomLip':
        landmarkIndices = AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    else:
        raise Exception("Landmarks unrecognized: {}".format(args.landmarks))

    align = AlignDlib(args.dlibFacePredictor)

    nFallbacks = 0
    for imgObject in imgs:
        print("=== {} ===".format(imgObject.path))
        outDir = os.path.join(args.outputDir, imgObject.cls)
        mkdirP(outDir)
        outputPrefix = os.path.join(outDir, imgObject.name)
        imgName = outputPrefix + "." + args.ext

        if os.path.isfile(imgName):
            if args.verbose:
                print("  + Already found, skipping.")
        else:
            rgb = imgObject.getRGB()
            if rgb is None:
                if args.verbose:
                    print("  + Unable to load.")
                outRgb = None
            else:
                outRgb = align.align(args.size, rgb, pad=args.pad, ts=args.ts,
                                     landmarkIndices=landmarkIndices, opencv_det=args.opencv_det,
                                     opencv_model=args.opencv_model, only_crop=args.only_crop)
                if outRgb is None and args.verbose:
                    print("  + Unable to align.")

            if args.fallbackLfw and outRgb is None:
                nFallbacks += 1
                deepFunneled = "{}/{}.jpg".format(os.path.join(args.fallbackLfw,
                                                               imgObject.cls),
                                                  imgObject.name)
                shutil.copy(deepFunneled, "{}/{}.jpg".format(os.path.join(args.outputDir,
                                                                          imgObject.cls),
                                                             imgObject.name))

            if outRgb is not None:
                if args.verbose:
                    print("  + Writing aligned file to disk.")
                outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(imgName, outBgr)

    if args.fallbackLfw:
        print('nFallbacks:', nFallbacks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('inputDir', type=str, help="Input image directory.")
    parser.add_argument('--opencv-det', action='store_true', default=False,
                        help='True means using opencv model for face detection(because sometimes dlib'
                             'face detection will failed')
    parser.add_argument('--opencv-model', type=str, default='../model/opencv/cascade.xml',
                        help="Path to dlib's face predictor.")
    parser.add_argument('--only-crop', action='store_true', default=False,
                        help='True : means we only use face detection and crop the face area\n'
                             'False : both face detection and then do face alignment')
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlib_model_dir, "shape_predictor_68_face_landmarks.dat"))

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    computeMeanParser = subparsers.add_parser(
        'computeMean', help='Compute the image mean of a directory of images.')
    computeMeanParser.add_argument('--numImages', type=int, help="The number of images. '0' for all images.",
                                   default=0)  # <= 0 ===> all imgs
    alignmentParser = subparsers.add_parser(
        'align', help='Align a directory of images.')
    alignmentParser.add_argument('landmarks', type=str,
                                 choices=['outerEyesAndNose', 'innerEyesAndBottomLip'],
                                 help='The landmarks to align to.')
    alignmentParser.add_argument(
        'outputDir', type=str, help="Output directory of aligned images.")
    alignmentParser.add_argument('--pad', type=float, nargs='+', help="pad (left,top,right,bottom) for face detection region")
    alignmentParser.add_argument('--ts', type=float, help="translation(,ts) the proportion position of eyes downward so that..."
                                                        " we can reserve more area of forehead",
                                 default='0')
    alignmentParser.add_argument('--size', type=int, help="Default image size.",
                                 default=96)
    alignmentParser.add_argument('--ext', type=str, help="Default image extension.",
                                 default='jpg')
    alignmentParser.add_argument('--fallbackLfw', type=str,
                                 help="If alignment doesn't work, fallback to copying the deep funneled version from this directory..")
    alignmentParser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    if args.mode == 'computeMean':
        computeMeanMain(args)
    else:
        alignMain(args)
