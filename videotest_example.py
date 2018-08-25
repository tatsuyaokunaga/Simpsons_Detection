import keras
import pickle
from videotest import VideoTest

import sys
sys.path.append("..")
from ssd import SSD300 as SSD

input_shape = (300,300,3)

# Change this if you run with other classes than VOC
# class_names = ["background", "aeroplane", "bicycle", "bird",
#  "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
#  "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", 
#  "sheep", "sofa", "train", "tvmonitor"];

class_names = ['back_ground','abraham_grampa_simpson', 'apu_nahasapeemapetilon',
                        'bart_simpson','charles_montgomery_burns', 'chief_wiggum',
                        'comic_book_guy','edna_krabappel','homer_simpson',
                        'kent_brockman', 'krusty_the_clown', 'lisa_simpson',
                        'marge_simpson', 'milhouse_van_houten','moe_szyslak',
                        'ned_flanders', 'nelson_muntz', 'principal_skinner', 'sideshow_bob']


NUM_CLASSES = len(class_names)

model = SSD(input_shape, num_classes=NUM_CLASSES)

# Change this path if you want to use your own trained weights
model.load_weights('./weights.07-2.08.hdf5') 
        
vid_test = VideoTest(class_names, model, input_shape)

# To test on webcam 0, remove the parameter (or change it to another number
# to test on that webcam)
vid_test.run('./001.mp4')
