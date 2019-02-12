#!/usr/bin/env python

import math
import os
import argparse

import csv
from cv_bridge import CvBridge

import cv2
import numpy as np
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _save_events(events,
                 image_times,
                 event_count_images,
                 event_time_images,
                 event_image_times,
                 rows,
                 cols,
                 max_aug,
                 n_skip,
                 event_image_iter, 
                 prefix, 
                 cam,
                 tf_writer,
                 t_start_ros):
    event_iter = 0
    cutoff_event_iter = 0
    image_iter = 0
    curr_image_time = (image_times[image_iter] - t_start_ros) / 1000000.
    
    event_count_image = np.zeros((rows, cols, 2), dtype=np.uint16)
    event_time_image = np.zeros((rows, cols, 2), dtype=np.float32)

    while image_iter < len(image_times) and events[-1][2] > curr_image_time:
      x = events[event_iter][0]
      y = events[event_iter][1]
      t = events[event_iter][2]

      if events[event_iter][3] > 0:
          event_count_image[y, x, 0] += 1
          event_time_image[y, x, 0] = t
      else:
          event_count_image[y, x, 1] += 1
          event_time_image[y, x, 1] = t

      event_iter += 1
      if t > curr_image_time:
          event_count_images.append(event_count_image)
          event_count_image = np.zeros((rows, cols, 2), dtype=np.uint16)
          event_time_images.append(event_time_image)
          event_time_image = np.zeros((rows, cols, 2), dtype=np.float32)
          cutoff_event_iter = event_iter
          event_image_times.append(image_times[image_iter] / 1000000.)
          image_iter += n_skip
          if (image_iter < len(image_times)):
              curr_image_time = (image_times[image_iter] - t_start_ros) / 1000000.

    del image_times[:image_iter]
    del events[:cutoff_event_iter]
    
    if len(event_count_images) >= max_aug:
        n_to_save = len(event_count_images) - max_aug + 1
        for i in range(n_to_save):
            image_times_out = np.array(event_image_times[i:i+max_aug+1])
            image_times_out = image_times_out.astype(np.float64)
            event_time_images_np = np.array(event_time_images[i:i+max_aug], dtype=np.float32)
            event_time_images_np -= image_times_out[0] - t_start_ros / 1000000.
            event_time_images_np = np.clip(event_time_images_np, a_min=0, a_max=None)
            image_shape = np.array(event_time_images_np.shape, dtype=np.uint16)
            
            event_image_tf = tf.train.Example(features=tf.train.Features(feature={
                'image_iter': _int64_feature(event_image_iter),
                'shape': _bytes_feature(
                    image_shape.tobytes()),
                'event_count_images': _bytes_feature(
                    np.array(event_count_images[i:i+max_aug], dtype=np.uint16).tobytes()),
                'event_time_images': _bytes_feature(event_time_images_np.tobytes()),
                'image_times': _bytes_feature(
                    image_times_out.tobytes()),
                'prefix': _bytes_feature(
                    prefix.encode()),
                'cam': _bytes_feature(
                    cam.encode())
            }))

            tf_writer.write(event_image_tf.SerializeToString())
            event_image_iter += n_skip

        del event_count_images[:n_to_save]
        del event_time_images[:n_to_save]
        del event_image_times[:n_to_save]
    return event_image_iter

def main():
    parser = argparse.ArgumentParser(
        description=("Extracts grayscale and event images from a ROS bag and "
                     "saves them as TFRecords for training in TensorFlow."))
    parser.add_argument("--max_aug", dest="max_aug",
                        help="Maximum number of images to combine for augmentation.",
                        type=int,
                        default = 6)
    parser.add_argument("--n_skip", dest="n_skip",
                        help="Maximum number of images to combine for augmentation.",
                        type=int,
                        default = 1)

    args = parser.parse_args()
    bridge = CvBridge()

    n_msgs = 0
    left_event_image_iter = 0
    right_event_image_iter = 0
    left_image_iter = 0
    right_image_iter = 0
    first_left_image_time = -1
    first_right_image_time = -1

    left_events = []
    right_events = []
    left_images = []
    right_images = []
    left_image_times = []
    right_image_times = []
    left_event_count_images = []
    left_event_time_images = []
    left_event_image_times = []
    
    right_event_count_images = []
    right_event_time_images = []
    right_event_image_times = []
    
    cols = 240
    rows = 180
    image_separation = 5 # ms
    prev_image_time = 0

    flip_vertically = False
    flip_horizontally = False

    ts_scaling = 1.e6

    path_from = 'boxes_6dof_25.csv'
    path_to = 'mvsec_data/' + path_from.split('.')[0] + '/'
    if os.path.isdir(path_to): os.system('rm -rf ' + path_to)
    os.system('mkdir ' + path_to)

    left_tf_writer = tf.python_io.TFRecordWriter(
        os.path.join(path_to, "left_event_images.tfrecord"))
    right_tf_writer = tf.python_io.TFRecordWriter(
        os.path.join(path_to, "right_event_images.tfrecord"))

    idx = 0
    start = True
    f = open(path_from, 'r')
    reader = csv.reader(f)
    print("Processing event file")
    for row in reader:

      row[0] = float(row[0])
      row[0] *= ts_scaling

      row[1] = 0 + int(row[1]) / 1
      row[2] = 0 + int(row[2]) / 1

      # read time
      if start == True: 
        t_start = float(row[0]) / 1000000. # s
        t_start_ros = float(row[0])
        prev_image_time = t_start
        first_left_image_time = float(row[0])
        left_event_image_times.append(float(row[0]) / 1000000.)
        prev_image_time = float(row[0])
        start = False

      # recent event
      ts = float(row[0])
      time = ts
      event = [int(row[1]), int(row[2]), (ts - t_start_ros) / 1000000., (float(row[3]) - 0.5) * 2]

      # flip data
      if flip_vertically:
        event[1] = event[1] + 2 * (rows / 2 - event[1]) - 1
      if flip_horizontally:
        event[0] = event[0] + 2 * (cols / 2 - event[0]) - 1

      # image statistics (images have to be separated)
      if (time - prev_image_time) > image_separation * 1000:
        cv2.imwrite(os.path.join(path_to, "left_image{:05d}.png".format(left_image_iter)), np.zeros((rows, cols)))
        if left_image_iter > 0: left_image_times.append(time)          
        left_image_iter += 1
        prev_image_time = time

      # process events
      if first_left_image_time != -1  and ts > first_left_image_time: 
        left_events.append(event)

      # store events
      if len(left_image_times) >= args.max_aug and left_events[-1][2] > (left_image_times[args.max_aug-1]-t_start_ros) / 1000000.:
        print idx
        print '----------------------------'
        idx += 1
        left_event_image_iter = _save_events(left_events,
                                             left_image_times,
                                             left_event_count_images,
                                             left_event_time_images,
                                             left_event_image_times,
                                             rows,
                                             cols,
                                             args.max_aug,
                                             args.n_skip,
                                             left_event_image_iter, 
                                             path_from.split('.')[0], 
                                             'left',
                                             left_tf_writer,
                                             t_start_ros)

    left_tf_writer.close()    
    image_counter_file = open(os.path.join(path_to, "n_images.txt") , 'w')
    image_counter_file.write("{} {}".format(left_event_image_iter, right_event_image_iter))
    image_counter_file.close()         
    
if __name__ == "__main__":
    main()
