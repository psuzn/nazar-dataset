"""
File: generate_tfrecord.py
Project: model-part
File Created: Thursday, 5th July 2018 3:44:27 pm
Author: https://github.com/datitran/raccoon_dataset/blob/master/test_generate_tfrecord.py
-----
Last Modified: Thursday, 5th July 2018 4:21:48 pm
Modified By: Sujan Poudel 


Usage:
  # From tensorflow/models/
  # Create data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv --images_path=path/to/images/folder  --output_path=train.record

"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('images_path','' ,'Path to the folder containing images,files will be searched in subdirectories(only one step down)')
FLAGS = flags.FLAGS

possible_paths = [x[0] for x in os.walk(FLAGS.images_path)]

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'Electrolytic Capacitor' or row_label == "Electrolytic-Capacitor" or row_label == "Electrolytic-capacitor":
            return 1
    elif row_label =='LED':
        return 2
    elif row_label =='ceramic capacitor' or row_label == "ceramic capacitor ":
        return 3
    elif row_label =='diode':
        return 4
    elif row_label =='resistor':
        return 5
    elif row_label =='transistor':
        return 6
    else:
        print("label=:{}:".format(row_label))
        return None



def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group):
    for path in possible_paths:
        filePath = os.path.join(os.getcwd(),path,"{}".format(group.filename))
        if(os.path.isfile(filePath)):
            break
    if(os.path.isfile(filePath)):
        print("file exist at {}".format(filePath))
    else:
        print("file not found at at {}".format(filePath))
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
