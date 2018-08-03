'''
File: xml_to_csv.py
Project: model-part
File Created: Thursday, 5th July 2018 2:54:38 pm
Author: https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py
-----
Last Modified: Thursday, 5th July 2018 3:37:38 pm
Modified By: Sujan Poudel 
'''

import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd

TRAIN_IMAGES_DIRECTORY = "./images/train"
TEST_IMAGES_DIRECTORY = "./images/test"

OUTPUT_DIRECTORY ="./converted-data"

def xml_to_csv(image_directory): 
    xml_list = []
    directories = [x[0] for x in os.walk(image_directory)] # directories inside image_directory including itself
    for path in directories: # for each directory
        for xml_file in glob.glob(path + '/*.xml'): 
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                        int(root.find('size')[0].text),
                        int(root.find('size')[1].text),
                        member[0].text,
                        int(member[4][0].text),
                        int(member[4][1].text),
                        int(member[4][2].text),
                        int(member[4][3].text)
                        )
                xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    xml_df = xml_to_csv(TRAIN_IMAGES_DIRECTORY)
    xml_df.to_csv('{}/train_data.csv'.format(OUTPUT_DIRECTORY), index=None)
    print('Successfully converted train image xmls to csv.')

    xml_df = xml_to_csv(TEST_IMAGES_DIRECTORY)
    xml_df.to_csv('{}/test_data.csv'.format(OUTPUT_DIRECTORY), index=None)
    print('Successfully converted test image xmls to csv.')

main()