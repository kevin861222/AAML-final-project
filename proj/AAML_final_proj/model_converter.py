import glob
import os
import subprocess
import re
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', nargs='?', default='./pretrainedResnet_quant.tflite')
    args = parser.parse_args()
    tflite_path = str(args.model_path)

    output_path = 'src/tiny/v0.1/training/image_classification/trained_models/pretrainedResnet_quant.cc'
    xxd_ret = subprocess.check_output(f"xxd -i {tflite_path}".split(' ')).decode('UTF-8')
    tflm_format = re.sub('unsigned char .*_tflite\\[\\] = {', 'const unsigned char pretrainedResnet_quant[] = {', xxd_ret)
    tflm_format = re.sub('unsigned int .*_len', 'unsigned int pretrainedResnet_quant_len', tflm_format)
    tflm_format = ( '#include "tiny/v0.1/training/image_classification/trained_models/pretrainedResnet_quant.h"\n'
                   ) + tflm_format
    with open(output_path, 'w') as f:
        f.write(tflm_format)
