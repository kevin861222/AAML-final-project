import argparse
import csv
import os
import re
import serial
from tqdm import tqdm
import time

READY_MSG='m-ready\r\n'
SEND_BYTES=64

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", nargs='?', default='1,32,32,3', type=str, 
                            help='Input NHWC dimension, e.g., --input_dim 1,32,32,3')
    parser.add_argument("--port", nargs='?', default='/dev/ttyUSB1', type=str, 
                            help='Device port, e.g, --port /dev/ttyUSB1.')
    parser.add_argument("-p", nargs='?', dest='port', type=str,
                            help='Device port, e.g, -p /dev/ttyUSB1.')
    return parser.parse_args()

class PerfFormat:
    def __init__(self, filename, output_len, ground_truth) -> None:
        self.filename = filename
        self.output_len = int(output_len)
        self.ground_truth = int(ground_truth)
        
if __name__ == '__main__':
    args = parse_arg()
    port = str(args.port)
    input_dim = [int(dim) for dim in args.input_dim.split(',')]
    input_size = input_dim[0]*input_dim[1]*input_dim[2]*input_dim[3]

    with open('y_labels.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        testcases = [PerfFormat(row[0],row[1],row[2]) for row in reader]

    com = serial.Serial(
        port = port,
        baudrate=1843200
    )
    if com.is_open:
        com.write("xxxxxxxx30%".encode())
        output = com.read_until(READY_MSG.encode()).decode()
        time.sleep(0.5)
        output = com.read_all().decode()
        com.write("name%".encode())
        output = com.read_until(READY_MSG.encode()).decode()
        student_ID = re.search("m-name-dut-\\[(NYCU-CAS-LAB)\\]", output).group(1)
    else:
        raise(BaseException('Opening serial port failed!'))

    result = {'correct_cnt':0, 'latency':[]}
    for testcase in tqdm(testcases):
        with open(os.path.join('perf_samples', testcase.filename), 'rb') as test_input:
            com.read_all()
            com.write(f"db load {32*32*3}%".encode())
            com.read_until(READY_MSG.encode()).decode()
            data = test_input.read(SEND_BYTES//2)
            while len(data) > 0:
                com.write(f"db {data.hex()}%".encode())
                s = com.read_until(READY_MSG.encode()).decode()
                data = test_input.read(SEND_BYTES//2)
            com.write(f"infer 1 0%".encode())
            msg = com.read_until(READY_MSG.encode()).decode()
            m = [ int(x) for x in re.findall('m-lap-us-([0-9]*)', msg)]
            result['latency'].append(m[1]-m[0])
            m = re.search('m-results-\\[((?:-?[0-9]+,?)+)\\]', msg).group(1)
            m = [int(x) for x in m.split(',')]
            arg_max = [i for i, x in enumerate(m) if x == max(m)][0]
            result['correct_cnt'] += arg_max == testcase.ground_truth
            print(result)

    acc = result['correct_cnt'] / len(testcases)
    lat = sum(result['latency'])/len(testcases)

    print(f"Accuracy: {acc:.3f} %")
    print(f"Latency: {lat} us")