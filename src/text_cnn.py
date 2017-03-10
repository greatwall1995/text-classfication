import mxnet as mx
import numpy as np
import time
import data_helpers

input_size = 2048

def main():
	#data_helpers.split()
	data_helpers.preprocess(input_size)
	data_helpers.statistic()
	#load_data()

if __name__ == '__main__':
	main()