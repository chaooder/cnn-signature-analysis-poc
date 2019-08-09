import os
import numpy as np
import tensorflow as tf
import cv2

from cnn_signature_analysis_poc.topology import cnn_model_fn
from cnn_signature_analysis_poc.preprocessing import process_image

trainFolder = 'dev_data/train'
testFolder = 'dev_data/test'
modelFolder = 'model'

customers = ['002', '046']

N_STEPS = 1000

tf.logging.set_verbosity(tf.logging.INFO)

def getData(folder):
	"""
	Retrieve image data from specified folder.
	Reads data from an input source folder, parse it for processing, and returns data.

	Parameters
	----------
	folder: str
	Source folder with image data

	Returns
	-------
	dict
	    returns dictionary with: 
	    {
	        'data': np.array,
            'label': np.array
        }
	"""
	data = []
	label = []
	for dirname in os.listdir(folder):
		subDir = folder+'/'+dirname
		for filename in os.listdir(subDir):
			labelRow = [0]*2*len(customers)
			custId = filename[(filename.find('.')-3):filename.find('.')]
			imgData = process_image(os.path.join(subDir, filename))
			data.append(imgData)
			if(dirname == 'real'):
				label.append(2*customers.index(custId))
			elif(dirname == 'forge'):
				label.append(2*customers.index(custId)+1)
	return {'data': np.array(data), 'label': np.array(label)}
	
def main(unused_argv):
	#Load Train and Test Data
	trainSet = getData(trainFolder)
	trainData = trainSet['data']
	trainLabel = trainSet['label']
	
	testSet = getData(testFolder)
	testData = testSet['data']
	testLabel = testSet['label']
	
	signClassifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn, 
		model_dir=modelFolder,
	)
	
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
	
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x = trainData,
		y = trainLabel,
		batch_size = 10,
		num_epochs = None,
		shuffle = True
	)
	signClassifier.train(
		input_fn = train_input_fn,
		steps = N_STEPS,
		hooks = [logging_hook]
	)
	
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x = testData,
		y = testLabel,
		num_epochs = 1,
		shuffle = False
	)
	eval_results = signClassifier.evaluate(input_fn=eval_input_fn)
	
	print(eval_results)

if __name__ == "__main__":
	tf.app.run()