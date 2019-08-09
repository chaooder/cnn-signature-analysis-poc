import os
import tensorflow as tf
import sys

from cnn_signature_analysis_poc.topology import cnn_model_fn
from cnn_signature_analysis_poc.preprocessing import process_image

modelFolder = 'model'
predictFolder = 'predict_data'

def main(unused_argv):
	"""
    Function for reading image, passing it through the CNN model, and providing predictions 
	"""
	signClassifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=modelFolder)
	for img_file in os.listdir(predictFolder):
		print('Predicting image file: {}'.format(img_file))
		predictData = process_image(os.path.join(predictFolder,img_file))
		predictInputFn = tf.estimator.inputs.numpy_input_fn(
			x = predictData,
			shuffle = False
		)
		results = signClassifier.predict(
			input_fn = predictInputFn
		)
		
		for result in results:
			predictedClass = result['classes']
			if(predictedClass==0):
				print('Prediction: Real signature of Client 002.')
				print('Probability: '+str(result['probabilities'][predictedClass]*100)+'%')
			elif(predictedClass==1):
				print('Prediction: Forged signature of Client 002.')
				print('Probability: '+str(result['probabilities'][predictedClass]*100)+'%')
			elif(predictedClass==2):
				print('Prediction: Real signature of Client 046.')
				print('Probability: '+str(result['probabilities'][predictedClass]*100)+'%')
			elif(predictedClass==3):
				print('Prediction: Forged signature of Client 046.')
				print('Probability: '+str(result['probabilities'][predictedClass]*100)+'%')

if __name__ == "__main__":
	tf.app.run()