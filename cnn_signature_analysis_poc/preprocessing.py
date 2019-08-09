import cv2

def process_image(img_file):
	"""
	Processes image file, resizing it, and normalizing it (to between 0 & 1)

	Parameters
	----------
	img_file: str
	Path of source img_file

	Returns
	-------
	np.array of data
	"""
	img = cv2.imread(img_file,0)
	norm_img = (255-cv2.resize(img, (40, 30), interpolation=cv2.INTER_AREA))/255
	return norm_img