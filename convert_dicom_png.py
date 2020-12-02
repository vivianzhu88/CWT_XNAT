import pydicom
import numpy as np

WINDOW_CENTER	= 50
WINDOW_WIDTH	= 100

import xnat
import os

def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

  
def get_gray_img(dicom, window_center, window_width):
	try:
		img = dicom.pixel_array
		slope = get_first_of_dicom_field_as_int(dicom.RescaleSlope)
		intercept = get_first_of_dicom_field_as_int(dicom.RescaleIntercept)
		img = (img*slope +intercept)
		img_min = window_center - window_width//2
		img_max = window_center + window_width//2
		img[img<img_min] = img_min
		img[img>img_max] = img_max

		img = (img - img_min) / (img_max - img_min)
		return img
	except:
		return None


def get_MRI_gray_img(dicom):
	try:
		img = dicom.pixel_array
		img_min = img.min()
		img_max = img.max()
		img = (img - img_min) / (img_max - img_min)
		return img
	except:
		return None

def process_dicom(dicom_path, if_xnat=False, rgb=True):
	if if_xnat == True:
		d = dicom_path.split('/')
		session = xnat.connect('http://rufus.stanford.edu', user='admin', password='admin') #make XNAT connection
		scan = session.projects[d[3]].subjects[d[5]].experiments[d[7]].scans[d[9]]
		dicom = scan.read_dicom(read_pixel_data=True)
		session.disconnect()
		print(dicom)
	else:
		dicom = pydicom.read_file(dicom_path)
		

	brain_img = get_gray_img(dicom, 40, 80)
	subdural_img = get_gray_img(dicom, 80, 200)
	bone_img = get_gray_img(dicom, 600, 2000)

	if not brain_img is None and not subdural_img is None and not bone_img is None:
		bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
		bsb_img[:, :, 0] = brain_img
		bsb_img[:, :, 1] = subdural_img
		bsb_img[:, :, 2] = bone_img
		return bsb_img
	else:
		## we may deal with not CT images
		MR_img = get_MRI_gray_img(dicom)
		if not MR_img is None:
			bsb_img = np.zeros((MR_img.shape[0], MR_img.shape[1], 3))
			bsb_img[:, :, 0] = MR_img
			bsb_img[:, :, 1] = MR_img
			bsb_img[:, :, 2] = MR_img
			return bsb_img
	print('The provided dicom is not the specified CT or MRI images')
		
	return None



