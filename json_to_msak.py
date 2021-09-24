from skimage import draw
from skimage import io
import numpy as np
import json
import logging
import os
import sys
import cv2
import glob as gb
import argparse

#enable info logging.
logging.getLogger().setLevel(logging.INFO)

def poly2mask(blobs,name_image, path, cat_damge,shape, label,structuredArr=None):
    """
        creates folder for masks and save masks from polygon !!!
        :blobs: list of polygons
        :name_image: name of image
        :path_to_masks_folder: the path wher to save masks 
        :shape: the shape of image
        :label: the label of de masks
    """
    name = name_image.split('.')
    directory = name[0]
    path_dir = os.path.join(path, directory)
    img_shape = (shape[1],shape[0])
    
    for value in  blobs:
        # print(np.array(value[1]).T)
        mask = draw.polygon2mask(img_shape, np.array(value[0]))
           
    if (not os.path.isdir(path_dir)):
        os.mkdir(path_dir)
    name_mask =  label+'.png'
    io.imsave(path_dir+"/"+label+".png", mask.T)

    dtype = [('name_mask', (np.str_, 30)), ('type', (np.str_, 30))]
    
    if structuredArr!=None :
        structuredArr=np.append(structuredArr,np.array([(name_mask,cat_damge)], dtype=dtype))
    else :
        structuredArr=(np.array([(name_mask,cat_damge)], dtype=dtype))
    return structuredArr,path_dir


def convert_data_to_masks(path_to_data_annotation_json, path_to_original_images_folder, path_to_masks_folder):
    """
        convert json data made by LABELME to masks (numpy)

        :path_to_data_annotation_json: the path of json data
        :path_to_original_images_folder: the path of original images
        :path_to_masks_folder : the path of masks
    """
    # make sure everything is setup.
    if (not os.path.isdir(path_to_original_images_folder)):
        logging.exception(
            "Please specify a valid directory path to download images, " + path_to_original_images_folder + " doesn't exist")
        return
    if (not os.path.isdir(path_to_masks_folder)):
        logging.exception(
            "Please specify a valid directory path to write mask files, " + path_to_masks_folder + " doesn't exist")
        return
    if (not os.path.exists(path_to_data_annotation_json)):
        logging.exception(
            "Please specify a valid path to dataturks JSON output file, " + path_to_data_annotation_json + " doesn't exist")
        return

    with open(path_to_data_annotation_json) as f:
        train_data = json.load(f)
        annotations = train_data['shapes']
        name_image = train_data['imagePath']
        labels = train_data['labels']
        image_Path = path_to_original_images_folder+"/"+str(name_image)
        print(image_Path)
        img = cv2.imread(image_Path,cv2.IMREAD_GRAYSCALE)
        print(img)
        shape = img.shape
        values_cat = []
        damage_type = []
        categorie_damages = ['Capot_scratch',	'Calandre_scratch',	'Insigne_scratch',	'BAV_scratch',	'BAR_scratch',	'Op_scratch',	'A_scratch','PAV_scratch',	'PAR_scratch',	'RL_scratch',	'FAR_scratch',	'M_scratch',
                                'Capot_broken',	'Calandre_broken',	'Insigne_broken',	'BAV_broken',	'BAR_broken',	'Op_broken',	'A_broken',	'PAV_broken',	'PAR_broken',	'RL_broken',	'FAR_broken',	'M_broken',
                                'Capot_bosses',	'Calandre_bosses',	'Insigne_bosses',	'BAV_bosses',	'BAR_bosses',	'OP_bosses',	'A_bosses',	'PAV_bosses',	'PAR_bosses',	'RL_bosses',	'FAR_bosses',	'M_bosses']
        # categorie_damages = ['f_boss','n_fingger','od_casse','og_boss','ghazali']
        k=0
        for cat in labels :
            values_cat.append(cat['values'][0])
        damage_type=values_cat[0].split(',')
        print(damage_type)
        assert set(categorie_damages).issuperset(set(damage_type)) , 'Error in lists'
        assert len(damage_type)==len(annotations),'lent damage diffrents of lent of annotations'
        structuredArr = None
        for annot in annotations:
            blobs = []
            l=[]
            label = annot['label']
            if (label != ''):
                points = annot['points']
                l.append(points)
                # l.append(shape)
                # update of number de classes == number degat !!!!!!!!!!!!
                cat_damge = damage_type[k]
                blobs.append(l)
                structuredArr,path_dir = poly2mask(blobs, name_image, path_to_masks_folder,cat_damge,shape, label,structuredArr)
                k=k+1
            
        np.savetxt(path_dir+'/struct_array.csv', structuredArr, delimiter=',', fmt=['%s' , '%s'], header='name_mask,type', comments='')
            
if __name__ == '__main__':

    ### parsing the paramters ###
    parser = argparse.ArgumentParser()

    parser.add_argument("path_json_data", type=str,help="path to json data")
    parser.add_argument("path_original_img", type=str,help="path to orginal images")
    parser.add_argument("path_masks", type=str,help="path to masks for images ")
    
    args = parser.parse_args()
   

    path_to_data_annotation_json =args.path_json_data  #'C:/Users/gh/Desktop/PFE/project/json/data' #args.path_json_data
    path_to_original_images_folder = args.path_original_img  #'C:/Users/gh/Desktop/PFE/project/json/images' #args.path_original_img 
    path_to_masks_folder =args.path_masks #'C:/Users/gh/Desktop/PFE/project/json/masks' #args.path_masks

    files = gb.glob(pathname= str(path_to_data_annotation_json +'/*.json') )
    for data in files:
        convert_data_to_masks(data, path_to_original_images_folder, path_to_masks_folder)
