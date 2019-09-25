import numpy as np
from os import path, makedirs, listdir
from operator import itemgetter
from sampler import loadJSON
import shutil
from PIL import Image
import imageio
from tqdm import tqdm

def load_img_paths_and_params(dir, img_folder):
    img_dir = path.join(dir, img_folder)

    imgs_folder = path.join(img_dir)
    jsons_folder = path.join(img_dir, 'json')

    imgs_paths = [path.join(imgs_folder, f) for f in listdir(imgs_folder)
                   if f.endswith('.png') ]

    imgs_params = []
    for img in imgs_paths:
        img_name = path.splitext(path.basename(img))[0]
        json_path = path.join(jsons_folder, img_name + '.json')

        imgs_params.append(loadJSON(json_path))

    return imgs_paths, imgs_params

def sort_img_dirs_and_params(save_folder, img_folder, filter_keys):

    imgs_paths, imgs_params = load_img_paths_and_params(save_folder, img_folder)

    # sort so similar params are together, with iteration number as well
    sort_keys = filter_keys[:]
    sort_keys.append('iteration')

    sort_list = [tuple((img_params[k]) for k in sort_keys) for img_params in imgs_params]
    sort_indices = sorted(enumerate(sort_list), key=itemgetter(1))
    sort_indices = [i[0] for i in sort_indices]

    # if you want to sort the indices with respect to different img_params then use:
    # sorted(enumerate(sort_list), key=lambda x: enum_sort_order(x, [1, 0, 2, 3, 4, 5]))

    # here p[-1] should correspond to the 'iteration' parameter which designates the
    # img number for the same seed. The idea is to sort, and then average, over imgs
    # generated from the same seed.
    # max_iters = np.max([p[-1] for p in sort_list]) + 1

    corrs_paths = [imgs_paths[i] for i in sort_indices]
    imgs_params = [imgs_params[i] for i in sort_indices]

    return corrs_paths, imgs_params, sort_list

if __name__ == '__main__':

    # image dataset dir
    save_folder = 'save/img_architectures/'
    img_folder = 'big'
    label_keys = [
        'total_neurons',
        'num_layers',
        'omega',
        'alpha',
        'mu'
    ]
    # load image files and their parameters into a sorted list
    imgs_paths, imgs_params, sort_list = sort_img_dirs_and_params(save_folder, img_folder,
                                                                   filter_keys=label_keys)

    BW = True  # include black and white sub-folders per subject
    Ns = 60  # number of subjects to generate samples for
    Na = 200  # number of architectures to sample
    Ni = 1  # number of images per architecture to sample
    Nt = Na * Ni  # total number of images per subject

    # total number of architectures
    NArchTotal = len(imgs_paths) / 40  # 40 images per architecture

    if BW:
        imgType = ['COLOUR', 'BW']
    else:
        imgType = ['COLOUR']

    for imgType_i in imgType:
        # create sub-folders
        folder = path.join(save_folder, img_folder + '_aesthetic_experiment/')
        print(f'\nPreparing {imgType_i} images...')
        if not path.exists(folder):
            makedirs(folder)

        # sample images from full set. Copy them to new folders and convert to BW if necessary
        for S in tqdm(range(0, Ns)):
            rA = np.random.randint(0, NArchTotal, size=Na)
            rI = np.random.randint(0, 40, size=(Na, Ni))

            # image index
            img_index = [40*a + b for i, a in enumerate(rA) for b in rI[i]]

            # create sub-folders for subjects
            Sfolder = path.join(folder, f'subject_{S}')
            imfolder = path.join(Sfolder, imgType_i)

            if not path.exists(Sfolder):
                makedirs(Sfolder)
            if not path.exists(imfolder):
                makedirs(imfolder)

            for img_i in img_index:

                img_base = path.splitext(path.basename(imgs_paths[img_i]))[0]
                rand_prefix = str(np.random.randint(0, 1e12)) + '_'
                new_base = rand_prefix + img_base
                new_dir = path.join(imfolder,  new_base)

                if imgType_i == 'BW':
                    img = np.array(Image.open(imgs_paths[img_i]).convert('L'))
                    imageio.imwrite(new_dir, (img).astype(np.uint8), format='png')
                else:
                    shutil.copy(imgs_paths[img_i], new_dir)