'''
Modified by Anita Rau, 2024
arau@stanford.edu
'''
import os

import numpy as np
import json
import cv2

import torchvision.transforms as transforms
import imageio
import torch


LERES_SIZE = 448
LERES_RGB_PIXEL_MEANS = (0.485, 0.456, 0.406)
LERES_RGB_PIXEL_VARS = (0.229, 0.224, 0.225)

def read_files(basedir, rgb_file, depth_file):
    fname = os.path.join(basedir, rgb_file)
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:
        convert_fn = cv2.COLOR_BGRA2RGBA
    else:
        convert_fn = cv2.COLOR_BGR2RGB
    img = (cv2.cvtColor(img, convert_fn) / 255.).astype(np.float32) # keep 4 channels (RGBA) if available
    depth_fname = os.path.join(basedir, depth_file)
    depth = cv2.imread(depth_fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
    #resize by half
    #img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv2.INTER_LINEAR)
    #depth = cv2.resize(depth, (depth.shape[1]//2, depth.shape[0]//2), interpolation=cv2.INTER_NEAREST)
    return img, depth

def read_leres_image(basedir, rgb_file):
    
    fname = os.path.join(basedir, rgb_file)
    img = cv2.imread(fname)[:, :, ::-1]

    img = img.copy()
    ### Resize input image
    img = cv2.resize(img, (LERES_SIZE, LERES_SIZE), interpolation=cv2.INTER_LINEAR)

    ### Scale input image
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(LERES_RGB_PIXEL_MEANS, LERES_RGB_PIXEL_VARS)])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)

    return img

def read_leres_depth(basedir, rgb_file, depth_scaling_factor, near, far):
    
    fname = os.path.join(basedir, rgb_file)

    fname = fname.replace("rgb", "target_depth")
    fname = fname.replace(".jpg", ".png")
    
    depth_img = cv2.imread(fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
    depth_img = (depth_img / depth_scaling_factor).astype(np.float32)

    depth_img = cv2.resize(depth_img, (LERES_SIZE, LERES_SIZE), interpolation=cv2.INTER_NEAREST)

    ## Clip with near and far plane
    depth_img = np.clip(depth_img, near, far)


    depth_img = depth_img[np.newaxis, :, :]
    depth_img = torch.from_numpy(depth_img)

    return depth_img


def load_ground_truth_depth(basedir, train_filenames, image_size, depth_scaling_factor):
    H, W = image_size
    gt_depths = []
    gt_valid_depths = []
    for filename in train_filenames:
        filename = filename.replace("rgb", "target_depth")
        filename = filename.replace(".jpg", ".png")
        gt_depth_fname = os.path.join(basedir, filename)
        if os.path.exists(gt_depth_fname):
            gt_depth = cv2.imread(gt_depth_fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
            gt_valid_depth = gt_depth > 0.5
            gt_depth = (gt_depth / depth_scaling_factor).astype(np.float32)
        else:
            gt_depth = np.zeros((H, W))
            gt_valid_depth = np.full_like(gt_depth, False)
        gt_depths.append(np.expand_dims(gt_depth, -1))
        gt_valid_depths.append(gt_valid_depth)
    gt_depths = np.stack(gt_depths, 0)
    gt_valid_depths = np.stack(gt_valid_depths, 0)
    return gt_depths, gt_valid_depths

def load_scene(basedir, train_json = "transforms_train.json"):
    splits = ['train', 'val', 'test', 'video']

    all_imgs = []
    all_depths = []
    all_valid_depths = []
    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []
    for s in splits:
        if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format(s))):

            if s == "train":
                json_fname =  os.path.join(basedir, train_json)
            else:
                json_fname =  os.path.join(basedir, 'transforms_{}.json'.format(s))

            with open(json_fname, 'r') as fp:
                meta = json.load(fp)

            if 'train' in s:
                near = float(meta['near'])
                far = float(meta['far'])
                depth_scaling_factor = float(meta['depth_scaling_factor'])
           
            imgs = []
            depths = []
            valid_depths = []
            poses = []
            intrinsics = []
            
            for frame in meta['frames']:
                if len(frame['file_path']) != 0 or len(frame['depth_file_path']) != 0:
                    img, depth = read_files(basedir, frame['file_path'], frame['depth_file_path'])
                    
                    if depth.ndim == 2:
                        depth = np.expand_dims(depth, -1)

                    valid_depth = depth[:, :, 0] > 0.5 # 0 values are invalid depth
                    depth = (depth / depth_scaling_factor).astype(np.float32)

                    filenames.append(frame['file_path'])
                    
                    imgs.append(img)
                    depths.append(depth)
                    valid_depths.append(valid_depth)

                poses.append(np.array(frame['transform_matrix']))
                H, W = img.shape[:2]
                fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
                all_depths.append(np.array(depths))
                all_valid_depths.append(np.array(valid_depths))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))
        else:
            counts.append(counts[-1])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    depths = np.concatenate(all_depths, 0)
    valid_depths = np.concatenate(all_valid_depths, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    
    gt_depths, gt_valid_depths = load_ground_truth_depth(basedir, filenames, (H, W), depth_scaling_factor)
    
    return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, gt_depths, gt_valid_depths

def load_scene_nogt(basedir, train_json = "transforms_train.json"):
    splits = ['train', 'val', 'test', 'video']

    all_imgs = []
    all_depths = []
    all_valid_depths = []
    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []
    for s in splits:
        if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format(s))):

            if s == "train":
                json_fname =  os.path.join(basedir, train_json)
            else:
                json_fname =  os.path.join(basedir, 'transforms_{}.json'.format(s))

            with open(json_fname, 'r') as fp:
                meta = json.load(fp)

            if 'train' in s:
                near = float(meta['near'])
                far = float(meta['far'])
                depth_scaling_factor = float(meta['depth_scaling_factor'])
           
            imgs = []
            depths = []
            valid_depths = []
            poses = []
            intrinsics = []
            
            for frame in meta['frames']:
                if len(frame['file_path']) != 0 or len(frame['depth_file_path']) != 0:
                    # img, depth = read_files(basedir, frame['file_path'], frame['depth_file_path'])
                    img, depth = read_files(basedir, frame['file_path'], frame['depth_file_path'].split(".")[0]+".png")
                    
                    if depth.ndim == 2:
                        depth = np.expand_dims(depth, -1)

                    valid_depth = depth[:, :, 0] > 0.5 # 0 values are invalid depth
                    depth = (depth / depth_scaling_factor).astype(np.float32)

                    filenames.append(frame['file_path'])
                    
                    imgs.append(img)
                    depths.append(depth)
                    valid_depths.append(valid_depth)

                poses.append(np.array(frame['transform_matrix']))
                H, W = img.shape[:2]
                fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
                all_depths.append(np.array(depths))
                all_valid_depths.append(np.array(valid_depths))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))
        else:
            counts.append(counts[-1])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    depths = np.concatenate(all_depths, 0)
    valid_depths = np.concatenate(all_valid_depths, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    
    # gt_depths, gt_valid_depths = load_ground_truth_depth(basedir, filenames, (H, W), depth_scaling_factor)
    
    return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, None, None


def load_ddp_test_hypos(basedir, i_split, filenames, near, far):
        hypo_dir = os.path.join(basedir, "test", "DiffusionDP")
        
        train_idx = i_split[2]

        all_depth_hypothesis, all_filenames = [], []

        for i in range(len(train_idx)):
            filename = filenames[train_idx[i]]
            img_id = filename.split("/")[-1].split(".")[0]

            cimle_depth_name = os.path.join(hypo_dir, filename.split('/')[-1]+".npy")
            cimle_depth = np.load(cimle_depth_name).astype(np.float32).squeeze(1)
            
            cimle_depth = np.expand_dims(cimle_depth, -1)
            all_depth_hypothesis.append(cimle_depth)
            all_filenames.append("test/rgb/" + filename.split('/')[-1])

        all_depth_hypothesis = np.array(all_depth_hypothesis)

        ### Clamp depth hypothesis to near plane and far plane
        #all_depth_hypothesis = np.clip(all_depth_hypothesis + np.random.randn(*all_depth_hypothesis.shape) * 0.1, near, far)
        all_depth_hypothesis = np.clip(all_depth_hypothesis, near, far)
        return all_depth_hypothesis, all_filenames

def load_scene_scannet(basedir, hypo_dir, num_hypothesis=20, train_json = "transforms_train.json", init_scales=False, scales_dir=None, gt_init=False, hypo_model='cimle', uncertainty_maps=None):
    splits = ['train', 'val', 'test', 'video']

    all_imgs = []
    all_depths = []
    all_valid_depths = []
    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []
    for s in splits:
        if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format(s))):

            if s == "train":
                json_fname =  os.path.join(basedir, train_json)
            else:
                json_fname =  os.path.join(basedir, 'transforms_{}.json'.format(s))

            with open(json_fname, 'r') as fp:
                meta = json.load(fp)

            if 'train' in s:
                near = float(meta['near'])
                far = float(meta['far'])
                depth_scaling_factor = float(meta['depth_scaling_factor'])
           
            imgs = []
            depths = []
            valid_depths = []
            poses = []
            intrinsics = []
            
            for frame in meta['frames']:
                frame['file_path'] = frame['file_path'].replace('\r','')
                frame['depth_file_path'] = frame['depth_file_path'].replace('\r','')
                # SfM depths
                if len(frame['file_path']) != 0 or len(frame['depth_file_path']) != 0:
                    img, depth = read_files(basedir, frame['file_path'], frame['depth_file_path'].replace('..','.'))
                    
                    if depth.ndim == 2:
                        depth = np.expand_dims(depth, -1)

                    valid_depth = depth[:, :, 0] > 0.5 # 0 values are invalid depth
                    depth = (depth / depth_scaling_factor).astype(np.float32)

                    filenames.append(frame['file_path'])
                    
                    imgs.append(img)
                    depths.append(depth)
                    valid_depths.append(valid_depth)

                poses.append(np.array(frame['transform_matrix']))
                H, W = img.shape[:2]
                fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
                #fx = fx //2
                #fy = fy //2
                #cx = cx //2
                #cy = cy //2
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
                all_depths.append(np.array(depths))
                all_valid_depths.append(np.array(valid_depths))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))
        else:
            counts.append(counts[-1])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    #i_split[0] = np.array([1, 2, 3, 4, 5, 8, 11, 12, 13])  # TODO remove this!!!
    imgs = np.concatenate(all_imgs, 0)
    depths = np.concatenate(all_depths, 0)
    valid_depths = np.concatenate(all_valid_depths, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)

    gt_depths, gt_valid_depths = load_ground_truth_depth(basedir, filenames, (H, W), depth_scaling_factor)

    ############################################    
    #### Load cimle depth maps ####
    ############################################    
    ## For now only for train poses
    if hypo_model == 'cimle':
        leres_dir = os.path.join(basedir, "train", "leres_cimle", hypo_dir)
        paths = os.listdir(leres_dir)
        
        train_idx = i_split[0]

        all_depth_hypothesis = []

        for i in range(len(train_idx)):
            filename = filenames[train_idx[i]]
            img_id = filename.split("/")[-1].split(".")[0]
            curr_depth_hypotheses = []

            for j in range(num_hypothesis):
                cimle_depth_name = os.path.join(leres_dir, img_id+"_"+str(j)+".npy")
                cimle_depth = np.load(cimle_depth_name).astype(np.float32)

                ## To adhere to the shape of depths
                # cimle_depth = cimle_depth.T ## Buggy version
                cimle_depth = cimle_depth
                
                cimle_depth = np.expand_dims(cimle_depth, -1)
                curr_depth_hypotheses.append(cimle_depth)

            curr_depth_hypotheses = np.array(curr_depth_hypotheses)
            all_depth_hypothesis.append(curr_depth_hypotheses)  # (20, 468, 624, 1)

        all_depth_hypothesis = np.array(all_depth_hypothesis) # (18, 20, 468, 624, 1)

        ### Clamp depth hypothesis to near plane and far plane
        all_depth_hypothesis = np.clip(all_depth_hypothesis, near, far)
        #########################################
    elif hypo_model == 'ddp':
        #hypo_dir = os.path.join(basedir.replace('scade','scade_vs'), "train", "DiffusionDP")
        hypo_dir = os.path.join(basedir, "train", "DiffusionDP")

        
        train_idx = i_split[0]

        if uncertainty_maps is not None:
            #uncertainty_map_dir = os.path.join(basedir.replace('scade','scade_vs'), "train", uncertainty_maps)
            uncertainty_map_dir = os.path.join(basedir, "train", uncertainty_maps)


        all_depth_hypothesis = []
        all_uncertainties = []

        for i in range(len(train_idx)):
            filename = filenames[train_idx[i]]
            img_id = filename.split("/")[-1].split(".")[0]

            cimle_depth_name = os.path.join(hypo_dir, filename.split('/')[-1]+".npy")
            cimle_depth = np.load(cimle_depth_name).astype(np.float32).squeeze(1)
            if uncertainty_maps is not None:
                uncertainty_map_name = os.path.join(uncertainty_map_dir, filename.split('/')[-1]+".npy")
                uncertainty_map = np.load(uncertainty_map_name).astype(np.float32)
                all_uncertainties.append(uncertainty_map)
            
            cimle_depth = np.expand_dims(cimle_depth, -1)
            all_depth_hypothesis.append(cimle_depth)

        all_depth_hypothesis = np.array(all_depth_hypothesis)
        if uncertainty_maps is not None:
            uncertainty_maps = np.array(all_uncertainties)  # n_ims, height //4, width //4

        ### Clamp depth hypothesis to near plane and far plane
        #all_depth_hypothesis = np.clip(all_depth_hypothesis + np.random.randn(*all_depth_hypothesis.shape) * 0.1, near, far)
        all_depth_hypothesis = np.clip(all_depth_hypothesis, near, far)  # n_ims, n_hypos, height, width, 1
    elif hypo_model == 'dino':
        hypo_dir = os.path.join(basedir.replace('scade','scade_vs'), "train", "Dinov2")
        hypos = np.load(hypo_dir + '/pred_depth_maps_050.npy')  # n_hypos, n_ims, height, width
        all_depth_hypothesis = hypos.transpose(1,0,2,3)
        if uncertainty_maps is not None:
            uncertainty_maps = all_depth_hypothesis.std(1)

        all_depth_hypothesis = np.expand_dims(np.clip(all_depth_hypothesis, near, far),4)

    elif hypo_model == 'midas':
        hypo_dir = os.path.join(basedir, "train", "midas_depth")

        
        train_idx = i_split[0]

        if uncertainty_maps is not None:
            uncertainty_map_dir = os.path.join(basedir, "train", uncertainty_maps)


        all_depth_hypothesis = []
        all_uncertainties = []

        for i in range(len(train_idx)):
            filename = filenames[train_idx[i]]
            img_id = filename.split("/")[-1].split(".")[0]

            cimle_depth_name = os.path.join(hypo_dir, filename.split('/')[-1].split('.')[0]+"_depth.npy")
            cimle_depth = np.load(cimle_depth_name).astype(np.float32)
            if uncertainty_maps is not None:
                uncertainty_map_name = os.path.join(uncertainty_map_dir, filename.split('/')[-1]+".npy")
                uncertainty_map = np.load(uncertainty_map_name).astype(np.float32)
                all_uncertainties.append(uncertainty_map)
            
            cimle_depth = np.expand_dims(cimle_depth, -1)
            all_depth_hypothesis.append(cimle_depth)

        all_depth_hypothesis = np.array(all_depth_hypothesis)
        if uncertainty_maps is not None:
            uncertainty_maps = np.array(all_uncertainties)  # n_ims, height //4, width //4

        ### Clamp depth hypothesis to near plane and far plane
        #all_depth_hypothesis = np.clip(all_depth_hypothesis + np.random.randn(*all_depth_hypothesis.shape) * 0.1, near, far)
        all_depth_hypothesis = np.clip(all_depth_hypothesis, near, far)  # n_ims, n_hypos, height, width, 1
        all_depth_hypothesis = np.expand_dims(all_depth_hypothesis, 1)

    elif hypo_model == 'depth-anything':
        hypo_dir = os.path.join(basedir, "train", "depth_anything")

        train_idx = i_split[0]

        all_depth_hypothesis = []

        for i in range(len(train_idx)):
            filename = filenames[train_idx[i]]
            img_id = filename.split("/")[-1].split(".")[0]

            cimle_depth_name = os.path.join(hypo_dir, filename.split('/')[-1]+".npy")
            cimle_depth = np.load(cimle_depth_name).astype(np.float32)
            cimle_depth = np.expand_dims(cimle_depth, -1)
            all_depth_hypothesis.append(cimle_depth)

        all_depth_hypothesis = np.array(all_depth_hypothesis)
        all_depth_hypothesis = np.clip(all_depth_hypothesis, near, far)  # n_ims, n_hypos, height, width, 1
        all_depth_hypothesis = np.expand_dims(all_depth_hypothesis, 1)
    elif hypo_model == 'depth-anything-metric':
        hypo_dir = os.path.join(basedir, "train", "depth_anything_metric")

        train_idx = i_split[0]

        all_depth_hypothesis = []

        for i in range(len(train_idx)):
            filename = filenames[train_idx[i]]
            img_id = filename.split("/")[-1].split(".")[0]

            cimle_depth_name = os.path.join(hypo_dir, filename.split('/')[-1]+".npy")
            cimle_depth = np.load(cimle_depth_name).astype(np.float32)
            cimle_depth = np.expand_dims(cimle_depth, -1)
            all_depth_hypothesis.append(cimle_depth)

        all_depth_hypothesis = np.array(all_depth_hypothesis)
        all_depth_hypothesis = np.clip(all_depth_hypothesis, near, far)  # n_ims, n_hypos, height, width, 1
        all_depth_hypothesis = np.expand_dims(all_depth_hypothesis, 1)


    ############################################    
    #### Load scale/shift init ####
    ############################################        
    if init_scales:
        scale_shift_dir = os.path.join(basedir, "train", "scale_shift_inits", scales_dir)
        train_idx = i_split[0]

        all_scales_init = []        
        all_shifts_init = []        

        for i in range(len(train_idx)):
            filename = filenames[train_idx[i]]
            img_id = filename.split("/")[-1].split(".")[0]

            if not gt_init:
                print("Use SfM scale init.")
                curr_scale_shift_name = os.path.join(scale_shift_dir, img_id+"_sfminit.npy")
            else:
                print("Use gt scale init.")
                curr_scale_shift_name = os.path.join(scale_shift_dir, img_id+"_gtinit.npy")

            curr_scale_shift = np.load(curr_scale_shift_name).astype(np.float32)
            print(curr_scale_shift)

            all_scales_init.append(curr_scale_shift[0])
            all_shifts_init.append(curr_scale_shift[1])

        all_scales_init = np.array(all_scales_init)
        all_shifts_init = np.array(all_shifts_init)

        return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, gt_depths, gt_valid_depths, all_depth_hypothesis, all_scales_init, all_shifts_init

    return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, gt_depths, gt_valid_depths, all_depth_hypothesis, filenames, uncertainty_maps


def load_scene_processed(basedir, hypo_dir, num_hypothesis=20, train_json = "transforms_train.json", init_scales=False, scales_dir=None, gt_init=False, hypo_model='cimle',uncertainty_maps=None):
    splits = ['train', 'val', 'test', 'video']

    all_imgs = []
    all_depths = []
    all_valid_depths = []
    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []

    # print(basedir)

    for s in splits:
        if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format(s))):

            # print("File exists.")

            if s == "train":
                json_fname =  os.path.join(basedir, train_json)
            else:
                json_fname =  os.path.join(basedir, 'transforms_{}.json'.format(s))

            with open(json_fname, 'r') as fp:
                meta = json.load(fp)

            if 'train' in s:
                near = float(meta['near'])
                far = float(meta['far'])
                depth_scaling_factor = float(meta['depth_scaling_factor'])
           
            imgs = []
            depths = []
            valid_depths = []
            poses = []
            intrinsics = []
            
            for frame in meta['frames']:
                if len(frame['file_path']) != 0 or len(frame['depth_file_path']) != 0:
                    frame['file_path'] = frame['file_path'].replace('\r','')
                    frame['depth_file_path'] = frame['depth_file_path'].replace('\r','')
                    # img, depth = read_files(basedir, frame['file_path'], frame['depth_file_path'])
                    img, depth = read_files(basedir, frame['file_path'], frame['depth_file_path'].split(".")[0]+".png")
                    
                    if depth.ndim == 2:
                        depth = np.expand_dims(depth, -1)

                    valid_depth = depth[:, :, 0] > 0.5 # 0 values are invalid depth
                    depth = (depth / depth_scaling_factor).astype(np.float32)

                    filenames.append(frame['file_path'])
                    
                    imgs.append(img)
                    depths.append(depth)
                    valid_depths.append(valid_depth)

                poses.append(np.array(frame['transform_matrix']))
                H, W = img.shape[:2]
                fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
                fx = fx //2
                fy = fy //2
                cx = cx //2
                cy = cy //2
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
                all_depths.append(np.array(depths))
                all_valid_depths.append(np.array(valid_depths))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))
        else:
            counts.append(counts[-1])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    depths = np.concatenate(all_depths, 0)
    valid_depths = np.concatenate(all_valid_depths, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
       
    # gt_depths, gt_valid_depths = load_ground_truth_depth(basedir, filenames, (H, W), depth_scaling_factor)

    ############################################    
    #### Load cimle depth maps ####
    ############################################    
    ## For now only for train poses
    if hypo_model == 'cimle':
        leres_dir = os.path.join(basedir, "train", "leres_cimle", hypo_dir)
        paths = os.listdir(leres_dir)
        
        train_idx = i_split[0]

        all_depth_hypothesis = []

        for i in range(len(train_idx)):
            filename = filenames[train_idx[i]]
            img_id = filename.split("/")[-1].split(".")[0]
            curr_depth_hypotheses = []

            for j in range(num_hypothesis):
                cimle_depth_name = os.path.join(leres_dir, img_id+"_"+str(j)+".npy")
                cimle_depth = np.load(cimle_depth_name).astype(np.float32)

                ## To adhere to the shape of depths
                # cimle_depth = cimle_depth.T ## Buggy version
                cimle_depth = cimle_depth
                
                cimle_depth = np.expand_dims(cimle_depth, -1)
                curr_depth_hypotheses.append(cimle_depth)

            curr_depth_hypotheses = np.array(curr_depth_hypotheses)
            all_depth_hypothesis.append(curr_depth_hypotheses)

        all_depth_hypothesis = np.array(all_depth_hypothesis)

        ### Clamp depth hypothesis to near plane and far plane
        all_depth_hypothesis = np.clip(all_depth_hypothesis, near, far)
        #########################################

    elif hypo_model == 'ddp':
        #hypo_dir = os.path.join(basedir.replace('scade','scade_vs'), "train", "DiffusionDP")
        hypo_dir = os.path.join(basedir, "train", "DiffusionDP")

        
        train_idx = i_split[0]

        if uncertainty_maps is not None:
            #uncertainty_map_dir = os.path.join(basedir.replace('scade','scade_vs'), "train", uncertainty_maps)
            uncertainty_map_dir = os.path.join(basedir, "train", uncertainty_maps)


        all_depth_hypothesis = []
        all_uncertainties = []

        for i in range(len(train_idx)):
            filename = filenames[train_idx[i]]
            img_id = filename.split("/")[-1].split(".")[0]

            cimle_depth_name = os.path.join(hypo_dir, filename.split('/')[-1]+".npy")
            cimle_depth = np.load(cimle_depth_name).astype(np.float32).squeeze(1)
            if uncertainty_maps is not None:
                uncertainty_map_name = os.path.join(uncertainty_map_dir, filename.split('/')[-1]+".npy")
                uncertainty_map = np.load(uncertainty_map_name).astype(np.float32)
                all_uncertainties.append(uncertainty_map)
            
            cimle_depth = np.expand_dims(cimle_depth, -1)
            all_depth_hypothesis.append(cimle_depth)

        all_depth_hypothesis = np.array(all_depth_hypothesis)
        if uncertainty_maps is not None:
            uncertainty_maps = np.array(all_uncertainties)  # n_ims, height //4, width //4

        ### Clamp depth hypothesis to near plane and far plane
        #all_depth_hypothesis = np.clip(all_depth_hypothesis + np.random.randn(*all_depth_hypothesis.shape) * 0.1, near, far)
        all_depth_hypothesis = np.clip(all_depth_hypothesis, near, far)  # n_ims, n_hypos, height, width, 1
    ############################################    
    #### Load scale/shift init ####
    ############################################        
    if init_scales:
        scale_shift_dir = os.path.join(basedir, "train", "scale_shift_inits", scales_dir)
        train_idx = i_split[0]

        all_scales_init = []        
        all_shifts_init = []        

        for i in range(len(train_idx)):
            filename = filenames[train_idx[i]]
            img_id = filename.split("/")[-1].split(".")[0]

            if not gt_init:
                print("Use SfM scale init.")
                curr_scale_shift_name = os.path.join(scale_shift_dir, img_id+"_sfminit.npy")
            else:
                print("Use gt scale init.")
                curr_scale_shift_name = os.path.join(scale_shift_dir, img_id+"_gtinit.npy")

            curr_scale_shift = np.load(curr_scale_shift_name).astype(np.float32)
            print(curr_scale_shift)

            all_scales_init.append(curr_scale_shift[0])
            all_shifts_init.append(curr_scale_shift[1])

        all_scales_init = np.array(all_scales_init)
        all_shifts_init = np.array(all_shifts_init)

        return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, None, None, all_depth_hypothesis, all_scales_init, all_shifts_init

    return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, None, None, all_depth_hypothesis, filenames, uncertainty_maps


























