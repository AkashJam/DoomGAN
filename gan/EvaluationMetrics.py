import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pix2pix import Generator as pix2pixGen
from wgan import Generator as wganGen
from HybridGen import hybrid_fmaps
from WganGen import wgan_fmaps
from GanMeta import read_record
import ripleyk, math
from skimage import morphology


def plot_prop_graph(props):

    
    sets = ("Monsters", "Ammunitions", "Power-ups", "Artifacts", "Weapons")

    x = np.arange(len(sets))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in props.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Proportions')
    ax.set_title('Game Object Category Proportions')
    ax.set_xticks(x + width, sets)
    ax.legend()
    ax.set_ylim(0, 100)
    plt.show()


def calc_proportions(real, wgan, hybrid, keys, meta):
    real_props = dict()
    wgan_props = dict()
    hybrid_props = dict()
    obj_cate = ['monsters','ammunitions','powerups','artifacts','weapons']
    id = keys.index('essentials')
    for i in range(real.shape[0]):
        for cate in obj_cate:
            min = meta[cate]['min']
            max = meta[cate]['max']
            real_map_props = tf.reduce_sum(tf.cast(tf.logical_and(real[i,:,:,id]>min,real[i,:,:,id]<=max),tf.float16)) / tf.reduce_sum(tf.cast(real[i,:,:,id]>0,tf.float16))
            wgan_map_props = tf.reduce_sum(tf.cast(tf.logical_and(wgan[i,:,:,id]>min,wgan[i,:,:,id]<=max),tf.float16)) / tf.reduce_sum(tf.cast(wgan[i,:,:,id]>0,tf.float16))
            hybrid_map_props = tf.reduce_sum(tf.cast(tf.logical_and(hybrid[i,:,:,id]>min,hybrid[i,:,:,id]<=max),tf.float16)) / tf.reduce_sum(tf.cast(hybrid[i,:,:,id]>0,tf.float16))
            if real_map_props >= 1:
                print(cate, tf.reduce_sum(tf.cast(tf.logical_and(real[i,:,:,id]>min,real[i,:,:,id]<=max),tf.float16)), tf.reduce_sum(tf.cast(real[i,:,:,id]>0,tf.float16)), min, max)
                plt.subplot(1,2,1)
                plt.imshow(real[i,:,:,id])
                plt.subplot(1,2,2)
                plt.imshow(tf.cast(real[i,:,:,id]>0,tf.float16))
                plt.show()
            if i == 0:
                real_props[cate] = [real_map_props]
                wgan_props[cate] = [wgan_map_props]
                hybrid_props[cate] = [hybrid_map_props]
            else:
                real_props[cate].append(real_map_props)
                wgan_props[cate].append(wgan_map_props)
                hybrid_props[cate].append(hybrid_map_props)


    proportions = {'Real': tuple(round(100*sum(real_props[cate])/len(real_props[cate])) for cate in obj_cate), 
                    'Wgan': tuple(round(100*sum(wgan_props[cate])/len(wgan_props[cate])) for cate in obj_cate), 
                    'Hybrid': tuple(round(100*sum(hybrid_props[cate])/len(hybrid_props[cate])) for cate in obj_cate)}
    plot_prop_graph(proportions)


def calc_stats(real, wgan, hybrid, keys, meta):
    obj_cate = ['monsters','ammunitions','powerups','artifacts','weapons']
    id = keys.index('essentials')
    n_items = meta['essentials']['max']
    real_count = [tf.where(real[:,:,:,id]==i+1).shape[0]/real.shape[0] for i in range(n_items)]
    wgan_count = [tf.where(wgan[:,:,:,id]==i+1).shape[0]/wgan.shape[0] for i in range(n_items)]
    hybrid_count = [tf.where(hybrid[:,:,:,id]==i+1).shape[0]/hybrid.shape[0] for i in range(n_items)]
    for cate in obj_cate:
        min = meta[cate]['min']
        max = meta[cate]['max']
        real_cate = [0] + real_count[min:max]
        wgan_cate = [0] + wgan_count[min:max]
        hybrid_cate = [0] + hybrid_count[min:max]
        fig, ax = plt.subplots(1, 1)
        plt.xlabel(cate+' type')
        plt.ylabel('Average object population')
        plt.plot(real_cate, label="real")
        plt.plot(wgan_cate, label="wgan")
        plt.plot(hybrid_cate, label="hybrid")
        ax.set_xlim(1, max-min)
        ax.xaxis.get_major_locator().set_params(integer=True)
        plt.legend() # must be after labels
        # plt.savefig(location+'disc_loss_graph')
        plt.show()


def calc_spatial_homogenity(maps,threshold=1,gen=True):
    floor_id = keys.index('floormap')
    things_id = keys.index('essentials')
    position = [n for n in range(256)]
    for i in range(maps.shape[0]):
        if gen:
            map_size = np.sum((maps[i,:,:,floor_id]>0).astype(int))
            map = morphology.remove_small_holes(maps[i,:,:,floor_id]>0,24)
            map = morphology.remove_small_objects(map,map_size/2).astype(int)
        else:
            map = (maps[i,:,:,floor_id]>0).astype(int)
        if np.sum(map)==0:
            print('next map')
            continue
        empty_space = np.logical_and(np.logical_not(maps[i,:,:,things_id]>0),map).astype(int)
        x_min = max(position, key= lambda x: map[x,:].any()*(256-x))
        x_max = max(position, key= lambda x: map[x,:].any()*x)
        y_min = max(position, key= lambda y: map[:,y].any()*(256-y))
        y_max = max(position, key= lambda y: map[:,y].any()*y)
        min_side = min(x_max-x_min,y_max-y_min)
        window_sizes = np.linspace(2,min_side,num=20).astype(int).tolist()
        cropped_empty_space = empty_space[x_min:x_max,y_min:y_max]
        cropped_map = map[x_min:x_max+1,y_min:y_max+1]
        # print(empty_space.shape, cropped_empty_space.shape)
        # window_sizes = [i for i in range(2,max_size,2)]
        for ws in window_sizes:
            y = sliding_window_slicing(cropped_empty_space, no_items=ws, item_type=1)
            map_y = sliding_window_slicing(cropped_map, no_items=ws, item_type=1)
            for j in range(y.shape[0]):
                Ty = np.transpose(y[j,:,:])
                z = sliding_window_slicing(Ty, no_items=ws, item_type=1)
                map_ty = np.transpose(map_y[j,:,:])
                map_z = sliding_window_slicing(map_ty, no_items=ws, item_type=1)
                slices = z if j == 0 else np.concatenate((slices,z))
                map_slices = map_z if j == 0 else np.concatenate((map_slices,map_z))
            valid_w = 0
            for j in range(slices.shape[0]):
                valid_slice = np.sum(map_slices[j,:,:])/ws**2 == threshold
                if valid_slice:
                    fill = np.sum(slices[j,:,:])/np.sum(map_slices[j,:,:])
                    empty = fill if fill == threshold else 0
                    wslice_homogenity = [empty] if valid_w == 0 else wslice_homogenity+[empty]
                    valid_w += 1
            if valid_w == 0: 
                # print('break')
                break
            else:
                w_homogenity = sum(wslice_homogenity)/valid_w
                map_homogenity = [w_homogenity] if window_sizes.index(ws) == 0 else map_homogenity+[w_homogenity]
                print(ws,w_homogenity)
        print(sum([map_homogenity[k]/(len(map_homogenity)-k) for k in range(len(map_homogenity))]),ws,np.sum(map),min_side)
        # plt.subplot(1,2,1)
        # plt.imshow(cropped_empty_space)
        # plt.subplot(1,2,2)
        # plt.imshow(cropped_map)
        # plt.show()

# def plot_spac_homogenity():


    


def calc_RipleyK(maps, radii,gen=True):
    floor_id = keys.index('floormap')
    things_id = keys.index('essentials')
    position = [n for n in range(256)]
    for i in range(maps.shape[0]):
        if gen:
            map_size = np.sum((maps[i,:,:,floor_id]>0).astype(int))
            map = morphology.remove_small_holes(maps[i,:,:,floor_id]>0,24)
            map = morphology.remove_small_objects(map,map_size/3).astype(int)
        else:
            map = (maps[i,:,:,floor_id]>0).astype(int)
        f_area = np.sum(map)
        if f_area==0:
            continue
        x_min = max(position, key= lambda x: map[x,:].any()*(256-x))
        x_max = max(position, key= lambda x: map[x,:].any()*x)
        y_min = max(position, key= lambda y: map[:,y].any()*(256-y))
        y_max = max(position, key= lambda y: map[:,y].any()*y)
        samples = np.transpose(np.nonzero(maps[i,:,:,things_id]*map))
        rescaled_x = 2*(samples[:,0]-x_min-(x_max-x_min)/2)/(x_max-x_min)
        rescaled_y = 2*(samples[:,1]-y_min-(y_max-y_min)/2)/(y_max-y_min)
        if len(rescaled_x) == 0 or len(rescaled_y) == 0:
            continue
        for j,r in enumerate(radii):
            k = ripleyk.calculate_ripley(r, 1, d1=rescaled_x, d2=rescaled_y, boundary_correct=True, CSR_Normalise=True)
            ripk = [k] if j == 0 else ripk + [k]
        area = (x_max-x_min)*(y_max-y_min)
        level_areas = [f_area/area] if i == 0 else level_areas + [f_area/area]
        level_ripk = [ripk] if i == 0 else level_ripk + [ripk]
    return level_areas, level_ripk

def plot_RipleyK(real, wgan, hybrid):  
    test_radii = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]               
    rlevel_areas, rlevel_ripk = calc_RipleyK(real, test_radii,gen=False)
    hlevel_areas, hlevel_ripk = calc_RipleyK(hybrid, test_radii)
    wlevel_areas, wlevel_ripk = calc_RipleyK(wgan, test_radii)
    
    # accr_ripk = [np.array([ripk[i] for ripk in rlevel_ripk]) * np.array(rlevel_areas) for i in range(len(test_radii))]
    # acch_ripk = [np.array([ripk[i] for ripk in hlevel_ripk]) * np.array(hlevel_areas) for i in range(len(test_radii))]
    # accw_ripk = [np.array([ripk[i] for ripk in wlevel_ripk]) * np.array(wlevel_areas) for i in range(len(test_radii))]


    # print('real: {} hybrid: {} wgan: {}'.format(sum([sum([ripk[i] for ripk in np.array(accr_ripk).T.tolist()])/len(rlevel_areas) for i in range(len(test_radii))])/len(test_radii), 
    #                                             sum([sum([ripk[i] for ripk in np.array(acch_ripk).T.tolist()])/len(hlevel_areas) for i in range(len(test_radii))])/len(test_radii), 
    #                                             sum([sum([ripk[i] for ripk in np.array(accw_ripk).T.tolist()])/len(wlevel_areas) for i in range(len(test_radii))])/len(test_radii)))
    
    
    print('real: {} hybrid: {} wgan: {}'.format([sum([ripk[i] for ripk in rlevel_ripk])/len(rlevel_areas) for i in range(len(test_radii))], 
                                                [sum([ripk[i] for ripk in hlevel_ripk])/len(hlevel_areas) for i in range(len(test_radii))], 
                                                [sum([ripk[i] for ripk in wlevel_ripk])/len(wlevel_areas) for i in range(len(test_radii))]))
    
    for i,r in enumerate(test_radii):
        # plt.plot(radii, np.array(rlevel_ripk).T.tolist(), 'bo')
        # plt.show()
        plt.figure(figsize=(16, 8))
        plt.xlabel('Percentage of Covered Area')
        plt.ylabel('Ripley K with radius '+ str(r))
        ax1 = plt.subplot(1, 3, 1)
        plt.plot(rlevel_areas, [radius[i] for radius in rlevel_ripk], 'bo', label='Real')
        ax2 = plt.subplot(1, 3, 2)
        plt.plot(wlevel_areas, [radius[i] for radius in wlevel_ripk], 'ro', label='Wgan')
        ax3 = plt.subplot(1, 3, 3)
        plt.plot(hlevel_areas, [radius[i] for radius in hlevel_ripk], 'rx', label='Hybrid')
        ax1.set_ylim(-0.5, 2.5)
        ax2.set_ylim(-0.5, 2.5)
        ax3.set_ylim(-0.5, 2.5)
        plt.show()

def sliding_window_slicing(a, no_items, item_type=0):
    """This method perfoms sliding window slicing of numpy arrays

    Parameters
    ----------
    a : numpy
        An array to be slided in subarrays
    no_items : int
        Number of sliced arrays or elements in sliced arrays
    item_type: int
        Indicates if no_items is number of sliced arrays (item_type=0) or
        number of elements in sliced array (item_type=1), by default 0

    Return
    ------
    numpy
        Sliced numpy array
    """
    if item_type == 0:
        no_slices = no_items
        no_elements = len(a) + 1 - no_slices
        if no_elements <=0:
            raise ValueError('Sliding slicing not possible, no_items is larger than ' + str(len(a)))
    else:
        no_elements = no_items                
        no_slices = len(a) - no_elements + 1
        if no_slices <=0:
            raise ValueError('Sliding slicing not possible, no_items is larger than ' + str(len(a)))

    subarray_shape = a.shape[1:]
    shape_cfg = (no_slices, no_elements) + subarray_shape
    strides_cfg = (a.strides[0],) + a.strides
    as_strided = np.lib.stride_tricks.as_strided #shorthand
    return as_strided(a, shape=shape_cfg, strides=strides_cfg)


if __name__ == "__main__":
    b_size = 1
    z_dim = 100

    trad_wgen = wganGen(4, z_dim)
    checkpoint_dir = './training_checkpoints/wgan'
    checkpoint = tf.train.Checkpoint(generator=trad_wgen)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    hybrid_wgen = wganGen(3, z_dim)
    checkpoint_dir = './training_checkpoints/hybrid/wgan'
    checkpoint = tf.train.Checkpoint(generator=hybrid_wgen)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    hybrid_pgen = pix2pixGen()
    checkpoint_dir = './training_checkpoints/hybrid/pix2pix'
    checkpoint = tf.train.Checkpoint(generator=hybrid_pgen)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    dataset, map_meta, sample = read_record(batch_size=b_size)
    for i, data in dataset.enumerate().as_numpy_iterator():
        z = tf.random.normal([b_size, z_dim])
        noise = tf.random.normal([b_size, 256, 256, 1])
        wgan_maps, keys, n_items = wgan_fmaps(trad_wgen, z)
        hybrid_maps, keys, n_items = hybrid_fmaps(hybrid_wgen, hybrid_pgen, z, noise)
        real_maps = np.stack([data[m] for m in keys], axis=-1)
        r_maps = np.concatenate((r_maps,real_maps), axis=0) if i != 0 else real_maps
        h_maps = np.concatenate((h_maps,hybrid_maps.numpy()), axis=0) if i != 0 else hybrid_maps.numpy()
        w_maps = np.concatenate((w_maps,wgan_maps.numpy()), axis=0) if i != 0 else wgan_maps.numpy()
        break
    calc_spatial_homogenity(w_maps)
    # calc_proportions(r_maps,w_maps,h_maps,keys,map_meta)
    # calc_stats(r_maps,w_maps,h_maps,keys,map_meta)
    # plot_RipleyK(r_maps,w_maps,h_maps)
    
    

