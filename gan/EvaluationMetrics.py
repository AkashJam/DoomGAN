import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from can import Generator as canGen
from wgan import Generator as wganGen
from HybridGen import hybrid_fmaps
from WganGen import wgan_fmaps
from DataProcessing import read_record
import os, json
from NetworkArchitecture import topological_maps, object_maps
from eval_metrics.metrics import level_props, calc_RipleyK


def plot_train_metrics(save_path = 'eval_metrics/'):
    file_path = save_path + 'training_metrics.json'
    if not os.path.isfile(file_path):
        print('No metric record found')
    else:
        with open(file_path, 'r') as jsonfile:
            metrics = json.load(jsonfile)
            GAN_types = list(metrics.keys())
            CAN_types = [n for n in GAN_types if "CAN" in n]
            for key in list(metrics[CAN_types[0]].keys()):
                if 'loss' not in key:
                    for CAN_type in CAN_types:
                        steps = [stp*100 for stp in list(range(len(metrics[CAN_type][key])))]
                        plt.xlabel('Number of Steps')
                        plt.ylabel(key)
                        plt.plot(steps, metrics[CAN_type][key] , label=CAN_type)
                    plt.legend(loc="upper right")
                    # plt.show()
                    plt.savefig(save_path + key + '_graph')
                    plt.close()


def plot_prop_graph(props):
    sets = ("Monsters", "Ammunitions", "Power-ups", "Artifacts", "Weapons")

    x = np.arange(len(sets))  # the label locations
    width = 0.3  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in props.items():
        offset = width * multiplier
        rects = ax.bar(1.5*x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=2)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Proportions')
    ax.set_title('Game Object Category Proportions')
    ax.set_xticks(1.5*x + width, sets)
    ax.legend()
    ax.set_ylim(0, 100)
    plt.show()
    # plt.savefig('./eval_metrics/props')
    # plt.close()


def calc_proportions(real, wgan, hybrid_trad, hybrid_mod, keys, meta):
    real_props = level_props(real, keys, meta, test_size)
    wgan_props = level_props(wgan, keys, meta, test_size)
    hybrid_trad_props = level_props(hybrid_trad, keys, meta, test_size)
    hybrid_mod_props = level_props(hybrid_mod, keys, meta, test_size)

    proportions = {'Real': tuple(round(100*sum(real_props[cate])/len(real_props[cate])) for cate in object_maps), 
                    'WGAN': tuple(round(100*sum(wgan_props[cate])/len(wgan_props[cate])) for cate in object_maps), 
                    'Hybrid - Trad CAN': tuple(round(100*sum(hybrid_trad_props[cate])/len(hybrid_trad_props[cate])) for cate in object_maps),
                    'Hybrid - Mod CAN': tuple(round(100*sum(hybrid_mod_props[cate])/len(hybrid_mod_props[cate])) for cate in object_maps)} 
    plot_prop_graph(proportions)


def calc_stats(real, wgan, hybrid_trad, hybrid_mod, keys, meta):
    id = keys.index('essentials')
    n_items = meta['essentials']['max']
    real_count = [tf.where(real[:,:,:,id]==i+1).shape[0]/real.shape[0] for i in range(n_items)]
    wgan_count = [tf.where(wgan[:,:,:,id]==i+1).shape[0]/wgan.shape[0] for i in range(n_items)]
    hybrid_trad_count = [tf.where(hybrid_trad[:,:,:,id]==i+1).shape[0]/hybrid_trad.shape[0] for i in range(n_items)]
    hybrid_mod_count = [tf.where(hybrid_mod[:,:,:,id]==i+1).shape[0]/hybrid_mod.shape[0] for i in range(n_items)]
    for cate in object_maps:
        min = meta[cate]['min']
        max = meta[cate]['max']
        real_cate = real_count[min:max]
        wgan_cate = wgan_count[min:max]
        hybrid_trad_cate = hybrid_trad_count[min:max]
        hybrid_mod_cate = hybrid_mod_count[min:max]
        fig, ax = plt.subplots(1, 1)
        plt.xlabel(cate + ' type')
        plt.ylabel('Average object population')
        plt.plot(list(range(1,max-min+1)), real_cate, label = "Real")
        plt.plot(list(range(1,max-min+1)), wgan_cate, label = "WGAN")
        plt.plot(list(range(1,max-min+1)), hybrid_trad_cate, label = "Hybrid-Trad CAN")
        plt.plot(list(range(1,max-min+1)), hybrid_mod_cate, label = "Hybrid-Mod CAN")
        ax.set_xlim(1, max-min)
        if cate == 'monsters': ax.set_ylim(0, 80)
        ax.xaxis.get_major_locator().set_params(integer=True)
        plt.legend() # must be after labels
        # plt.show()
        plt.savefig('./eval_metrics/'+cate+'_count')
        plt.close()


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


def calc_spatial_homogenity(maps,threshold=1,gen=True):
    floor_id = keys.index('floormap')
    things_id = keys.index('essentials')
    position = [n for n in range(256)]
    for i in range(maps.shape[0]):
        map = (maps[i,:,:,floor_id]>0).astype(int)
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


def plot_RipleyK(real, wgan, hybrid_trad, hybrid_mod):  
    test_radii = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]               
    rlevel_areas, rlevel_ripk = calc_RipleyK(real, keys, test_size, test_radii)
    wlevel_areas, wlevel_ripk = calc_RipleyK(wgan, keys, test_size, test_radii)
    htlevel_areas, htlevel_ripk = calc_RipleyK(hybrid_trad, keys, test_size, test_radii)
    hmlevel_areas, hmlevel_ripk = calc_RipleyK(hybrid_mod, keys, test_size, test_radii)
    
    # accr_ripk = [np.array([ripk[i] for ripk in rlevel_ripk]) * np.array(rlevel_areas) for i in range(len(test_radii))]
    # acch_ripk = [np.array([ripk[i] for ripk in hlevel_ripk]) * np.array(hlevel_areas) for i in range(len(test_radii))]
    # accw_ripk = [np.array([ripk[i] for ripk in wlevel_ripk]) * np.array(wlevel_areas) for i in range(len(test_radii))]


    # print('real: {} hybrid: {} wgan: {}'.format(sum([sum([ripk[i] for ripk in np.array(accr_ripk).T.tolist()])/len(rlevel_areas) for i in range(len(test_radii))])/len(test_radii), 
    #                                             sum([sum([ripk[i] for ripk in np.array(acch_ripk).T.tolist()])/len(hlevel_areas) for i in range(len(test_radii))])/len(test_radii), 
    #                                             sum([sum([ripk[i] for ripk in np.array(accw_ripk).T.tolist()])/len(wlevel_areas) for i in range(len(test_radii))])/len(test_radii)))
    
    
    print('real: {} wgan: {} hybrid_trad: {} hybrid_mod: {}'.format([sum([ripk[i] for ripk in rlevel_ripk])/len(rlevel_areas) for i in range(len(test_radii))], 
                                                [sum([ripk[i] for ripk in wlevel_ripk])/len(wlevel_areas) for i in range(len(test_radii))],
                                                [sum([ripk[i] for ripk in htlevel_ripk])/len(htlevel_areas) for i in range(len(test_radii))], 
                                                [sum([ripk[i] for ripk in hmlevel_ripk])/len(hmlevel_areas) for i in range(len(test_radii))]))
    
    for i,r in enumerate(test_radii):

        real_ripk = [ripk[i] for ripk in rlevel_ripk]
        wgan_ripk = [ripk[i] for ripk in wlevel_ripk]
        trad_can_ripk = [ripk[i] for ripk in htlevel_ripk]
        mod_can_ripk = [ripk[i] for ripk in hmlevel_ripk]
        level_ripk = [real_ripk, wgan_ripk, trad_can_ripk, mod_can_ripk]
        labels = ['Real', 'WGAN', 'Hybrid - Trad CAN', 'Hybrid - Mod CAN']

        fig = plt.figure(figsize=(10,6))
        fig.subplots(1, 4, sharey=True)
        # plt.ylabel('Probability')
        # plt.xlabel('Ripley K with radius '+ str(r))
        for i,ripk in enumerate(level_ripk):
            plt.subplot(1,4,i+1)
            n = len(ripk)
            IQR = np.percentile(ripk, 75) - np.percentile(ripk, 25)
            bin_width = 2*IQR*n**(-1/3)
            n_bins = round((max(ripk) - min(ripk)) / bin_width)
            plt.hist(ripk, bins=n_bins)
            plt.title(labels[i])

        fig.text(0.5, 0.05, 'Ripley K with radius '+ str(r), ha="center", va="center")
        fig.text(0.08, 0.5, "Count", ha="center", va="center", rotation=90)
        # fig.tight_layout(pad=0.4)
        # plt.show()
        plt.savefig('./eval_metrics/ripk'+str(int(r*10)))
        plt.close()


if __name__ == "__main__":
    b_size = 100
    z_dim = 100
    test_size = 100

    trad_wgen = wganGen(4, z_dim)
    checkpoint_dir = './training_checkpoints/wgan'
    checkpoint = tf.train.Checkpoint(generator=trad_wgen)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    hybrid_wgen = wganGen(3, z_dim)
    checkpoint_dir = './training_checkpoints/hybrid/wgan'
    checkpoint = tf.train.Checkpoint(generator=hybrid_wgen)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    hybrid_cgen = canGen(len(topological_maps),len(object_maps))
    checkpoint_dir = './training_checkpoints/hybrid/mod_can'
    checkpoint = tf.train.Checkpoint(generator=hybrid_cgen)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    hybrid_trad_cgen = canGen(len(topological_maps),len(object_maps))
    checkpoint_dir = './training_checkpoints/hybrid/trad_can'
    checkpoint = tf.train.Checkpoint(generator=hybrid_trad_cgen)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    training_set, validation_set, map_meta, sample= read_record(batch_size=b_size)
    for i, data in training_set.enumerate().as_numpy_iterator():
        z = tf.random.normal([b_size, z_dim])
        noise = tf.random.normal([b_size, 256, 256, 1])
        wgan_maps, keys, n_items = wgan_fmaps(trad_wgen, z, map_meta)
        hybrid_trad_maps, keys, n_items = hybrid_fmaps(hybrid_wgen, hybrid_trad_cgen, z, noise, map_meta)
        hybrid_maps, keys, n_items = hybrid_fmaps(hybrid_wgen, hybrid_cgen, z, noise, map_meta)
        real_maps = np.stack([data[m] for m in keys], axis=-1)
        r_maps = np.concatenate((r_maps,real_maps), axis=0) if i != 0 else real_maps
        w_maps = np.concatenate((w_maps,wgan_maps.numpy()), axis=0) if i != 0 else wgan_maps.numpy()
        ht_maps = np.concatenate((ht_maps,hybrid_trad_maps.numpy()), axis=0) if i != 0 else hybrid_trad_maps.numpy()
        hm_maps = np.concatenate((hm_maps,hybrid_maps.numpy()), axis=0) if i != 0 else hybrid_maps.numpy()
        if i==2: break
    # calc_spatial_homogenity(w_maps)
    calc_proportions(r_maps,w_maps,ht_maps,hm_maps,keys,map_meta)
    calc_stats(r_maps,w_maps,ht_maps,hm_maps,keys,map_meta)
    plot_RipleyK(r_maps,w_maps,ht_maps,hm_maps)
    plot_train_metrics()
    
    

