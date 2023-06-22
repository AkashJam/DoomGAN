import sys
sys.path.insert(0,'..')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gan.cgan import Generator as cganGen
from gan.wgan import Generator as wganGen
from DOOMLevelGen import hybrid_fmaps, wgan_fmaps
from gan.DataProcessing import read_record
import os, json
from gan.NetworkArchitecture import topological_maps, object_maps
from gan.eval_metrics.metrics import level_props, calc_RipleyK


def plot_train_metrics(save_path = 'eval_metrics/'):
    file_path = save_path + 'training_metrics.json'
    if not os.path.isfile(file_path):
        print('No metric record found')
    else:
        with open(file_path, 'r') as jsonfile:
            metrics = json.load(jsonfile)
            GAN_types = list(metrics.keys())
            cGAN_types = [n for n in GAN_types if "cGAN" in n]
            for key in list(metrics[cGAN_types[0]].keys()):
                if 'loss' not in key:
                    for cGAN_type in cGAN_types:
                        steps = [stp*1023 for stp in list(range(len(metrics[cGAN_type][key])))]
                        plt.xlabel('Number of Steps')
                        plt.ylabel(key)
                        plt.plot(steps, metrics[cGAN_type][key] , label=cGAN_type)
                    plt.legend(loc="upper right")
                    if key=='out_of_bounds_err': plt.ylim(0, 1000)
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
    # plt.show()
    plt.savefig('./eval_metrics/cate_props')
    plt.close()


def calc_proportions(real, wgan, hybrid_trad, hybrid_mod, keys, meta):
    real_props = level_props(real, keys, meta, test_size)
    wgan_props = level_props(wgan, keys, meta, test_size)
    hybrid_trad_props = level_props(hybrid_trad, keys, meta, test_size)
    hybrid_mod_props = level_props(hybrid_mod, keys, meta, test_size)

    proportions = {'Real': tuple(round(100*sum(real_props[cate])/len(real_props[cate])) for cate in object_maps), 
                    'WGAN-GP': tuple(round(100*sum(wgan_props[cate])/len(wgan_props[cate])) for cate in object_maps), 
                    'Traditional cGAN': tuple(round(100*sum(hybrid_trad_props[cate])/len(hybrid_trad_props[cate])) for cate in object_maps),
                    'Modified cGAN': tuple(round(100*sum(hybrid_mod_props[cate])/len(hybrid_mod_props[cate])) for cate in object_maps)} 
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
        plt.plot(list(range(1,max-min+1)), wgan_cate, label = "WGAN-GP")
        plt.plot(list(range(1,max-min+1)), hybrid_trad_cate, label = "Traditional cGAN")
        plt.plot(list(range(1,max-min+1)), hybrid_mod_cate, label = "Modified cGAN")
        ax.set_xlim(1, max-min)
        if cate == 'monsters': ax.set_ylim(0, 80)
        ax.xaxis.get_major_locator().set_params(integer=True)
        plt.legend() # must be after labels
        # plt.show()
        plt.savefig('./eval_metrics/'+cate+'_count')
        plt.close()


def plot_RipleyK(real, wgan, hybrid_trad, hybrid_mod):  
    test_radii = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]               
    rlevel_areas, rlevel_ripk = calc_RipleyK(real, keys, test_size, test_radii)
    wlevel_areas, wlevel_ripk = calc_RipleyK(wgan, keys, test_size, test_radii)
    htlevel_areas, htlevel_ripk = calc_RipleyK(hybrid_trad, keys, test_size, test_radii)
    hmlevel_areas, hmlevel_ripk = calc_RipleyK(hybrid_mod, keys, test_size, test_radii)
    
    print('real: {} wgan: {} hybrid_trad: {} hybrid_mod: {}'.format([sum([ripk[i] for ripk in rlevel_ripk])/len(rlevel_areas) for i in range(len(test_radii))], 
                                                [sum([ripk[i] for ripk in wlevel_ripk])/len(wlevel_areas) for i in range(len(test_radii))],
                                                [sum([ripk[i] for ripk in htlevel_ripk])/len(htlevel_areas) for i in range(len(test_radii))], 
                                                [sum([ripk[i] for ripk in hmlevel_ripk])/len(hmlevel_areas) for i in range(len(test_radii))]))
    
    for i,r in enumerate(test_radii):

        real_ripk = [ripk[i] for ripk in rlevel_ripk]
        wgan_ripk = [ripk[i] for ripk in wlevel_ripk]
        trad_cgan_ripk = [ripk[i] for ripk in htlevel_ripk]
        mod_cgan_ripk = [ripk[i] for ripk in hmlevel_ripk]
        level_ripk = [real_ripk, wgan_ripk, trad_cgan_ripk, mod_cgan_ripk]
        labels = ['Real', 'WGAN-GP', 'Traditional cGAN', 'Modified cGAN']

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
        plt.savefig('./eval_metrics/ripk_'+str(int(r*10)))
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

    hybrid_mod_cgen = cganGen(len(topological_maps),len(object_maps))
    checkpoint_dir = './training_checkpoints/hybrid/mod_cgan'
    checkpoint = tf.train.Checkpoint(generator=hybrid_mod_cgen)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    hybrid_trad_cgen = cganGen(len(topological_maps),len(object_maps))
    checkpoint_dir = './training_checkpoints/hybrid/trad_cgan'
    checkpoint = tf.train.Checkpoint(generator=hybrid_trad_cgen)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    training_set, validation_set, map_meta, sample= read_record(batch_size=b_size)
    for i, data in training_set.enumerate().as_numpy_iterator():
        z = tf.random.normal([b_size, z_dim])
        noise = tf.random.normal([b_size, 256, 256, 1])
        wgan_maps, keys = wgan_fmaps(trad_wgen, z, map_meta)
        hybrid_trad_maps, keys = hybrid_fmaps(hybrid_wgen, hybrid_trad_cgen, z, noise, map_meta)
        hybrid_mod_maps, keys = hybrid_fmaps(hybrid_wgen, hybrid_mod_cgen, z, noise, map_meta)
        real_maps = np.stack([data[m] for m in keys], axis=-1)
        r_maps = np.concatenate((r_maps,real_maps), axis=0) if i != 0 else real_maps
        w_maps = np.concatenate((w_maps,wgan_maps.numpy()), axis=0) if i != 0 else wgan_maps.numpy()
        ht_maps = np.concatenate((ht_maps,hybrid_trad_maps.numpy()), axis=0) if i != 0 else hybrid_trad_maps.numpy()
        hm_maps = np.concatenate((hm_maps,hybrid_mod_maps.numpy()), axis=0) if i != 0 else hybrid_mod_maps.numpy()
        if i==1: break
    calc_proportions(r_maps,w_maps,ht_maps,hm_maps,keys,map_meta)
    calc_stats(r_maps,w_maps,ht_maps,hm_maps,keys,map_meta)
    plot_RipleyK(r_maps,w_maps,ht_maps,hm_maps)
    plot_train_metrics()
    
    

