import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os, json
from gan.NetworkArchitecture import object_maps
from gan.metrics import level_props, calc_RipleyK


def generate_loss_graph(train_loss, valid_loss, key, sfactor = 10, location = 'generated_maps/'):
    plt.xlabel('Number of Steps')
    plt.ylabel('Loss')
    smoothened_loss = list()
    for i in range(0, len(train_loss), sfactor):
        loss =  train_loss[i:i + sfactor]
        smoothened_loss.append(sum(loss)/len(loss))
    train_batch = [tb*sfactor for tb in list(range(len(smoothened_loss)+1))]
    valid_batch = [vb*len(train_loss)/len(valid_loss) for vb in list(range(len(valid_loss)+1))]
    train_loss = [0] + smoothened_loss
    valid_loss = [0] + valid_loss
    plt.plot(train_batch, train_loss, label="train")
    plt.plot(valid_batch, valid_loss, label="validation")
    plt.legend() # must be after labels
    if not os.path.exists(location):
        os.makedirs(location)
    plt.savefig(location + key + '_loss_graph')
    plt.close()


def plot_train_metrics(save_path = 'artifacts/eval_metrics/'):
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
    plt.savefig('artifacts/eval_metrics/cate_props')
    plt.close()


def calc_proportions(real, wgan, hybrid_trad, hybrid_mod, keys, meta, test_size):
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
        plt.savefig('artifacts/eval_metrics/'+cate+'_count')
        plt.close()


def plot_RipleyK(real, wgan, hybrid_trad, hybrid_mod, keys, test_size):  
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
        plt.savefig('artifacts/eval_metrics/ripk_'+str(int(r*10)))
        plt.close()