import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from HybridGen import hybrid_fmaps
from WganGen import wgan_fmaps
from GanMeta import read_record


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


# def calc_RipleyK(real, wgan, hybrid):                 
#     floor_id = keys.index('floormap')
#     things_id = keys.index('essentials')
#     position = [n for n in range(256)]
#     for i in range(real.shape[0]):
#         x_min = min(position, key= lambda x: real[i,x,:,floor_id])
#         x_max = max(position, key= lambda x: real[i,x,:,floor_id])
#         y_min = min(position, key= lambda y: real[i,:,y,floor_id])
#         y_max = max(position, key= lambda y: real[i,:,y,floor_id])
#         area = (x_max - x_min)*(y_max - y_min)
#         radii = np.linspace(0,min(x_max - x_min, y_max - y_min),50).reshape(50,1)
        
#         samples = np.transpose(np.nonzero(real[i,:,:,things_id]))
#         npts = np.shape(samples)[0]

#         diff = np.zeros(shape = (npts*(npts-1)//2,2))               
#         k = 0
#         for j in range(npts - 1):
#             size = npts - j - 1
#             diff[k:k + size] = abs(samples[j] - samples[j+1:])
#             k += size
        
#         n_ripley = np.zeros(len(radii))
#         distances = np.hypot(diff[:,0], diff[:,1])

#         for r in range(len(radii)):
#             n_ripley[r] = (distances<radii[r]).sum()

#         n_ripley = area * 2. * n_ripley / (npts * (npts - 1))


if __name__ == "__main__":
    b_size = 100
    dataset, map_meta, sample = read_record(batch_size=b_size)
    i = 0
    for data in dataset:
        z = tf.random.normal([b_size, 100])
        noise = tf.random.normal([b_size, 256, 256, 1])
        hybrid_maps, keys, n_items = hybrid_fmaps(z,noise) # generate each set of feature maps individually and stack them with axis =0
        wgan_maps, keys, n_items = wgan_fmaps(z)
        real_maps = np.stack([data[m] for m in keys], axis=-1)
        r_maps = tf.concat([r_maps,real_maps], axis=0) if i != 0 else real_maps
        h_maps = tf.concat([h_maps,hybrid_maps], axis=0) if i != 0 else hybrid_maps
        w_maps = tf.concat([w_maps,wgan_maps], axis=0) if i != 0 else wgan_maps
        i+=1
        break
    calc_proportions(real_maps,wgan_maps,hybrid_maps,keys,map_meta)
    calc_stats(real_maps,wgan_maps,hybrid_maps,keys,map_meta)



