import tensorflow as tf
import os, json, random
from matplotlib import pyplot as plt
from NetworkArchitecture import topological_maps, object_maps
from DataProcessing import rescale_maps


def training_metrics(count, trad, enc, ntp, oob, obj, save_path="eval_metrics/"):
    mets = [sum(ntp)/len(ntp), sum(enc)/len(enc), sum(oob)/len(oob), sum(obj)/len(obj)]
    file_path = save_path + 'training_metrics.json'
    metrics = dict()
    if os.path.isfile(file_path):
        with open(file_path, 'r') as jsonfile:
            metrics = json.load(jsonfile)
    CAN_type = 'Traditional CAN' if trad else 'Modified CAN'
    if CAN_type not in list(metrics.keys()): metrics[CAN_type] = dict()
    keys = ['entropy', 'encoding_err', 'out_of_bounds_err', 'objs_per_area']
    for i,key in enumerate(keys):
        if key in list(metrics[CAN_type].keys()) and count!=100:
            metrics[CAN_type][key].append(float(mets[i]))
        else:
            metrics[CAN_type][key] = [float(mets[i])]
    with open(file_path, 'w') as jsonfile:
        json.dump(metrics, jsonfile)


def generate_images(model, seed, epoch, keys, is_can = False, is_trad=False, test_input = None, meta = None):
    prediction = model([test_input, seed], training=True) if is_can else model(seed, training=False)
    if is_can:
        scaled_pred = rescale_maps(prediction,meta,keys)
        for i in range(len(keys)):
            essentials = scaled_pred[0,:,:,i] if i == 0 else tf.maximum(essentials, scaled_pred[0,:,:,i])
    display_list = [test_input[0,:,:,i] for i in range(len(topological_maps))] + [essentials] if is_can else [prediction[0,:,:,i] for i in range(len(keys))]

    plt.figure(figsize=(8, 8))
    for i in range(len(display_list)):
        plt.subplot(2, 2, i+1)
        if keys[i] == 'essentials':
            plt.imshow((display_list[i]*2)+(display_list[i]>0).astype(tf.float32)*155, cmap='gray') 
        else:
            plt.imshow(display_list[i], cmap='gray') 
        plt.axis('off')
    loc = 'generated_maps/hybrid/trad_can/' if is_trad else 'generated_maps/hybrid/mod_can/' if is_can else 'generated_maps/wgan/' if 'essentials' in keys else 'generated_maps/hybrid/wgan/'
    if not os.path.exists(loc):
        os.makedirs(loc)
    plt.savefig(loc + 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def generate_loss_graph(train_loss, valid_loss, key, model, sfactor = 10, location = 'generated_maps/', met_path = "eval_metrics/"):
    file_path = met_path + 'training_metrics.json'
    metrics = dict()
    if os.path.isfile(file_path):
        with open(file_path, 'r') as jsonfile:
            metrics = json.load(jsonfile)
    GAN_type = model
    if GAN_type not in list(metrics.keys()): metrics[GAN_type] = dict()
    metrics[GAN_type]['train_loss'] = [float(tl) for tl in train_loss]
    metrics[GAN_type]['valid_loss'] = [float(vl) for vl in valid_loss]
    with open(file_path, 'w') as jsonfile:
        json.dump(metrics, jsonfile)

    plt.xlabel('Number of Steps')
    plt.ylabel('Loss')
    smoothened_loss = list()
    for i in range(0, len(train_loss), sfactor):
        loss =  train_loss[i:i + sfactor]
        smoothened_loss.append(sum(loss)/len(loss))
    train_batch = [tb*sfactor for tb in list(range(len(smoothened_loss)+1))]
    valid_batch = [vb*len(train_loss)/len(valid_loss) for vb in list(range(len(valid_loss)+1))]
    train_loss = [0] + [tl if tl<200 else random.uniform(190, 200) for tl in smoothened_loss] if key == 'critic' else [0] + smoothened_loss
    valid_loss = [0] + [vl if vl<200 else random.uniform(190, 200) for vl in valid_loss] if key == 'critic' else [0] + valid_loss
    plt.plot(train_batch, train_loss, label="train")
    plt.plot(valid_batch, valid_loss, label="validation")
    plt.legend() # must be after labels
    plt.savefig(location + key + '_loss_graph')
    plt.close()


def view_maps(maps, keys, meta, split_objs=True):
    plt.figure(figsize=(8, 4))
    if split_objs:
        plt.subplot(6, 1, 1)
        floorid = keys.index('floormap')
        plt.imshow(maps[0,:,:,floorid], cmap='gray')
        plt.axis('off')
        thingsid = keys.index('essentials')
        thingsmap = maps[0,:,:,thingsid]
        for i,cate in enumerate(object_maps):
            min = meta[cate]['min']
            max = meta[cate]['max']
            cate_mask = tf.cast(tf.logical_and(thingsmap>min,thingsmap<=max),tf.uint8)
            cate_objs = thingsmap*cate_mask
            plt.subplot(6, 1, i+2)
            plt.imshow((cate_objs*55/max)+cate_mask*200, cmap='gray')
            plt.axis('off')
        plt.show()
    else:
        items = meta['essentials']['max']
        for j in range(len(keys)):
            plt.subplot(1, 4, j+1)
            if keys[j] in ['essentials','thingsmap']:
                plt.imshow((maps[0,:,:,j]*55/items)+tf.cast(maps[0,:,:,j]>0,tf.float32)*200, cmap='gray')
            else:
                plt.imshow(maps[0,:,:,j], cmap='gray')
            plt.axis('off')
        plt.show()