import tensorflow as tf
import ripleyk
import numpy as np
from NetworkArchitecture import object_maps
from scipy.stats import entropy



def encoding_error(objmaps, meta):
    enc_err = 0
    objmaps_rescaled = rescale_maps(objmaps, meta, object_maps)
    for i in range(objmaps.shape[0]):
        for j,cate in enumerate(object_maps):
            min = meta[cate]['min']
            max = meta[cate]['max']
            enc_err += tf.reduce_sum(tf.cast(objmaps_rescaled[i,:,:,j]>max,tf.float16))
            enc_err += tf.reduce_sum(tf.cast(tf.logical_and(objmaps_rescaled[i,:,:,j]<min,objmaps_rescaled[i,:,:,j]!=0),tf.float16))
    return enc_err/objmaps.shape[0]


def oob_error(thingsmap, floormap, obj_cate=True):
    oob_err = 0
    for i in range(thingsmap.shape[0]):
        not_floor_mask = tf.cast((floormap[i,:,:]<1),tf.float32)
        if obj_cate:
            for j,cate in enumerate(object_maps):
                oob_err += tf.reduce_sum(tf.cast(thingsmap[i,:,:,j]>0,tf.float32)*not_floor_mask)
        else:
            oob_err += tf.reduce_sum(tf.cast(thingsmap[i,:,:,0]>0,tf.float32)*not_floor_mask)
    return oob_err/thingsmap.shape[0]



def objs_per_unit_area(objmaps, floormap):
    objs_arr = 0
    for i in range(objmaps.shape[0]):
        floor_mask = tf.cast(floormap[i,:,:]>0,tf.float32)
        area = tf.reduce_sum(floor_mask)
        for j in range(len(object_maps)):
            objs_arr += tf.reduce_sum(tf.cast(objmaps[i,:,:,j]>0,tf.float32)*floor_mask)/area
    return objs_arr/objmaps.shape[0]


def mat_entropy(thingsmap, meta):
    etp = 0
    for i in range(thingsmap.shape[0]):
        for j in range(thingsmap.shape[3]):
            essentials = thingsmap[i,:,:,j] if j == 0 else tf.maximum(essentials, thingsmap[i,:,:,j])
        n_items = meta['essentials']['max']
        hist = np.histogram(essentials, bins=n_items, range=(0, 1), density=True)[0]
        etp += entropy(hist)
    return etp/thingsmap.shape[0]



def level_props(maps, keys, meta, test_size):
    id = keys.index('essentials')
    maps_props = dict()
    for i in range(maps.shape[0]):
        if tf.reduce_sum(tf.cast(maps[i,:,:,id]>0,tf.float16)) == 0:
            continue
        for cate in object_maps:
            min = meta[cate]['min']
            max = meta[cate]['max']
            map_props = tf.reduce_sum(tf.cast(tf.logical_and(maps[i,:,:,id]>min,maps[i,:,:,id]<=max),tf.float16)) / tf.reduce_sum(tf.cast(maps[i,:,:,id]>0,tf.float16))
            maps_props[cate] = [map_props] if cate not in list(maps_props.keys()) else maps_props[cate] + [map_props]
        if len(maps_props)==test_size: break
    return maps_props

def calc_RipleyK(maps, keys, test_size, radii):
    floor_id = keys.index('floormap')
    things_id = keys.index('essentials')
    position = [n for n in range(256)]
    level_areas = list()
    level_ripk = list()
    for i in range(maps.shape[0]):
        map = (maps[i,:,:,floor_id]>0).astype(int)
        f_area = np.sum(map)
        x_min = max(position, key= lambda x: map[x,:].any()*(256-x))
        x_max = max(position, key= lambda x: map[x,:].any()*x)
        y_min = max(position, key= lambda y: map[:,y].any()*(256-y))
        y_max = max(position, key= lambda y: map[:,y].any()*y)
        samples = np.transpose(np.nonzero(maps[i,:,:,things_id]*map))
        # Normalise values between -1 and 1
        rescaled_x = 2*(samples[:,0]-x_min-(x_max-x_min)/2)/(x_max-x_min)
        rescaled_y = 2*(samples[:,1]-y_min-(y_max-y_min)/2)/(y_max-y_min)
        if len(rescaled_x) == 0 or len(rescaled_y) == 0:
            continue
        for j,r in enumerate(radii):
            k = ripleyk.calculate_ripley(r, 1, d1=rescaled_x, d2=rescaled_y, boundary_correct=True, CSR_Normalise=True)
            ripk = [k] if j == 0 else ripk + [k]
        area = (x_max-x_min)*(y_max-y_min)
        level_areas = [f_area/area] if len(level_areas) == 0 else level_areas + [f_area/area]
        level_ripk = [ripk] if len(level_ripk) == 0 else level_ripk + [ripk]
        if len(level_ripk)==test_size: break
    return level_areas, level_ripk