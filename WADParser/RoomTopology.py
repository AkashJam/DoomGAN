from skimage import filters, color, io
from skimage.future import graph as skg
from skimage.segmentation import watershed
from skimage.measure import regionprops, label, find_contours
from skimage.transform import resize
from skimage.feature import peak_local_max, corner_harris, corner_peaks
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.signal import sawtooth
from scipy.stats import entropy, describe
import networkx as nx
import numpy as np
import warnings
import itertools

def vertices_to_segment_list(vertices):
    if vertices[0] != vertices[-1]:
        vertices += [vertices[0]]
    startiter, enditer = itertools.tee(vertices, 2)
    next(enditer, None)  # shift the end iterator by one
    return zip(startiter, enditer)

def encoding_error(x, interval):
    """
    Returns the encoding error for the value x, considering a given encoding interval.
    Returned error range from 0 to 1.
    Example:
    If the only two meaningful ("correct") values of a pixel are 0 and 255, then the encoding interval (distance between
     two consecutive meaningful values) is 255, and
     encoding_error(0, 255) = 0
     encoding_error(127.5, 255) = 1
     encoding_error(0, 255) = 0

    Example:
     encoding_error(x, 1) = 0 iff x is an integer, 1 if x = k+0.5 for k in Z

    WARNING: This function is not precise if the magnitude of x is more then one order above the magnitude of interval
    and the effect is the more visible the more the two orders differ
    E.G
    encoding_error(10, 1) = 0
    encoding_error(11, 1) = 2.2759572004815709e-15 != 0
    encoding_error(1e8, 1) = 2.2759572004815709e-15 != 0
    encoding_error(1e14, 1) = 0.00057308744474998674 != 0


    :param x: the value for which calculate the error
    :param interval: the interval between two values for which (error = 0)
    :return:
    """
    #if np.greater(np.abs(x), 10*interval).any():
    #    warnings.warn("Encoding error might not be accurate due to approximations", Warning)
    return (sawtooth(2 * np.pi * x / interval, 0.5) + 1) / 2



def _reverse_wall(wall):
    """
    Return the same wall with swapped extrema:
    :param wall:  (((x1, y1), (x2, y2)), destination_node)
    :return: (((x2, y2), (x1, y1)), destination_node)
    """
    return tuple(reversed(wall[0])), wall[1]

def create_graph(floormap, return_dist=False, room_coordinates=False):
    """
    Segment the floormap into rooms and create a Region Adjacency Graph for the level.
    Many settings for decorating the graph are possible, by default the simplest form is returned.
    :param floormap: Binary image representing the level floor
    :param return_dist: If true, also returns the distance map of each point to the closest wall.
    :param room_coordinates: If true, each graph node will contain the room vertices and information about the walls
    :return: (Roommap, Graph) if return_dist is False, (Roommap, Graph, dist) otherwise
    """
    # Ensuring that floormap is always a boolean array
    floormap = floormap.astype(np.bool)
    #floormap = rescale(floormap, 2)
    dist = ndi.distance_transform_edt(floormap)
    threshold = int(dist.max())
    optimal_threshold = 0
    number_of_centers = 0
    # Finding room center and finding the optimal threshold (the one that maximizes the number of rooms)
    for i in range(int(dist.max()), int(dist.min())-1,-1):
        peak_mask = peak_local_max(dist, threshold_abs=threshold-i, labels=floormap, min_distance=3)
        local_max = np.zeros_like(dist, dtype=bool)
        local_max[tuple(peak_mask.T)] = True
        markers = ndi.label(local_max)[0]
        if markers.max() > number_of_centers:
            optimal_threshold = threshold-i
            number_of_centers = markers.max()

    # Computing roommap with the optimal threshold
    peak_mask = peak_local_max(dist, min_distance=3, labels=floormap, threshold_abs=optimal_threshold)
    local_max = np.zeros_like(dist, dtype=bool)
    local_max[tuple(peak_mask.T)] = True
    markers = ndi.label(local_max)[0]
    roommap = watershed(-dist, markers, mask=floormap)
    # plt.imshow(roommap)
    # io.show()
    
    # Find edges in an image
    edges = filters.sobel(roommap)
    room_RAG_boundaries = skg.rag_boundary(roommap, edges)

    # View graph
    # roommapRGB = color.gray2rgb(edges)
    # lc = skg.show_rag(roommap, room_RAG_boundaries, roommapRGB, img_cmap=None, edge_cmap='viridis', edge_width=1.2)
    # plt.colorbar(lc, fraction=0.03)
    # io.show()
    

    if room_coordinates:
        # For each floor...
        floors = label(floormap)
        for floor_id in range(max(1, floors.min()), floors.max() + 1):  # Skipping label 0 (background)
            # Building the wall list for floor boundaries
            # Here the map is upsampled by a factor 2 before finding the contours, then coordinates are divided by two.
            # This is for avoiding "X" shaped connections between rooms due to how find_contours work
            floor_contour = find_contours(resize(floors == floor_id, (floors.shape[0]*2, floors.shape[1]*2), order=0), 0.5, positive_orientation='low')[0] / 2
            walls_vertices = [tuple(v) for v in floor_contour]
            floor_boundaries = tuple(vertices_to_segment_list(walls_vertices))
            # Map of rooms belonging to current floor
            rooms = roommap * (floors == floor_id)
            for room_id in range(max(1, rooms.min()), rooms.max() + 1):  # Skipping label 0 (background)
                if room_id not in rooms:
                    # Some room id may be in another floor, if they are enumerated horizontally
                    continue
                # Here the map is upsampled by a factor 2 before finding the contours, then coordinates are divided by two.
                # This is for avoiding "X" shaped connections between rooms due to how find_contours work
                room_contour = find_contours(resize(rooms == room_id, (rooms.shape[0]*2, rooms.shape[1]*2), order=0), 0.5, fully_connected='high', positive_orientation='low')[0] / 2
                rooms_vertices = [tuple(v) for v in room_contour]
                room_boundaries = tuple(vertices_to_segment_list(rooms_vertices))


                room_RAG_boundaries.nodes[room_id]['walls'] = list()
                for segment in room_boundaries:
                    leads_to = 0 if segment in floor_boundaries else None # We cannot still know edges for other rooms but background
                    room_RAG_boundaries.nodes[room_id]['walls'].append((segment, leads_to))

            # Here we still miss the relation between boundary and edges.
            # Second pass
            for room_id in range(max(1, rooms.min()), rooms.max() + 1):
                if room_id not in rooms:
                    # Some room id may be in another floor, if they are enumerated horizontally
                    continue
                boundaries_current = {wall for wall in room_RAG_boundaries.nodes[room_id]['walls'] if wall[1] is None}
                for neigh in room_RAG_boundaries.adj[room_id]:
                    if neigh == 0:
                        continue
                    # Finding the neighbour boundaries. We must consider both directions for each vertex
                    boundaries_neigh = {wall for wall in room_RAG_boundaries.nodes[neigh]['walls'] if wall[1] is None}
                    boundaries_neigh_reverse = {_reverse_wall(wall) for wall in room_RAG_boundaries.nodes[neigh]['walls'] if wall[1] is None}

                    common_segments = boundaries_current.intersection(boundaries_neigh)
                    common_segments_reversed = boundaries_current.intersection(boundaries_neigh_reverse)
                    # Marking the boundary in the two nodes with the destination node
                    # Each node will contain the list
                    for cs in common_segments:
                        i_current = room_RAG_boundaries.nodes[room_id]['walls'].index(cs)
                        i_neighbour = room_RAG_boundaries.nodes[neigh]['walls'].index(cs)
                        room_RAG_boundaries.nodes[room_id]['walls'][i_current] = (cs[0], neigh)
                        room_RAG_boundaries.nodes[neigh]['walls'][i_neighbour] = (cs[0], room_id)
                    # Same thing in the case of reversed segments
                    for cs in common_segments_reversed:
                        rev_cs = _reverse_wall(cs)
                        i_current = room_RAG_boundaries.nodes[room_id]['walls'].index(cs)
                        i_neighbour = room_RAG_boundaries.nodes[neigh]['walls'].index(rev_cs)
                        room_RAG_boundaries.nodes[room_id]['walls'][i_current] = (cs[0], neigh)
                        room_RAG_boundaries.nodes[neigh]['walls'][i_neighbour] = (rev_cs[0], room_id)
    if return_dist:
        return roommap, room_RAG_boundaries, dist
    return roommap, room_RAG_boundaries

def topological_features(floormap, prepare_for_doom=False):
    """
    Create the level graph from the floormap and compute some topological features on the graph.
    :param floormap:
    :param prepare_for_doom: (Default:False) If true each node will also contain vertices and walls information for converting the level to a WAD file.
    :return: (room map, room_graph, dict of metrics)
    """
    roommap, room_graph, dist = create_graph(floormap, return_dist=True, room_coordinates=prepare_for_doom)
    room_props = regionprops(roommap)
    for r in range(roommap.max()):
        # Room Size
        room_graph.nodes[r+1]["area"] = room_props[r]["area"]
        room_graph.nodes[r+1]["perimeter"] = room_props[r]["perimeter"]
        mask = (roommap == r+1)
        max_dist = np.max(mask * dist)
        room_graph.nodes[r+1]["max_dist"] = max_dist
        room_graph.nodes[r+1]["centroid"] = room_props[r]["centroid"]


        # TODO: Add information about other maps, such as enemies, etc.

    centroid_distance = dict()
    for i, j in room_graph.edges():
        # Decorate the edges with the distance
        if i==0 or j == 0:
            continue
        centroid_distance[(i,j)] = np.linalg.norm(np.asarray(room_graph.nodes[i]["centroid"])-np.asarray(room_graph.nodes[j]["centroid"])).item()
    nx.set_edge_attributes(room_graph, name='centroid_distance', values=centroid_distance)

    # To compute correct metrics we need to remove node 0, which is the background
    graph_no_background = room_graph.copy()
    graph_no_background.remove_node(0)
    metrics = dict()
    # Computing metrics from "Predicting the Global Structure of Indoor Environments: A costructive Machine Learning Approach", (Luperto, Amigoni, 2018)
    #####
    metrics["nodes"] = len(nx.nodes(graph_no_background))
    pl_list = list()
    diam_list = list()
    assort_list = list()
    for cc in nx.connected_components(graph_no_background):
        if len(graph_no_background.subgraph(cc).edges()) > 0:
            pl_list += [nx.average_shortest_path_length(graph_no_background.subgraph(cc))]
            diam_list += [nx.diameter(graph_no_background.subgraph(cc))]
            assort_list += [nx.degree_assortativity_coefficient(graph_no_background)]

    metrics["avg-path-length"] = np.mean(pl_list) if len(pl_list) > 0 else 0
    metrics["diameter-mean"] = np.mean(diam_list) if len(diam_list) > 0 else 0
    metrics["art-points"] = len(list(nx.articulation_points(graph_no_background)))
    metrics["assortativity-mean"] = nx.degree_assortativity_coefficient(graph_no_background) if len(graph_no_background.subgraph(cc).edges()) > 0 else 0
    # print(metrics)
    try:
        # Centrality measures
        metrics["betw-cen"] = nx.betweenness_centrality(graph_no_background)
        metrics["closn-cen"] = nx.closeness_centrality(graph_no_background)
        # These metrics may throw exceptions
        # metrics["eig-cen"] = nx.eigenvector_centrality_numpy(graph_no_background)
        # metrics["katz-cen"] = nx.katz_centrality_numpy(graph_no_background)

        # Describing node stat distributions and removing them from the dict
        for met in ['betw-cen', 'closn-cen']:
            values = list(metrics['{}'.format(met)].values())
            st = describe(values)

            metrics["{}-min".format(met)] = st.minmax[0]
            metrics["{}-max".format(met)] = st.minmax[1]
            metrics["{}-mean".format(met)] = st.mean
            metrics["{}-var".format(met)] = st.variance
            metrics["{}-skew".format(met)] = st.skewness
            metrics["{}-kurt".format(met)] = st.kurtosis
            # Quartiles
            metrics["{}-Q1".format(met)] = np.percentile(values, 25)
            metrics["{}-Q2".format(met)] = np.percentile(values, 50)
            metrics["{}-Q3".format(met)] = np.percentile(values, 75)
            del metrics[met]
    except:
        warnings.warn("Unable to compute centrality for this level")
        metrics["betw-cen"] = np.nan
        metrics["closn-cen"] = np.nan
    #####

    # Metrics on distance map. Ignoring black space surrounding the level
    cleandist = np.where(dist == 0, np.nan, dist)
    dstat = describe(cleandist, axis=None, nan_policy='omit')
    metrics["distmap-max".format(met)] = dstat.minmax[1]
    metrics["distmap-mean".format(met)] = dstat.mean
    metrics["distmap-var".format(met)] = dstat.variance
    metrics["distmap-skew".format(met)] = dstat.skewness
    metrics["distmap-kurt".format(met)] = dstat.kurtosis
    # Quartiles
    metrics["distmap-Q1".format(met)] = np.percentile(values, 25)
    metrics["distmap-Q2".format(met)] = np.percentile(values, 50)
    metrics["distmap-Q3".format(met)] = np.percentile(values, 75)
    roommap = np.uint8(roommap) # converting to int to make it JSON serializeable
    return roommap, room_graph, metrics

def quality_metrics(sample, maps):
    """
    Compute map quality metrics (entropy, encoding error, number of corners) for the provided sample.
    :param sample: array of shape (width, height, len(maps))
    :param maps: Array of map names, eg. ['floormap', 'wallmap', ...]. The position must correspond to the third sample dimension
    :return: a dictionary of metrics
    """
    metrics = dict()
    for m, mname in enumerate(maps):
        hist = np.histogram(sample[:, :, m], bins=255, range=(0, 255), density=True)[0]
        metrics['entropy_{}'.format(mname)] = entropy(hist)
        if mname in ['floormap', 'wallmap']:
            metrics['encoding_error_{}'.format(mname)] = np.mean(encoding_error(sample[:,:,m], 255))
            metrics['corners_{}'.format(mname)] = len(corner_peaks(corner_harris(sample[:,:,m])))

        elif mname in ['heightmap', 'thingsmap', 'triggermap']:
            metrics['encoding_error_{}'.format(mname)] = np.mean(encoding_error(sample[:, :, m], 1))
    return metrics