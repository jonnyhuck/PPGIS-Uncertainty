""" 
This version is for a sensitivity analysis of different bandwiths (neighbourhood radii)
"""
from math import ceil
from sys import float_info
from pandas import read_csv
from scipy.spatial import cKDTree
from rasterio import open as rio_open
from rasterio.transform import from_origin
from numpy import zeros, meshgrid, arange, sqrt, where
from geopandas import GeoDataFrame, points_from_xy, read_file
from ling_rudd2 import Participant, lr_combination, compute_plausibility

def idw(d, r):
    """ Calculate IDW Weighting """
    return 1 - d / r

def get_topics(topics_list):
    """ Convert a Series of comma-separated topics into a set of unique topics """
    topics_list = topics_list[topics_list.notnull()].to_list()  # drop any NAs
    return { t for topics in topics_list for t in topics.split(",")} if len(topics_list) > 0 else {}

def get_topics_string(topics_list):
    """ Convert a Series of comma-separated topics into a comma-separated string of unique topics """
    return ','.join(get_topics(topics_list))


''' SETTINGS '''
PREVIEW = False         # do you want to preview the results?
OUTPUT = True           # do you want to write the results to GeoTiff?
RESOLUTION = 25         # dataset resolution

# load blob and answer data from Map-Me
blobs = read_csv('../data/blobs.csv')
answers = read_csv('../data/answers_with_terms.csv')[['id_person', 'id_subquestion', 'spray_number', 'certainty', 'terms']] # calculated using topics.py

# filter blobs and answers to relevant question 
blobs = blobs[blobs.id_question.isin([21395, 21396])]
answers = answers.loc[answers.id_subquestion.isin([27468.0, 27469.0])] 

# update answers to reflect the question instead of subquestion
answers['id_question'] = where(answers['id_subquestion'] == 27468.0, 21395, 21396)
answers.drop('id_subquestion', axis=1)

# join answers to blobs
blobs = blobs.merge(answers, how='left', on=['id_person', 'spray_number', 'id_question'])

# convert to geodataframe
gdf = GeoDataFrame(blobs, geometry=points_from_xy(blobs['lng'], blobs['lat']), crs='EPSG:4326').to_crs('EPSG:27700').cx[331000:336500, 505700:509100]

# enforce Cromwell's rule
gdf['certainty'] = where(gdf['certainty'] == 1, 1-float_info.epsilon, gdf['certainty'])
gdf['certainty'] = where((gdf['certainty'] == 0), float_info.epsilon, gdf['certainty'])

# drop data inside the lakes
lakes = read_file('data/lakes.shp')
gdf = gdf[~gdf.geometry.within(lakes.union_all())]

# get total participant count (and m value if required)
total_participants = len(gdf['id_person'].unique())
print(f"participant count: {total_participants}")

# rescale certainty
gdf['certainty'] /= 10

# get bounds for output raster
bounds = gdf.total_bounds
width = ceil((bounds[2] - bounds[0]) / RESOLUTION)
height = ceil((bounds[3] - bounds[1]) / RESOLUTION)

# create affine transformation object
affine = from_origin(bounds[0], bounds[3], RESOLUTION, RESOLUTION)

# load point data into cKDTree
tree_kdtree = cKDTree([(geom.x, geom.y) for geom in gdf.geometry])

# create grid of cell center points
x_vals = arange(bounds[0] + RESOLUTION / 2, bounds[2], RESOLUTION)
y_vals = arange(bounds[3] - RESOLUTION / 2, bounds[1], -RESOLUTION)
grid_x, grid_y = meshgrid(x_vals, y_vals)

# flatten grid points 
grid_points = list(zip(grid_x.ravel(), grid_y.ravel()))

# loop througn possible radii (already have 150...)
for RADIUS in [50, 100, 200, 250]:
    print(f"Processing {RADIUS}m radius...")

    # create empty raster datasets
    scores = zeros((8, height, width))

    # query kdtree for ID of all points within radius of each location
    indices = tree_kdtree.query_ball_point(grid_points, RADIUS)

    # for each cell in the grid
    for idx, (r, c) in enumerate(zip(*divmod(arange(len(grid_points)), grid_x.shape[1]))):
        selected_points = indices[idx]

        # get the points from the geodataframe
        dots = gdf.iloc[selected_points].copy()

        # get the topics of information on which the participants' views are based
        all_sources = get_topics(dots.terms)

        # calculate IDW weights (and enforce Cromwell's rule)
        distances = sqrt((grid_points[idx][0] - dots.geometry.x)**2 + (grid_points[idx][1] - dots.geometry.y)**2)
        dots['gw'] = idw(distances, RADIUS)
        dots['gw'] = where(dots['gw'] == 1, 1-float_info.epsilon, dots['gw'])
        dots['gw'] = where((dots['gw'] == 0), float_info.epsilon, dots['gw'])

        # separate evidence for trees and no trees (mean certainty and max score - which is the min distance)
        grouped = dots[['id_person', 'id_question', 'certainty', 'gw', 'terms']].groupby(['id_person', 'id_question']).agg(
            {'certainty':'mean', 'gw': 'max', 'terms':get_topics_string})

        # if we have no data, just set 0 and go to next
        if grouped.empty:
            scores[0, r, c] = 0
            scores[1, r, c] = 0
            scores[2, r, c] = 0
            scores[3, r, c] = 0
            continue

        # loop through each participant
        participants = []
        for id_participant in grouped.index.get_level_values(0).unique():

            # init bpa
            bpa = {}
            ignorance = []
            sources = set()

            # get their trees mass
            trees = grouped.loc[(grouped.index.get_level_values(0) == id_participant) & (grouped.index.get_level_values(1) == 21395)]
            if not trees.empty:
                bpa[frozenset({'trees'})] = trees.gw.iloc[0]
                s = trees.certainty.iloc[0]
                sources.union(get_topics(trees.terms))
            
            # get their no trees mass
            no_trees = grouped.loc[(grouped.index.get_level_values(0) == id_participant) & (grouped.index.get_level_values(1) == 21396)]
            if not no_trees.empty:
                bpa[frozenset({'no trees'})] = no_trees.gw.iloc[0]
                s = no_trees.certainty.iloc[0]  # this will overwrite if both are specified, but that's OK as it's the same value
                sources.union(get_topics(no_trees.terms))

            # rescale to sum 0-1 (including the uncertainty that isn't in there)
            if len(bpa) == 2:
                bpa[frozenset({'trees'})] /= 2
                bpa[frozenset({'no trees'})] /= 2

            # add participant
            participants.append(Participant(bpa, s, sources))
        
        # combine using modified Ling & Rudd
        m = lr_combination(participants)

        # update cells with belief & plausibility
        scores[0, r, c] = m[frozenset({'trees'})]
        scores[1, r, c] = m[frozenset({'no trees'})]
        scores[2, r, c] = compute_plausibility(m, frozenset({'trees'}))
        scores[3, r, c] = compute_plausibility(m, frozenset({'no trees'}))

    # add DS uncertainty (plausibility - belief)
    scores[4] = scores[2] - scores[0]
    scores[5] = scores[3] - scores[1]

    # convert to probability (pignistic transformation)
    scores[6] = scores[0] + (scores[4] / 2)
    scores[7] = scores[1] + (scores[5] / 2)

    # preview the result
    if PREVIEW:
        from matplotlib import pyplot as plt
        my_fig, my_axs = plt.subplots(4, 2, figsize=(5, 6))
        my_axs[(0, 0)].imshow(scores[0])
        my_axs[(0, 0)].set_title("Trees Belief")

        my_axs[(0, 1)].imshow(scores[1])
        my_axs[(0, 1)].set_title("No Trees Belief")

        my_axs[(1, 0)].imshow(scores[2])
        my_axs[(1, 0)].set_title("Trees Plausibility")

        my_axs[(1, 1)].imshow(scores[3])
        my_axs[(1, 1)].set_title("No Trees Plausibility")

        my_axs[(2, 0)].imshow(scores[4])
        my_axs[(2, 0)].set_title("Trees Uncertainty")

        my_axs[(2, 1)].imshow(scores[5])
        my_axs[(2, 1)].set_title("No Trees Uncertainty")

        my_axs[(2, 0)].imshow(scores[6])
        my_axs[(2, 0)].set_title("Trees Probability")

        my_axs[(2, 1)].imshow(scores[7])
        my_axs[(2, 1)].set_title("No Trees Probability")

        # turn off axes
        for r in range(my_axs.shape[0]):
            for c in range(my_axs.shape[1]):
                my_axs[(r,c)].set_axis_off()
        plt.show()

    if OUTPUT:
        # write scores to raster
        with rio_open(f'../sensitivity/{RADIUS}m_radius.tif', 'w', driver='GTiff',
                    height=scores.shape[1], width=scores.shape[2], count=scores.shape[0],
                    dtype=scores.dtype, crs='EPSG:27700', transform=affine) as dataset:
            for i in range(scores.shape[0]):
                dataset.write(scores[i], i+1)
    print('done.')