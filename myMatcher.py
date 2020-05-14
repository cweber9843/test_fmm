"""
Class around the fmm module.
https://github.com/cyang-kth/fmm
"""

##########
# 2do
# - move network preprocessing here (see load_network)
#

#################
# imports
import fmm as fmm
import numpy as np
import pandas as pd
import geopandas as gpd
import sqlalchemy as sqa
import shapely.geometry as sgeom
from shapely.geometry import Point, LineString, shape

import importlib

class myMatcher(object):
    """docstring for myMatcher"""
    def __init__(self,
                 p_unx = '/home/christianweber/felles/FILFLYTT/CWE/mapdata/',
                 p_nw =  'networkX_graph/ANV/',
                 crs=25832):
        super(myMatcher, self).__init__()
        # paths and files:
        self.p_unx = p_unx
        #self.p_ = self.p_unx + 'bike_centerline/Bike_centerline/'
        self.p_nw = self.p_unx + p_nw
        self.crs = crs
        self.params = {'d_ip': np.nan} # a dict to store parameters
        self.is_interpolated = False

    def reproject(self, crs=25832):
        """
        Make sure all geometries are in the same crs.
        """
        print("not implemented")
        pass

    def load_network(self, shp_ = 'bike_centerline_v1_extract_ANVedges_source_target.shp'):
        """
        Load the network shape file.
        So far, this is preprocessed in
        ./Maptech/MapMatching/networkX/test_networkx_on_centerline.ipynb

        this should rather be integrated here as well!
        """

        self.gdf_nw = gpd.read_file(self.p_nw + shp_)
        self.gdf_nw0 = self.gdf_nw
        self.network = fmm.Network(self.p_nw + shp_)
        print("Loaded network: {} nodes, {} edges".format(self.network.get_node_count(), self.network.get_edge_count()))

        self.graph = fmm.NetworkGraph(self.network)
        self.model = fmm.STMATCH(self.network, self.graph)

    def load_data_csv(self):
        print("not implemented")
        pass


    def load_data_db(self, tIDs):
        """
        Get the location data from the database.
        Since we already created triplines for all bike trips, we can use these directly as wkt.
        """

        import dbconnect as dbc
        importlib.reload(dbc)
        qry_ = dbc.session.query(
            dbc.tl_O_bike.c.tripId.label('track_id'),
            sqa.func.ST_AsEWKT(dbc.tl_O_bike.c.geom).label('geometry')
            ).\
            filter(dbc.tl_O_bike.c.tripId.in_(tIDs))
        self.gdf = dbc.prepare_gdf(qry_, dbc.engine_pg)
        self.gdf = self.gdf.to_crs(self.crs)

    def load_data_shp(self, f_shp):
        """
        Translate dataframe into wkt linestring.

        For now this is only implemented for the generated synthetic data!
        ./map-matching/cwe/ANV_generate_synthetic_data.ipynb

        """

        # load data
        self.gdf = gpd.read_file(f_shp)
        # reproject
        self.gdf = self.gdf.to_crs(epsg=self.crs)
        # convert multipoint to point:
        if self.gdf.geometry.loc[0].geom_type != 'Point':
            self.gdf.geometry = self.gdf.geometry.apply(lambda x: Point(x[0].x, x[0].y))
        else:
            pass


    def generate_lineString(self):

        # make a linestring per track:
        self.gdf_lineString = self.gdf.groupby('track_id', as_index=False).agg({'geometry': lambda x: LineString(x.tolist())})
        # convert to wkt:
        self.gdf_lineString['wkt'] = self.gdf_lineString.geometry.apply(lambda x: x.to_wkt())
        self.gdf_lineString = gpd.GeoDataFrame(self.gdf_lineString, crs = self.crs)

        # placeholder for interpolated values:
        self.gdf_lineString['ls_ip'] = None

    def reduce_trajectory(self, d=100):
        """
        Reduce number of data points, keep only points in a distance of d meters).
        """

        # helper function to do the interpolation. shapely.interpolate only creates one point, so we have to iterate over the LineString.
        def interpolate_ls(ls, distance):
            num_vert = int(round(ls.length / distance))
            res = [ls.interpolate(float(n) / num_vert, normalized=True) for n in range(num_vert + 1)]
            return LineString(res)

        self.gdf_lineString['ls_ip'] = self.gdf_lineString.apply(lambda x: interpolate_ls(x.geometry, d), axis=1)
        self.params.update({'d_ip': d})
        self.is_interpolated = True


    def config_matcher(self,
                       k = 8,          # number of candidates
                       gps_error = 50, # gps sensor error (map units)
                       radius = 300,   # search radius (map units)
                       vmax = 30,      # max vehicle speed (map units, stmatch only)
                       factor = 1.5,   # Factor to limit shortest path search (stmatch only)
                       ):

        self.k = k
        self.gps_error = gps_error
        self.r = radius
        self.vmax = vmax
        self.factor = factor

        self.params.update({
            'k': self.k,
            'r': self.r,
            'gps_e': self.gps_error,
            'vmax': self.vmax,
            'fact': self.factor
        })

        self.config = fmm.STMATCHConfig(k, radius, gps_error, vmax, factor)

    def match_wkt(self):
        self.result = self.model.match_wkt(self.wkt, self.config)
        self.id_edges = list(self.result.opath)
        self.gdf_match = self.gdf_nw.loc[self.gdf_nw.id.isin(self.id_edges)]

    def print_results(self):
        print("Matched path: ", list(self.result.cpath))
        print("Matched edge for each point: ", list(self.result.opath))
        print("Matched edge index ",list(self.result.indices))
        print("Matched geometry: ",self.result.mgeom.export_wkt())
        print("Matched point ", self.result.pgeom.export_wkt())

    def plot(self, zoom_=False, with_params=False):

        import matplotlib as mpl
        if with_params:
            self.fig, self.ax = mpl.pyplot.subplots(figsize=[10,5])
            self.fig.subplots_adjust(right=0.8, bottom=0.05)
        else:
            self.fig, self.ax = mpl.pyplot.subplots()
        self.gdf_nw.plot(ax=self.ax, label='network')
        self.gdf_lineString.plot(ax=self.ax, color='r', label='data')
        if self.gdf_lineString.ls_ip.any(): #plot data as sqares if we have interpolated data
            ls = self.gdf_lineString.ls_ip
            mpl.pyplot.plot(ls.iloc[0].xy[0], ls.iloc[0].xy[1], 'rs', label='ip data')
        else:
            ls = self.gdf_lineString.geometry
            mpl.pyplot.plot(ls.iloc[0].xy[0], ls.iloc[0].xy[1], 'ro', label='data')

        self.gdf_match.plot(ax=self.ax, color='g', lw=5, label='match')

        if with_params:
            # place parameters in figure:
            s_ = ''
            for key, value in self.params.items():
                s_ += "{}: {}\n".format(key, value)
            self.fig.text(0.85, 0.5, s_, transform=self.fig.transFigure, fontsize='large')
            self.fig.legend(loc=[0.85,0.3], ncol=1)
        else:
            self.fig.legend(loc=3, ncol=2)

        if zoom_:
            tb = self.gdf_lineString.total_bounds
            self.ax.set_xlim(tb[0]-self.r, tb[2]+self.r)
            self.ax.set_ylim(tb[1]-self.r, tb[3]+self.r)



    def clip_network(self, buff_ = 200):
        """
        reduce the network size by cliping with a buffer around the data.
        Note: this only clips the network that is plotted, the graph for the matching is not affected!
        """

        clip_ = self.gdf_lineString.buffer(buff_).values[0]
        df_clip = gpd.GeoDataFrame({'geometry': [clip_]}, crs=self.crs)
        self.gdf_nw = gpd.overlay(self.gdf_nw0, df_clip, how='intersection')




























