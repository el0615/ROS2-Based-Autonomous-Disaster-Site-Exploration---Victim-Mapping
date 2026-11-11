"""Synthetic fixtures ONLY, never installed as ROS/project replacements."""
from pathlib import Path
import math, sys, types
import numpy as np
SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))

class Geometry:
    def __init__(self,w=100,h=80,r=.1,origin=(0.,0.,0.)):
        self.width=w;self.height=h;self.resolution=r;self.origin=origin
    def contains(self,i,j):return 0<=i<self.width and 0<=j<self.height
    def world_to_grid_float(self,x,y):
        a,b,t=self.origin;c=math.cos(t);s=math.sin(t)
        return ((x-a)*c+(y-b)*s)/self.resolution, (-(x-a)*s+(y-b)*c)/self.resolution
    def cell_to_world(self,i,j):
        x=(i+.5)*self.resolution;y=(j+.5)*self.resolution
        a,b,t=self.origin;c=math.cos(t);s=math.sin(t)
        return a+c*x-s*y,b+s*x+c*y

# Project helpers not included in the uploaded source ZIP. These test stubs are
# only for pure tests / fake ROS; deployment imports the existing project files.
m=types.ModuleType('hospital_coverage_core');m.Geometry=Geometry
m.load_bundle=lambda *a: (_ for _ in ()).throw(RuntimeError('Synthetic test has no live bundle'))
def planar(t,max_tilt):
    return (t.translation.x,t.translation.y,2*math.atan2(t.rotation.z,t.rotation.w))
m.planar_pose=planar
sys.modules['hospital_coverage_core']=m
m=types.ModuleType('entrance_navigation_overlay')
m.apply_entrance_overlay=lambda *a: (_ for _ in ()).throw(RuntimeError('No synthetic entrance overlay'))
sys.modules['entrance_navigation_overlay']=m
m=types.ModuleType('hospital_zone_visualizer');m.verify_map=lambda msg,info:(True,'synthetic')
sys.modules['hospital_zone_visualizer']=m


def bundle():
    g=Geometry()
    labels=np.zeros((g.height,g.width),np.int16)
    labels[5:75,5:45]=1
    labels[15:40,45:90]=2
    labels[45:70,45:90]=3
    free=np.ones_like(labels,dtype=bool)
    free[[0,-1],:]=False;free[:,[0,-1]]=False
    zones=[]
    for z,kind,k in [('LOBBY','lobby',1),('R_A','room',2),('R_B','room',3)]:
        zones.append(dict(id=z,kind=kind,mask_id=k,known_free_cells=int((labels==k).sum())))
    return ({'zones':zones,'map':{'width':g.width,'height':g.height,'resolution':g.resolution,'origin':list(g.origin),'frame_id':'map'}},g,labels,free,'test-signature')


def status(b,seen=None,seq=1,t=1.):
    labels=b[2];seen=np.zeros_like(labels,dtype=bool) if seen is None else seen
    zones=[dict(id=z['id'],reference_free_cells=z['known_free_cells'],observed_cells=int((seen&(labels==z['mask_id'])).sum())) for z in b[0]['zones']]
    return dict(schema='hospital_room_coverage_v1',session_id='test-session',zone_bundle_signature=b[4],
        map_validated=True,fault=None,data_health='LIVE',evidence_scans=seq,
        ros_time_s=t,last_evidence_scan_end_s=t,observed_cells=int(seen.sum()),
        max_observation_range_m=8.,overall_observed_percent=100*seen.sum()/max(1,(labels>0).sum()),zones=zones)


def snapshot(b,pose=(2.,2.,0.)):
    from planning_core import SensorModel
    cost=np.zeros_like(b[2],dtype=np.int16);cost[~b[3]]=100
    seen=np.zeros_like(b[2],dtype=bool)
    return dict(pose=pose,sensor=SensorModel(-math.pi,2*math.pi/80,80,.05,8.),
                cost=dict(values=cost,key='grid1',stamp=10.,wall=0.),observed=seen,
                status=status(b,seen,t=10.),epoch=1)
