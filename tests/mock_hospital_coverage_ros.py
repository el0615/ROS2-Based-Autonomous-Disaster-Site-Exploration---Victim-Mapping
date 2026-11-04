#!/usr/bin/env python3
"""Strict mocked ROS callbacks. This is NOT a real ROS/DDS/Gazebo test."""
from __future__ import annotations
import array
from collections import defaultdict
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace as NS
import numpy as np

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts/exploration'))
from hospital_coverage_core import load_bundle
from hospital_zone_visualizer import pixel_to_map
from hospital_zone_coverage import build_node_type,parser

class Time:
    def __init__(self,nanoseconds=0):self.nanoseconds=int(nanoseconds)
    @classmethod
    def from_msg(cls,msg):return cls(msg.sec*10**9+msg.nanosec)
    def to_msg(self):return NS(sec=self.nanoseconds//10**9,nanosec=self.nanoseconds%10**9)
    def __add__(self,d):return Time(self.nanoseconds+d.nanoseconds)
class Duration:
    def __init__(self,seconds=0.,nanoseconds=0):self.nanoseconds=round(seconds*1e9)+int(nanoseconds)
class Pub:
    def __init__(self,topic):self.topic=topic;self.last=None
    def publish(self,msg):self.last=msg
class Node:
    def __init__(self,name):self._test_pubs={};self._test_subs={};self._test_now_ns=int(10.2e9);self._logs=[]
    # Properties matching rclpy.Node reject accidental reserved-name assignments.
    @property
    def guards(self):return []
    @property
    def publishers(self):return list(self._test_pubs.values())
    @property
    def subscriptions(self):return list(self._test_subs.values())
    @property
    def timers(self):return []
    @property
    def clients(self):return []
    def create_publisher(self,typ,topic,qos):
        p=Pub(topic);self._test_pubs[topic]=p;return p
    def create_subscription(self,typ,topic,callback,qos):
        self._test_subs[topic]=(typ,callback);return NS()
    def get_clock(self):return NS(now=lambda:Time(self._test_now_ns))
    def get_logger(self):return NS(info=lambda s:self._logs.append(s),warning=lambda s:self._logs.append(s),error=lambda s:self._logs.append(s))
class Buffer:
    def __init__(self,cache_time):self.ready=False;self.pose=(0.,0.,0.);self.requests=[]
    def can_transform(self,t,s,when,timeout):
        assert t=='map' and s=='lidar2d_0_laser' and timeout.nanoseconds==0
        self.requests.append(when.nanoseconds);return self.ready
    def lookup_transform(self,t,s,when,timeout):
        assert when.nanoseconds>0 and timeout.nanoseconds==0
        x,y,yaw=self.pose
        return NS(transform=NS(translation=NS(x=x,y=y,z=0.3),rotation=NS(x=0.,y=0.,z=math.sin(yaw/2),w=math.cos(yaw/2))))
    def clear(self):self.ready=False
class TransformException(Exception):pass
class Marker:
    ADD=0;DELETE=2;CUBE_LIST=6;TEXT_VIEW_FACING=9
    def __init__(self):
        self.header=NS(frame_id='',stamp=None);self.pose=NS(position=NS(x=0.,y=0.,z=0.),orientation=NS(w=0.))
        self.scale=NS(x=0.,y=0.,z=0.);self.color=NS(r=0.,g=0.,b=0.,a=0.)
        self.lifetime=NS(sec=0,nanosec=0);self.points=[]
class MarkerArray:
    def __init__(self):self.markers=[]
class OccupancyGrid:
    def __init__(self):self.header=NS(frame_id='',stamp=None);self.info=None;self.data=[]
class String:
    def __init__(self):self.data=''
class LaserScan:pass
class Point:
    def __init__(self,*,x,y,z):self.x=x;self.y=y;self.z=z
class QoS:
    def __init__(self,**kw):self.__dict__.update(kw)

def module(name,**values):
    m=types.ModuleType(name)
    for k,v in values.items():setattr(m,k,v)
    sys.modules[name]=m

for parent in ('rclpy','nav_msgs','sensor_msgs','std_msgs','visualization_msgs','geometry_msgs'):
    module(parent)
module('rclpy.node',Node=Node);module('rclpy.time',Time=Time);module('rclpy.duration',Duration=Duration)
module('rclpy.qos',QoSProfile=QoS,ReliabilityPolicy=NS(RELIABLE=1),DurabilityPolicy=NS(TRANSIENT_LOCAL=1),qos_profile_sensor_data=QoS())
module('nav_msgs.msg',OccupancyGrid=OccupancyGrid);module('sensor_msgs.msg',LaserScan=LaserScan)
module('std_msgs.msg',String=String);module('visualization_msgs.msg',Marker=Marker,MarkerArray=MarkerArray)
module('geometry_msgs.msg',Point=Point)
module('tf2_ros',Buffer=Buffer,TransformListener=lambda b,n:NS(),TransformException=TransformException)

def main():
    args=parser().parse_args([])
    data,g,labels,free,signature=load_bundle(args.zones,args.reference)
    # Rebuild expected ROS occupancy from the source PGM if available.
    # For test portability use explicit 0/100 map reconstructed below from provided
    # stored-mask/free content only for callbacks, and stub map verification AFTER
    # testing the real verifier in the separate offline audit.
    import hospital_zone_coverage as mod
    real_verify=mod.verify_map
    mod.verify_map=lambda msg,info:(msg.test_ok,'reference map geometry and occupancy match' if msg.test_ok else 'test mismatch')
    with tempfile.TemporaryDirectory() as tmp:
        cls=build_node_type(args,data,g,labels,free,signature,Path(tmp))
        node=cls()
        assert set(node._test_pubs)=={'/hospital_coverage/grid','/hospital_coverage/status','/hospital_coverage/markers'}
        assert set(node._test_subs)=={'/j100_0001/map','/j100_0001/sensors/lidar2d_0/scan'}
        node.publish();assert node.make_report()['data_health']=='WAIT_MAP'
        msg=OccupancyGrid();msg.test_ok=True;msg.header.frame_id='map';msg.info=NS(width=g.width,height=g.height)
        node.on_map(msg)
        assert node.make_report()['data_health']=='WAIT_SCAN'
        # Pick known free cells inside the large lobby.
        rid=next(z['mask_id'] for z in data['zones'] if z['id']=='LOBBY')
        yy,xx=np.nonzero(labels==rid);k=len(xx)//2
        x,y=g.cell_to_world(int(xx[k]),int(yy[k]));node.tf_buffer.pose=(x,y,0.)
        def scan(t):
            return NS(header=NS(frame_id='lidar2d_0_laser',stamp=Time(round(t*1e9)).to_msg()),
                      ranges=[3.]*360,angle_min=-math.pi,angle_increment=math.tau/360,
                      range_min=.05,range_max=25.,time_increment=.0001)
        node.on_scan(scan(10.));node.process_pending();assert node.coverage.scan_count==0 and len(node.pending_scans)==1
        node.tf_buffer.ready=True;node.process_pending()
        assert node.coverage.scan_count==1 and node.coverage.observed_total>0
        assert node.make_report()['data_health']=='LIVE'
        node.publish();node.save()
        saved=json.loads((Path(tmp)/'coverage_status.json').read_text())
        assert saved['zone_count']==27 and saved['observed_cells']>0
        assert len(saved['zones'])==27 and all(not z['victim_search_complete'] for z in saved['zones'])
        assert len(node.grid_publisher.last.data)==g.width*g.height
        assert np.count_nonzero(np.asarray(node.grid_publisher.last.data)==100)==node.coverage.observed_total
        assert len(node.marker_publisher.last.markers)==28
        node._test_now_ns=int(11.2e9);node.on_scan(scan(11.));node.process_pending();assert node.coverage.scan_count==2
        node.on_scan(scan(11.));assert node.counters['out_of_order_or_duplicate']==1
        node._test_now_ns=int(25e9);assert node.make_report()['data_health']=='STALE'
        # Sim-time expiry, not immediate rejection or latest-transform fallback.
        node.tf_buffer.ready=False;node.on_scan(scan(12.));node.process_pending()
        assert node.counters['tf_timeout_drops']==1
        node.last_clock_ns=int(25e9);node._test_now_ns=int(1e9);node.poll()
        assert node.fault=='SIMULATION_CLOCK_MOVED_BACKWARD_RESTART_REQUIRED'
        assert all(m.action==Marker.DELETE for m in node.marker_publisher.last.markers)
        # Wrong map is a latched halt after acceptance.
        node2=cls();node2.on_map(msg);msg.test_ok=False;node2.on_map(msg)
        assert node2.fault=='REFERENCE_MAP_CHANGED_RESTART_REQUIRED'
        assert node2.make_report()['data_health']=='HALTED'
    mod.verify_map=real_verify
    print('[MOCK ROS PASS] constructors, exact-time queue, 3 output-only topics, labels/grid/report, stale data, clock/map halt')
    print('Mocked callbacks only: no real ROS2/DDS/TF transport, Gazebo, or robot test.')

if __name__=='__main__':main()
