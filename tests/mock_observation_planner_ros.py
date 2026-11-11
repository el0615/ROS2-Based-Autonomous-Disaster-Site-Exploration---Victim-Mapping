#!/usr/bin/env python3
"""Strict mocked ROS adapter regression test. NOT ROS2/DDS or robot validation."""
from __future__ import annotations
from collections import defaultdict
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import types
from types import SimpleNamespace as NS
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts/exploration'))
from hospital_coverage_core import load_bundle
from observation_goal_planner import parser,build_node_type,geometry_matches
from observation_planning_core import Settings

class Time:
    def __init__(self,nanoseconds=0):self.nanoseconds=int(nanoseconds)
    @classmethod
    def from_msg(cls,m):return cls(m.sec*10**9+m.nanosec)
    def to_msg(self):return NS(sec=self.nanoseconds//10**9,nanosec=self.nanoseconds%10**9)
class Duration:
    def __init__(self,seconds=0,nanoseconds=0):self.nanoseconds=round(seconds*1e9)+int(nanoseconds)
class Pub:
    def __init__(self,topic):self.topic=topic;self.last=None
    def publish(self,m):self.last=m
class Logger:
    def __init__(self,logs):self.logs=logs;self.levels={}
    def _call(self,s,level):
        frame=inspect.currentframe().f_back.f_back
        key=(frame.f_code.co_name,frame.f_lineno)
        if key in self.levels and self.levels[key]!=level:raise ValueError('Logger severity cannot be changed between calls.')
        self.levels[key]=level;self.logs.append(s)
    def info(self,s):self._call(s,'info')
    def warning(self,s):self._call(s,'warning')
    def error(self,s):self._call(s,'error')
class Node:
    def __init__(self,name):
        self._test_pubs={};self._test_subs={};self._test_now_ns=int(10.2e9);self._logs=[];self._logger=Logger(self._logs)
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
    def create_subscription(self,typ,topic,fn,qos):self._test_subs[topic]=(typ,fn);return NS()
    def get_clock(self):return NS(now=lambda:Time(self._test_now_ns))
    def get_logger(self):return self._logger
    def get_parameter(self,k):return NS(value=True)
class TransformException(Exception):pass
class Buffer:
    def __init__(self,cache_time,node=None):self.node=node;self.ready=False;self.pose=(2.,0.,0.);self.age=0.
    def lookup_transform(self,target,source,when,timeout):
        assert timeout.nanoseconds==0 and when.nanoseconds==0
        if not self.ready:raise TransformException('map frame unavailable')
        x,y,yaw=self.pose if target=='map' else (.1,0.,0.)
        return NS(header=NS(stamp=Time(self.node._test_now_ns-round(self.age*1e9)).to_msg()),
                  transform=NS(translation=NS(x=x,y=y,z=0.),rotation=NS(x=0.,y=0.,z=math.sin(yaw/2),w=math.cos(yaw/2))))
class Marker:
    ADD=0;DELETE=2;DELETEALL=3;ARROW=0;LINE_STRIP=4;CUBE_LIST=6;SPHERE_LIST=7;TEXT_VIEW_FACING=9
    def __init__(self):
        self.header=NS(frame_id='',stamp=None);self.pose=NS(position=NS(x=0.,y=0.,z=0.),orientation=NS(x=0.,y=0.,z=0.,w=0.))
        self.scale=NS(x=0.,y=0.,z=0.);self.color=NS(r=0.,g=0.,b=0.,a=0.);self.lifetime=NS(sec=0,nanosec=0);self.points=[]
class MarkerArray:
    def __init__(self):self.markers=[]
class OccupancyGrid:
    def __init__(self):self.header=NS(frame_id='',stamp=None);self.info=None;self.data=[]
class String:
    def __init__(self):self.data=''
class LaserScan:pass
class Point:
    def __init__(self,*,x,y,z):
        assert all(isinstance(a,float) for a in [x,y,z]);self.x=x;self.y=y;self.z=z
class QoS:
    def __init__(self,**kw):self.__dict__.update(kw)
def module(name,**values):
    m=types.ModuleType(name)
    for k,v in values.items():setattr(m,k,v)
    sys.modules[name]=m
for p in ('nav_msgs','sensor_msgs','std_msgs','visualization_msgs','geometry_msgs'):module(p)
module('rclpy',ok=lambda:True)
module('rclpy.node',Node=Node);module('rclpy.time',Time=Time);module('rclpy.duration',Duration=Duration)
module('rclpy.qos',QoSProfile=QoS,ReliabilityPolicy=NS(RELIABLE=1,BEST_EFFORT=2),DurabilityPolicy=NS(TRANSIENT_LOCAL=1,VOLATILE=2),qos_profile_sensor_data=QoS())
module('nav_msgs.msg',OccupancyGrid=OccupancyGrid);module('sensor_msgs.msg',LaserScan=LaserScan)
module('std_msgs.msg',String=String);module('visualization_msgs.msg',Marker=Marker,MarkerArray=MarkerArray)
module('geometry_msgs.msg',Point=Point)
module('tf2_ros',Buffer=Buffer,TransformListener=lambda b,n:NS(),TransformException=TransformException)

def main():
    args=parser().parse_args([]);bundle=load_bundle(args.zones,args.reference)
    data,g,labels,free,signature=bundle;cfg=Settings(max_candidates=16)
    import observation_goal_planner as mod
    real_verify=mod.verify_map
    mod.verify_map=lambda msg,info:(msg.test_ok,'reference matches' if msg.test_ok else 'mismatch')
    geom=NS(width=g.width,height=g.height,resolution=g.resolution,
            origin=NS(position=NS(x=g.origin[0],y=g.origin[1],z=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.)))
    def grid(values,t):
        m=OccupancyGrid();m.header.frame_id='map';m.header.stamp=Time(round(t*1e9)).to_msg();m.info=geom;m.data=values.ravel().tolist();m.test_ok=True;return m
    observed=np.zeros(free.shape,bool);evidence=1
    def report(t,obs,session='a',health='LIVE'):
        return dict(schema='hospital_room_coverage_v1',session_id=session,map_validated=True,fault=None,
                    zone_bundle_signature=signature,observed_cells=int(obs.sum()),data_health=health,
                    evidence_scans=evidence,last_evidence_scan_end_s=t,ros_time_s=t,max_observation_range_m=8.,
                    zones=[dict(id=z['id'],reference_free_cells=z['known_free_cells'],observed_cells=int(np.count_nonzero(obs&(labels==z['mask_id'])))) for z in data['zones']])
    def deliver(node,t,obs,session='a',health='LIVE'):
        s=String();s.data=json.dumps(report(t,obs,session,health));node.on_coverage_status(s)
        vals=np.where(labels>0,0,-1);vals[obs]=100;node.on_coverage_grid(grid(vals,t+.001))
    def sensors(node,t):
        node.on_scan(NS(header=NS(frame_id='lidar2d_0_laser',stamp=Time(round(t*1e9)).to_msg()),ranges=[25.]*180,
                             angle_min=-math.pi,angle_increment=math.tau/180,range_min=.1,range_max=25.))
        node.on_costmap(grid(np.where(free,0,100).astype(np.int16),t))
    def poll_now(node):node.next_publish_wall=0;node.poll()
    with tempfile.TemporaryDirectory() as tmp:
        cls=build_node_type(args,bundle,cfg,Path(tmp));node=cls()
        assert set(node._test_pubs)=={'/observation_planner/markers','/observation_planner/status'}
        assert len(node._test_subs)==5
        poll_now(node);assert node.last_report['state']=='WAIT_MAP'
        mp=grid(np.where(free,0,100),10.);node.on_map(mp)
        poll_now(node);assert node.last_report['state']=='WAIT_MATCHED_COVERAGE'
        deliver(node,10.,observed);sensors(node,10.)
        poll_now(node);assert node.last_report['state']=='WAIT_TF'
        node.tf_buffer.ready=True;poll_now(node)
        assert node.last_report['state']=='PLANNING'
        deadline=time.monotonic()+10
        while time.monotonic()<deadline and node.cached_plan is None:
            time.sleep(.03);poll_now(node)
        assert node.last_report['state']=='SELECTED',node.last_report
        assert node.last_report['selected']['new_cells']>0
        assert not node.last_report['nav2_plan_checked'] and not node.last_report['navigation_commands_sent']
        assert any(m.ns=='predicted_new_cells_NOT_OBSERVED' for m in node.marker_publisher.last.markers if hasattr(m,'ns'))
        assert np.count_nonzero(node.matched['observed'])==0
        # Reproduce INFO/WARN/INFO at separate callsites, including throttled helper paths.
        node.info('log1','info regression',0);node.warn('log2','warning regression',0);node.info('log1','info again',0)
        node.tf_buffer.age=3.;poll_now(node);assert node.last_report['state']=='WAIT_TF'
        node.tf_buffer.age=0.
        node._test_now_ns=int(30e9);poll_now(node);assert node.last_report['state']=='STALE_COVERAGE'
        # Session reset invalidates old candidates; actual >=90 needs a second evidence scan.
        node._test_now_ns=int(31.2e9);target=labels==next(z['mask_id'] for z in data['zones'] if z['id']=='R14_T')
        ids=np.flatnonzero(target);observed.ravel()[ids[:math.ceil(len(ids)*.92)]]=True
        sensors(node,31.);deliver(node,31.,observed,session='b');poll_now(node)
        assert node.last_report['state']=='THRESHOLD_PENDING'
        evidence=2;node._test_now_ns=int(32.5e9);sensors(node,32.3);deliver(node,32.3,observed,session='b');poll_now(node)
        assert node.last_report['state']=='OBSERVATION_SUFFICIENT'
        assert not any(getattr(m,'ns','')=='selected_view' for m in node.marker_publisher.last.markers)
        # A newer unhealthy report cannot be masked by older good state.
        deliver(node,32.4,observed,session='b',health='HALTED');poll_now(node)
        assert node.last_report['state']=='WAIT_COVERAGE_LIVE'
        node._test_now_ns=int(1e9);poll_now(node)
        assert node.last_report['state']=='CLOCK_RESET_RESTART_REQUIRED'
        node.close()
        saved=json.loads((Path(tmp)/'latest_plan.json').read_text())
        assert not saved['navigation_commands_sent']
        node2=cls();node2.on_map(mp);mp.test_ok=False;node2.on_map(mp)
        assert node2.fault=='REFERENCE_MAP_CHANGED_RESTART_REQUIRED';node2.close()
    mod.verify_map=real_verify
    print('[MOCK ROS PASS] 5 inputs, 2 display-only outputs, independent logger sites, TF gating, async evaluation, markers/JSON, stale/session/clock/map handling, actual 90% gate')
    print('No real ROS2/DDS/Gazebo test was performed.')

if __name__=='__main__':main()
