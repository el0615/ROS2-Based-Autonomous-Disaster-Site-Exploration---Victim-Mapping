#!/usr/bin/env python3
"""Strict fake ROS messages/clients test. NOT ROS2/DDS or Gazebo validation."""
from __future__ import annotations
from concurrent.futures import Future
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
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts/exploration'))
from hospital_coverage_core import Geometry
import hospital_observation_sequence as mod
from observation_planning_core import Settings


def module(name,**objects):
    m=types.ModuleType(name);m.__dict__.update(objects);sys.modules[name]=m

class Time:
    def __init__(self,nanoseconds=0):self.nanoseconds=int(nanoseconds)
    def to_msg(self):return NS(sec=self.nanoseconds//10**9,nanosec=self.nanoseconds%10**9)
class Duration:
    def __init__(self,seconds=0):self.nanoseconds=int(seconds*1e9)
class Pub:
    def __init__(self):self.last=None
    def publish(self,m):self.last=m
class Logger:
    def __init__(self):self.messages=[];self.levels={}
    def emit(self,level,msg):
        f=inspect.currentframe().f_back.f_back;key=(f.f_code.co_filename,f.f_lineno)
        if key in self.levels and self.levels[key]!=level:raise ValueError('Logger severity cannot change')
        self.levels[key]=level;self.messages.append(msg)
    def info(self,msg):self.emit('info',msg)
    def warning(self,msg):self.emit('warning',msg)
    def error(self,msg):self.emit('error',msg)
class Node:
    def __init__(self,name):self.node_name=name;self.clock_ns=int(10e9);self.logger=Logger();self.pubs={};self.subs={};self.svcs={}
    @property
    def guards(self):return []
    @property
    def clients(self):return []
    @property
    def publishers(self):return []
    @property
    def subscriptions(self):return []
    def get_logger(self):return self.logger
    def get_clock(self):return NS(now=lambda:Time(self.clock_ns))
    def create_publisher(self,typ,topic,qos):p=Pub();self.pubs[topic]=p;return p
    def create_subscription(self,typ,topic,cb,qos):self.subs[topic]=(typ,cb);return NS()
    def create_service(self,typ,name,cb):self.svcs[name]=cb;return NS()
    def create_client(self,typ,name):return FakeCancelClient()
class QoS:
    def __init__(self,**kw):self.__dict__.update(kw)
class TransformException(Exception):pass
class Buffer:
    def __init__(self,cache_time,node):self.node=node;self.pose=(2.,3.,0.);self.ready=True
    def lookup_transform(self,target,source,when,timeout):
        if not self.ready:raise TransformException('Missing TF')
        x,y,a=self.pose if source=='base_link' else (0.,0.,0.)
        return NS(header=NS(stamp=Time(self.node.clock_ns).to_msg()),transform=NS(translation=NS(x=x,y=y,z=0.),rotation=NS(x=0.,y=0.,z=math.sin(a/2),w=math.cos(a/2))))
class PoseStamped:
    def __init__(self):self.header=NS(frame_id='',stamp=NS(sec=0,nanosec=0));self.pose=NS(position=NS(x=0.,y=0.,z=0.),orientation=NS(x=0.,y=0.,z=0.,w=0.))
class Point:
    def __init__(self,*,x,y,z):
        assert all(isinstance(v,float) for v in (x,y,z));self.x=x;self.y=y;self.z=z
class Marker:
    ADD=0;DELETEALL=3;ARROW=0;LINE_STRIP=4;CUBE_LIST=6;TEXT_VIEW_FACING=9
    def __init__(self):
        p=PoseStamped();self.header=p.header;self.pose=p.pose;self.scale=NS(x=0.,y=0.,z=0.);self.color=NS(r=0.,g=0.,b=0.,a=0.);self.points=[]
class MarkerArray:
    def __init__(self):self.markers=[]
class String:
    def __init__(self):self.data=''
class UUID:
    def __init__(self):self.uuid=[0]*16
class Action:
    class Goal:
        def __init__(self):self.goal=PoseStamped();self.pose=PoseStamped();self.planner_id='';self.use_start=False;self.behavior_tree=''
class CancelGoal:
    class Request:
        def __init__(self):self.goal_info=NS(goal_id=UUID(),stamp=NS(sec=0,nanosec=0))
class FakeCancelClient:
    def __init__(self):self.requests=[]
    def service_is_ready(self):return True
    def call_async(self,req):self.requests.append(req);return Future()
class Handle:
    def __init__(self,uid=None,accepted=True):
        self.accepted=accepted;self.goal_id=uid or UUID();self.result_future=Future();self.cancels=[]
    def get_result_async(self):return self.result_future
    def cancel_goal_async(self):f=Future();self.cancels.append(f);return f
class Client:
    def __init__(self,node,typ,name):self.name=name;self.requests=[];self.ready=True
    def server_is_ready(self):return self.ready
    def send_goal_async(self,goal,feedback_callback=None,goal_uuid=None):
        f=Future();self.requests.append(NS(goal=goal,feedback=feedback_callback,uuid=goal_uuid,future=f));return f
class InlinePool:
    def __init__(self,**kw):pass
    def submit(self,fn,*args):
        f=Future()
        try:f.set_result(fn(*args))
        except Exception as e:f.set_exception(e)
        return f
    def shutdown(self,**kw):pass
for name in ('rclpy','nav_msgs','sensor_msgs','std_msgs','std_srvs','action_msgs','unique_identifier_msgs','nav2_msgs','geometry_msgs','visualization_msgs'):module(name)
sys.modules['rclpy'].ok=lambda:True
module('rclpy.node',Node=Node);module('rclpy.time',Time=Time);module('rclpy.duration',Duration=Duration)
module('rclpy.qos',QoSProfile=QoS,ReliabilityPolicy=NS(RELIABLE=1,BEST_EFFORT=2),DurabilityPolicy=NS(TRANSIENT_LOCAL=1,VOLATILE=2),qos_profile_sensor_data=QoS())
module('rclpy.action',ActionClient=Client)
module('nav_msgs.msg',OccupancyGrid=NS,Odometry=NS);module('sensor_msgs.msg',LaserScan=NS)
module('std_msgs.msg',String=String);module('std_srvs.srv',Trigger=NS)
module('action_msgs.msg',GoalStatusArray=NS);module('action_msgs.srv',CancelGoal=CancelGoal)
module('unique_identifier_msgs.msg',UUID=UUID);module('nav2_msgs.action',ComputePathToPose=Action,NavigateToPose=Action)
module('geometry_msgs.msg',Point=Point,PoseStamped=PoseStamped);module('visualization_msgs.msg',Marker=Marker,MarkerArray=MarkerArray)
module('tf2_ros',Buffer=Buffer,TransformListener=lambda *a:NS(),TransformException=TransformException)
mod.ThreadPoolExecutor=InlinePool;mod.verify_map=lambda msg,info:(msg.ok,'reference')
G=Geometry(100,70,.1,(0.,0.,0.));LABELS=np.zeros((70,100),np.uint16)
LABELS[10:60,10:40]=1;LABELS[10:60,40:70]=2;LABELS[10:60,70:90]=3
FREE=np.ones(LABELS.shape,bool)
Z=[dict(id=n,mask_id=k,known_free_cells=int(np.count_nonzero(LABELS==k))) for n,k in [('A',1),('B',2),('C',3)]]
INFO=dict(width=100,height=70,resolution=.1,origin=[0.,0.,0.],frame_id='map')
BUNDLE=({'zones':Z,'map':INFO},G,LABELS,FREE,'test-signature')

def batch(bundle,snapshot,zones,cfg,previous,locked,shortlist):
    out=[]
    for z in zones:
        x={'A':3.,'B':5.,'C':8.}[z['id']];y=3.;i,j=map(math.floor,G.world_to_grid_float(x,y))
        c=dict(zone_id=z['id'],id='P_'+z['id'],x=x,y=y,yaw=.7,cell=[i,j],visible_indices=np.flatnonzero(LABELS==z['mask_id']),
               new_indices=np.flatnonzero(LABELS==z['mask_id']),new_cells=z['known_free_cells'],new_area_m2=10.,
               path_est_m=abs(x-snapshot['pose'][0]),score=1.,path_cells=[])
        if not any(p['zone_id']==z['id'] for p in previous):out.append(c)
    return dict(candidates=out,failures={},fatal=None)
mod.build_batch=batch

def grid(vals,t):
    return NS(ok=True,header=NS(frame_id='map',stamp=Time(int(t*1e9)).to_msg()),
              info=NS(width=100,height=70,resolution=.1,origin=NS(position=NS(x=0.,y=0.,z=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.))),data=vals.ravel().tolist())

class Trial:
    def __init__(self,tmp,execute=True,early=False,limit=3):
        self.args=mod.parser().parse_args((['--execute'] if execute else [])+(['--cancel-on-coverage'] if early else [])+['--max-goals',str(limit)])
        out=Path(tmp);out.mkdir(parents=True,exist_ok=True)
        self.n=mod.make_node_type(self.args,BUNDLE,Settings(),out)();self.n.boot_wall-=3
        self.t=10.;self.seq=0;self.obs=np.zeros(LABELS.shape,bool)
        self.n.on_map(grid(np.where(FREE,0,100),self.t))
    def cover(self,zid,f):
        z=next(z for z in Z if z['id']==zid);idx=np.flatnonzero(LABELS==z['mask_id']);self.obs.ravel()[idx[:math.ceil(len(idx)*f)]]=True
    def tick(self,dt=.25,health='LIVE',odom=True,coverage=True,move=0.):
        self.t+=dt;self.seq+=1;n=self.n;n.clock_ns=int(self.t*1e9)
        n.on_scan(NS(header=NS(frame_id='laser',stamp=Time(n.clock_ns).to_msg()),angle_min=-math.pi,angle_increment=math.tau/180,
                     ranges=[20.]*180,range_min=.1,range_max=25.))
        n.on_costmap(grid(np.zeros(LABELS.shape,int),self.t));n.on_local_costmap(grid(np.zeros(LABELS.shape,int),self.t))
        if odom:n.on_odom(NS(header=NS(stamp=Time(n.clock_ns).to_msg()),twist=NS(twist=NS(linear=NS(x=move,y=0.,z=0.),angular=NS(z=0.)))))
        if coverage:
            s=dict(schema='hospital_room_coverage_v1',session_id='m',zone_bundle_signature='test-signature',map_validated=True,fault=None,
                   ros_time_s=self.t,last_evidence_scan_end_s=self.t,evidence_scans=self.seq,data_health=health,max_observation_range_m=8.,
                   observed_cells=int(self.obs.sum()),overall_observed_percent=100*int(self.obs.sum())/np.count_nonzero(LABELS),
                   zones=[dict(id=z['id'],reference_free_cells=z['known_free_cells'],observed_cells=int(np.count_nonzero(self.obs & (LABELS==z['mask_id'])))) for z in Z])
            n.on_status(NS(data=json.dumps(s)));n.on_grid(grid(np.where(LABELS>0,np.where(self.obs,100,0),-1),self.t+.001))
        n.poll()
    def until_plan(self):
        for _ in range(15):
            self.tick()
            if self.n.plan_send is not None:return
        raise AssertionError(('no preflight',self.n.state,self.n.logger.messages[-10:]))
    def plan_ok(self):
        n=self.n;req=n.plan_client.requests[-1];h=Handle();req.future.set_result(h);self.tick()
        p=n.current;start=n.tf_buffer.pose[:2]
        poses=[]
        for x,y in (start,(p['x'],p['y'])):
            q=PoseStamped();q.header.frame_id='map';q.pose.position.x=x;q.pose.position.y=y;poses.append(q)
        h.result_future.set_result(NS(status=4,result=NS(path=NS(header=NS(frame_id='map'),poses=poses))))
        self.tick();return n.nav_client.requests[-1] if n.nav_client.requests else None
    def nav_accept(self):
        n=self.n;r=n.nav_client.requests[-1];h=Handle(r.uuid);r.future.set_result(h);self.tick();return h
    def nav_finish(self,h,status=4):
        h.result_future.set_result(NS(status=status,result=NS()));self.tick()
    def close(self):self.n.close()


def main():
    checks=[]
    with tempfile.TemporaryDirectory() as tmp:
        t=Trial(Path(tmp)/'dry',False);t.until_plan();t.plan_ok()
        assert t.n.done and t.n.state=='DRY_RUN_PASS' and not t.n.nav_client.requests
        assert set(t.n.pubs)=={'/hospital_observation/markers','/hospital_observation/status'}
        t.close();checks.append('dry-run cannot navigate; only marker/status publishers')

        t=Trial(Path(tmp)/'two',True,limit=2);t.until_plan();req=t.plan_ok();assert req
        assert abs(req.goal.pose.pose.orientation.z-math.sin(.7/2))<1e-9
        h=t.nav_accept();assert t.n.state=='NAV_RUNNING'
        t.cover('A',.95)
        for _ in range(6):t.tick(move=.2)
        assert t.n.book.is_complete('A') and not h.cancels and len(t.n.nav_client.requests)==1
        t.n.tf_buffer.pose=(3.,3.,.7);t.nav_finish(h)
        # Old/initial stopped state cannot count as stopping after terminal result.
        assert t.n.state=='WAIT_STOP_AND_COVERAGE' and len(t.n.nav_client.requests)==1
        for _ in range(6):t.tick()
        t.until_plan();assert t.n.current['zone_id']=='B'
        t.plan_ok();h2=t.nav_accept();t.n.tf_buffer.pose=(5.,3.,.7);t.nav_finish(h2)
        for _ in range(6):t.tick()
        assert t.n.done and t.n.state=='GOAL_LIMIT_REACHED' and t.n.sent_goals==2
        t.close();checks.append('preserved yaw; one active Goal; completed room skipped; fresh stop; 2-goal bound')

        t=Trial(Path(tmp)/'early',True,True);t.until_plan();t.plan_ok();h=t.nav_accept();t.cover('A',.95)
        for _ in range(6):t.tick(move=.2)
        assert t.n.cancel_reason=='COVERAGE_SUFFICIENT' and len(h.cancels)==1
        h.cancels[0].set_result(NS(return_code=0,goals_canceling=[NS(goal_id=h.goal_id)]));t.tick()
        assert t.n.nav_terminal is None and len(t.n.nav_client.requests)==1
        t.nav_finish(h,5)
        for _ in range(6):t.tick()
        assert t.n.coverage_cancels==1 and not t.n.stop_requested
        t.n.request_stop('USER_TEST');t.tick();t.close();checks.append('early cancel needs terminal result and stop, not ACK alone')

        t=Trial(Path(tmp)/'abort',True);t.until_plan();t.plan_ok();h=t.nav_accept();t.nav_finish(h,6)
        for _ in range(6):t.tick()
        assert t.n.done and t.n.sent_goals==1 and t.n.exit_code==2
        t.close();checks.append('ABORTED stops sequence without falsely completing room')

        t=Trial(Path(tmp)/'external',True);t.until_plan();t.plan_ok();h=t.nav_accept();t.nav_finish(h,5)
        for _ in range(6):t.tick()
        assert t.n.done and t.n.sent_goals==1;t.close();checks.append('external canceled stops; no next Goal')

        t=Trial(Path(tmp)/'reject',True);t.until_plan();t.plan_ok()
        t.n.nav_client.requests[-1].future.set_result(Handle(accepted=False));t.tick()
        assert t.n.done and t.n.state=='NAV_REJECTED';t.close();checks.append('rejected navigation exits instead of waiting for nonexistent terminal state')

        t=Trial(Path(tmp)/'stop_pending',True);t.until_plan();t.plan_ok()
        t.n.request_stop('USER_TEST')
        req=t.n.exact_cancel_client.requests[-1]
        assert any(req.goal_info.goal_id.uuid) and req.goal_info.stamp.sec==req.goal_info.stamp.nanosec==0
        h=t.nav_accept();assert len(h.cancels)==1
        t.nav_finish(h,5)
        for _ in range(6):t.tick()
        assert t.n.done and t.n.sent_goals==1;t.close();checks.append('interrupt before acceptance: exact UUID, then cancel late-accepted owned Goal')

        t=Trial(Path(tmp)/'all_complete',True)
        for z in Z:t.cover(z['id'],.95)
        for _ in range(8):t.tick()
        assert t.n.done and t.n.state=='ALL_REQUESTED_ZONES_OBSERVED' and not t.n.nav_client.requests
        t.close();checks.append('already sufficient all zones require no motion')

        t=Trial(Path(tmp)/'data_loss',True);t.until_plan();t.plan_ok();h=t.nav_accept()
        t.tick(health='STALE');t.n.bad_since=time.monotonic()-3;t.tick(health='STALE')
        assert t.n.stop_requested and h.cancels
        t.nav_finish(h,5)
        for _ in range(6):t.tick(health='STALE')
        assert t.n.done and len(t.n.nav_client.requests)==1;t.close();checks.append('stale observations cancel own Goal and halt')

        t=Trial(Path(tmp)/'foreign',True)
        u=UUID();u.uuid=[1]*16;t.n.on_action_status(NS(status_list=[NS(goal_info=NS(goal_id=u),status=2)]));t.tick()
        assert t.n.done and not t.n.nav_client.requests and not t.n.exact_cancel_client.requests
        t.close();checks.append('other active Goal blocks mission; no foreign cancellation')

        t=Trial(Path(tmp)/'stopmissing',True);t.until_plan();t.plan_ok();h=t.nav_accept();t.nav_finish(h)
        t.n.post_result_wall-=31;t.tick(odom=False)
        assert t.n.done and t.n.state=='STOP_UNCONFIRMED_MANUAL_CHECK';t.close();checks.append('no odometry stop confirmation -> no transition')

        t=Trial(Path(tmp)/'clock_reset',True);t.until_plan();t.plan_ok();h=t.nav_accept();t.tick(dt=-5)
        assert t.n.stop_requested and h.cancels
        t.n.cancel_started-=16;t.tick()
        assert t.n.done and t.n.state=='CANCEL_UNCONFIRMED_MANUAL_STOP_REQUIRED';t.close();checks.append('clock reset + cancel timeout fail closed')

        t=Trial(Path(tmp)/'early_blocked',True,True);t.until_plan();t.plan_ok();h=t.nav_accept()
        t.n.early_stop_area_ok=lambda inp:False;t.cover('A',.95)
        for _ in range(8):t.tick(move=.2)
        assert t.n.book.is_complete('A') and not h.cancels and len(t.n.nav_client.requests)==1
        t.n.request_stop('USER_TEST');t.nav_finish(h,5)
        for _ in range(6):t.tick()
        t.close();checks.append('coverage >=90 cannot trigger early cancellation when stopping-area gate fails')

        t=Trial(Path(tmp)/'coverage_restart',True);t.until_plan();t.plan_ok();h=t.nav_accept()
        raw=dict(t.n.matched['status']);raw['session_id']='new-session'
        t.n.on_status(NS(data=json.dumps(raw)));t.tick()
        assert t.n.stop_requested and h.cancels and t.n.sent_goals==1
        t.nav_finish(h,5)
        for _ in range(6):t.tick()
        assert t.n.done;t.close();checks.append('coverage restart during motion cancels and requires explicit mission restart')

        t=Trial(Path(tmp)/'late_accept_failure',True);t.until_plan();t.plan_ok()
        t.n.nav_client.requests[-1].future.set_exception(RuntimeError('DDS response lost'));t.tick()
        assert t.n.stop_requested=='NAV_HANDSHAKE_ERROR' and t.n.exact_cancel_client.requests
        t.n.cancel_started-=16;t.tick()
        assert t.n.done and t.n.state=='CANCEL_UNCONFIRMED_MANUAL_STOP_REQUIRED'
        t.close();checks.append('uncertain goal handshake never silently exits as stopped')

        t=Trial(Path(tmp)/'no_new_coverage',True);t.until_plan();t.plan_ok();h=t.nav_accept();t.nav_finish(h)
        for _ in range(6):t.tick(coverage=False)
        assert t.n.state=='WAIT_STOP_AND_COVERAGE' and t.n.sent_goals==1
        t.n.post_result_ros-=21;t.tick(coverage=False)
        assert t.n.done and t.n.state=='POST_RESULT_COVERAGE_TIMEOUT';t.close();checks.append('post-result coverage must advance; old observations cannot authorize next Goal')

        t=Trial(Path(tmp)/'log',False);t.n.info('a','one',0);t.n.warn('b','two',0);t.n.info('a','three',0)
        t.n.request_stop('USER_TEST');t.tick();t.close();checks.append('strict Humble-like logger callsites, no Node reserved-property collision')

    for s in checks:print('[MOCK PASS]',s)
    print(f'[MOCK ROS PASS] {len(checks)} scenarios. Fake clients only; NO real ROS2, DDS, Gazebo or robot used.')
    return len(checks)

if __name__=='__main__':main()
