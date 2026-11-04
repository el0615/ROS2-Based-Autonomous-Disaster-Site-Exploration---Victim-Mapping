"""Deterministic fake ROS transport. NOT DDS, Gazebo, ROS simulation or physics."""
import sys, types, math
from types import SimpleNamespace as NS

class Future:
    def __init__(self,result=None,done=False):self.value=result;self.complete=done
    def done(self):return self.complete
    def result(self,timeout=None):return self.value
    def set(self,v):self.value=v;self.complete=True
    def cancel(self):self.complete=True
class Handle:
    def __init__(self,accepted=True):self.accepted=accepted;self.result_future=Future();self.cancels=0
    def get_result_async(self):return self.result_future
    def cancel_goal_async(self):
        self.cancels+=1
        return Future(NS(return_code=0,goals_canceling=[]),True)
class Client:
    def __init__(self,*args,**kwargs):self.sent=[];self.handles=[]
    def server_is_ready(self):return True
    def service_is_ready(self):return True
    def send_goal_async(self,goal,**kwargs):
        self.sent.append((goal,kwargs));h=Handle();self.handles.append(h);return Future(h,True)
    def call_async(self,*a):return Future(NS(return_code=0,goals_canceling=[]),True)
class Logger:
    def info(self,msg):pass
    def warning(self,msg):pass
    def error(self,msg):pass
class Publisher:
    def __init__(self):self.messages=[]
    def publish(self,msg):self.messages.append(msg)
class Clock:
    def __init__(self,node):self.node=node
    def now(self):
        t=self.node.sim
        return NS(nanoseconds=int(t*1e9),to_msg=lambda:NS(sec=int(t),nanosec=int((t%1)*1e9)))
class Node:
    def __init__(self,name):self.sim=10.;self.name=name;self.pubs={};self.logger=Logger()
    def get_clock(self):return Clock(self)
    def get_logger(self):return self.logger
    def get_parameter(self,name):return NS(value=True)
    def create_subscription(self,*a,**kw):return NS()
    def create_publisher(self,cls,topic,*args):
        p=Publisher();self.pubs[topic]=p;return p
    def create_service(self,*a,**kw):return NS()
    def create_client(self,*a,**kw):return Client()
    def destroy_node(self):pass
class Time:
    def __init__(self,*args,**kw):pass
class Duration:
    def __init__(self,*args,**kw):pass
class Buffer:
    def __init__(self,*a,**kw):pass
    def lookup_transform(self,*a,**kw):
        return NS(transform=NS(translation=NS(x=2.,y=2.,z=0.),rotation=NS(x=0.,y=0.,z=0.,w=1.)),header=NS(stamp=NS(sec=10,nanosec=0)))
class PoseStamped:
    def __init__(self):
        self.header=NS(frame_id='',stamp=NS(sec=0,nanosec=0))
        self.pose=NS(position=NS(x=0.,y=0.,z=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.))
class Marker:
    DELETEALL=3;ARROW=0;TEXT_VIEW_FACING=9;LINE_STRIP=4;CUBE_LIST=6;ADD=0
    def __init__(self):
        self.header=NS(frame_id='',stamp=None);self.pose=PoseStamped().pose
        self.color=NS(r=0.,g=0.,b=0.,a=0.);self.scale=NS(x=0.,y=0.,z=0.);self.points=[]
class MarkerArray:
    def __init__(self):self.markers=[]
class Goal:
    def __init__(self):self.goal=None;self.pose=None;self.behavior_tree=''
class CancelRequest:
    def __init__(self):self.goal_info=NS(goal_id=NS(uuid=[]),stamp=NS(sec=0,nanosec=0))
def module(name,**items):
    m=types.ModuleType(name);m.__dict__.update(items);sys.modules[name]=m;return m

def install():
    module('rclpy',ok=lambda:True,init=lambda **k:None,shutdown=lambda:None)
    module('rclpy.node',Node=Node);module('rclpy.time',Time=Time);module('rclpy.duration',Duration=Duration)
    module('rclpy.action',ActionClient=Client)
    module('rclpy.qos',QoSProfile=lambda **k:NS(**k),ReliabilityPolicy=NS(RELIABLE=1,BEST_EFFORT=2),
           DurabilityPolicy=NS(TRANSIENT_LOCAL=1,VOLATILE=2),qos_profile_sensor_data=NS())
    module('nav_msgs.msg',OccupancyGrid=type('OccupancyGrid',(),{}),Odometry=type('Odometry',(),{}))
    module('sensor_msgs.msg',LaserScan=type('LaserScan',(),{}))
    module('std_msgs.msg',String=type('String',(),{}))
    module('std_srvs.srv',Trigger=type('Trigger',(),{}))
    module('action_msgs.msg',GoalStatusArray=type('GoalStatusArray',(),{}))
    module('action_msgs.srv',CancelGoal=NS(Request=CancelRequest))
    module('unique_identifier_msgs.msg',UUID=type('UUID',(),{}))
    module('nav2_msgs.action',ComputePathToPose=NS(Goal=Goal),NavigateToPose=NS(Goal=Goal))
    module('geometry_msgs.msg',Point=lambda **k:NS(**k),PoseStamped=PoseStamped)
    module('visualization_msgs.msg',Marker=Marker,MarkerArray=MarkerArray)
    module('tf2_ros',Buffer=Buffer,TransformListener=lambda *a:None,TransformException=type('TransformException',(Exception,),{}))
