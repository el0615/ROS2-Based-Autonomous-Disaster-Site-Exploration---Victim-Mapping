import json
import math


def stamp_s(stamp): return float(stamp.sec)+float(stamp.nanosec)*1e-9

def geometry_matches(msg, info):
    m=msg.info;p=m.origin.position;q=m.origin.orientation
    norm=math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
    if norm<1e-9:return False
    yaw=math.atan2(2*q.w*q.z,q.w*q.w-q.z*q.z)
    delta=math.atan2(math.sin(yaw-info['origin'][2]),math.cos(yaw-info['origin'][2]))
    return ((m.width,m.height)==(info['width'],info['height'])
        and len(msg.data)==m.width*m.height
        and abs(m.resolution-info['resolution'])<1e-7
        and math.hypot(p.x-info['origin'][0],p.y-info['origin'][1])<1e-4
        and abs(p.z)<1e-4 and abs(q.x)<1e-5 and abs(q.y)<1e-5
        and abs(delta)<1e-5 and msg.header.frame_id.lstrip('/')==info['frame_id'])

def atomic_json(path, data):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)
