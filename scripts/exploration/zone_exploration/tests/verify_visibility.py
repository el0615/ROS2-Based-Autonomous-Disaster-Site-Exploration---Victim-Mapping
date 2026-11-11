#!/usr/bin/env python3
"""Deterministic synthetic Python-vs-C equivalence checks; no live ROS data."""
from fixtures import *
from planning_core import visibility_cells as reference, SensorModel
from zb_visibility_c import visibility_cells as accelerated
import json, time

def run():
    rng=np.random.default_rng(11012026)
    calls=0;reference_s=fast_s=0.
    for case in range(160):
        w=int(rng.integers(24,75));h=int(rng.integers(24,70));res=float(rng.choice([.05,.1,.2]))
        g=Geometry(w,h,res,(float(rng.uniform(-10,10)),float(rng.uniform(-10,10)),float(rng.uniform(-math.pi,math.pi))))
        free=rng.random((h,w))>.08
        target=rng.random((h,w))>.4
        i=int(rng.integers(2,w-2));j=int(rng.integers(2,h-2));free[j-1:j+2,i-1:i+2]=True
        x,y=g.cell_to_world(i,j)
        beams=int(rng.choice([36,90,181,360,540]))
        span=float(rng.choice([math.pi,1.5*math.pi,2*math.pi]))
        sensor=SensorModel(-span/2,span/beams,beams,float(rng.choice([0.,.05,.1])),8.,
            float(rng.uniform(-.05,.05)),float(rng.uniform(-.05,.05)),float(rng.uniform(-.3,.3)))
        for yaw in (float(rng.uniform(-math.pi,math.pi)),g.origin[2]):
            st=time.perf_counter();a=reference(g,free,target,x,y,yaw,sensor,8.);reference_s+=time.perf_counter()-st
            st=time.perf_counter();b=accelerated(g,free,target,x,y,yaw,sensor,8.);fast_s+=time.perf_counter()-st
            if not np.array_equal(a,b):raise AssertionError(f'VISIBILITY_MISMATCH case={case} yaw={yaw}')
            calls+=1
    # Out-of-map, blocked origin, empty target and range smaller than minimum.
    g=Geometry(30,30,.1);free=np.ones((30,30),bool);target=free.copy()
    sensor=SensorModel(-math.pi,math.pi/90,180,.2,8.)
    for x,y,cap in [(-1.,-1.,8.),(1.5,1.5,.1),(0.,0.,8.),(3.,3.,8.)]:
        a=reference(g,free,target,x,y,0.,sensor,cap);b=accelerated(g,free,target,x,y,0.,sensor,cap)
        np.testing.assert_array_equal(a,b);calls+=1
    result=dict(result='PASS',calls=calls,mismatches=0,reference_wall_s=reference_s,fast_wall_s=fast_s,
                scope='Synthetic cases only, not a proof for all inputs or navigation safety')
    print('[VISIBILITY_EQUIVALENCE_PASS] '+json.dumps(result),flush=True)
    return result

if __name__=='__main__':run()
