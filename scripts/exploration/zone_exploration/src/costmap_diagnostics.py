#!/usr/bin/env python3
"""Pure, allocation-light observability helper for existing costmap freshness rules.

Never changes validity, thresholds, action state, or the subscribed QoS.
All ages are seconds; a negative ros_age means the message stamp is in the future.
"""
from __future__ import annotations


def rounded(value):
    return None if value is None else round(float(value),4)


def snapshot_costmap_freshness(cost,monitor,now_ros_s,now_wall_s):
    """JSON serializable diagnostic snapshot of the exact preexisting threshold predicate."""
    ros_age=None if cost is None else float(now_ros_s)-float(cost['stamp'])
    wall_age=None if cost is None else float(now_wall_s)-float(cost['wall'])
    future=ros_age is not None and ros_age < -0.1
    ros_old=ros_age is not None and ros_age > 5.0
    wall_old=wall_age is not None and wall_age > 15.0
    if cost is None:cause='NO_VALID_COSTMAP'
    elif future:cause='HEADER_IN_FUTURE'
    elif ros_old and wall_old:cause='ROS_AND_WALL_STALE'
    elif ros_old:cause='ROS_HEADER_STALE'
    elif wall_old:cause='NO_RECENT_VALID_RECEIPT'
    else:cause='WITHIN_LIMITS'
    return dict(
        cause=cause,
        ros_now_s=rounded(now_ros_s),
        last_valid_header_stamp_s=None if cost is None else rounded(cost['stamp']),
        ros_age_s=rounded(ros_age),
        wall_age_s=rounded(wall_age),
        ros_age_limit_s=5.0,
        wall_age_limit_s=15.0,
        max_future_tolerance_s=0.1,
        ros_stale=bool(future or ros_old),
        wall_stale=bool(wall_old),
        received_count=int(monitor.get('received_count',0)),
        accepted_count=int(monitor.get('accepted_count',0)),
        invalid_geometry_count=int(monitor.get('invalid_geometry_count',0)),
        invalid_values_count=int(monitor.get('invalid_values_count',0)),
        last_received_header_stamp_s=rounded(monitor.get('last_header_stamp_s')),
        last_any_receive_age_wall_s=rounded(None if monitor.get('last_receive_wall') is None
            else float(now_wall_s)-float(monitor['last_receive_wall'])),
        last_valid_receive_age_wall_s=rounded(None if monitor.get('last_valid_receive_wall') is None
            else float(now_wall_s)-float(monitor['last_valid_receive_wall'])),
        last_interarrival_wall_s=rounded(monitor.get('last_interarrival_wall_s')),
        max_interarrival_wall_s=rounded(monitor.get('max_interarrival_wall_s',0.0)))
