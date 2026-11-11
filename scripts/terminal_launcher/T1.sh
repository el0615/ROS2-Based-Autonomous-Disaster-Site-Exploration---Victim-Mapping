#!/usr/bin/env bash
# T1: user-verified Gazebo entrypoint
cd "$PROJECT" || exit 1
exec bash ./scripts/launch_hospital.sh disaster
