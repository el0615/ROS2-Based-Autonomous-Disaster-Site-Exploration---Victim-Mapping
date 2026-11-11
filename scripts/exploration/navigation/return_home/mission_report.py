"""Offline evidence projection. Same BLOCKED policy as live checkpoint output.

Writes NEW report artifacts only, never edits source logs or detector memories.
PNG encoding, SVG overlays and CSV export use the Python standard library.
"""
from __future__ import annotations
import base64, copy, csv, hashlib, html, json, math, re, struct, zlib
from pathlib import Path
from datetime import datetime, timezone
from blocked_policy import normalize_report, normalize_points, VERSION


def pgm_png(path):
    data=path.read_bytes();pos=0
    def token():
        nonlocal pos
        while pos<len(data):
            if data[pos:pos+1]==b'#':
                end=data.find(b'\n',pos);pos=len(data) if end<0 else end+1
            elif chr(data[pos]).isspace():pos+=1
            else:break
        start=pos
        while pos<len(data) and not chr(data[pos]).isspace():pos+=1
        return data[start:pos]
    mode=token();width,height,maximum=(int(token()) for _ in range(3))
    if mode not in (b'P5',b'P2') or not 0<width*height<=20_000_000 or maximum!=255:
        raise ValueError('MAP_REQUIRES_8BIT_PGM')
    if mode==b'P5':
        if data[pos:pos+2]==b'\r\n':pos+=2
        else:pos+=1
        pixels=data[pos:pos+width*height]
    else:pixels=bytes(int(token()) for _ in range(width*height))
    if len(pixels)!=width*height:raise ValueError('TRUNCATED_PGM')
    def chunk(tag,payload):
        return struct.pack('>I',len(payload))+tag+payload+struct.pack('>I',zlib.crc32(tag+payload)&0xffffffff)
    image=b''.join(b'\x00'+pixels[i*width:(i+1)*width] for i in range(height))
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,0,0,0,0))
    return width,height,png+chunk(b'IDAT',zlib.compress(image))+chunk(b'IEND',b'')


def collect(summary):
    region=summary.get('region_review') or {}
    result=copy.deepcopy(region.get('final_results') or {})
    groups=result.setdefault('categories',{})
    groups.setdefault('blocked_points',[])
    # The mutable memory AND immutable evidence are consumed. Missing-map-mask
    # eligibility/old proof flags MUST NOT hide qualified detour observations.
    known={(p.get('source'),p.get('id')) for p in groups['blocked_points']}
    rr=summary.get('reroute_memory_v1_4') or {}
    for r in rr.get('points') or []:
        if ('REROUTE',r.get('id')) not in known:
            groups['blocked_points'].append(dict(copy.deepcopy(r),xy=r.get('point_xy'),source='REROUTE'))
            known.add(('REROUTE',r.get('id')))
    for r in (rr.get('evidence_v1_7') or {}).get('records',[]):
        ident=r.get('record_id')
        if ('REROUTE',ident) in known:continue
        snap=r.get('confirmation') or r.get('first_detection') or {}
        groups['blocked_points'].append(dict(copy.deepcopy(snap),id=ident,xy=snap.get('point_xy'),source='REROUTE',evidence=copy.deepcopy(r)))
        known.add(('REROUTE',ident))
    for r in (region.get('blockage_memory') or {}).get('points',[]):
        if ('PATH_BLOCKAGE',r.get('id')) not in known:
            groups['blocked_points'].append(dict(copy.deepcopy(r),source='PATH_BLOCKAGE'))
            known.add(('PATH_BLOCKAGE',r.get('id')))
    normalized=normalize_report(result)
    normalized['geometry']=result.get('geometry') or region.get('geometry')
    normalized['zone_lifecycle_v187']=copy.deepcopy(summary.get('zone_lifecycle_v187'))
    normalized['zones']=copy.deepcopy(summary.get('zones'))
    normalized['navigation_v189']=copy.deepcopy(summary.get('navigation_v189'))
    normalized['tasks']=copy.deepcopy(region.get('tasks') or [])
    normalized['source_session']=summary.get('source_session_v18')
    normalized['source_policy']=VERSION
    return normalized


def export_report(root,session):
    root,session=Path(root),Path(session)
    source=session/'summary.json'
    if not source.is_file():source=session/'latest_status.json'
    summary=json.loads(source.read_text())
    result=collect(summary)
    result['source_summary_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    # Enrich ONLY a derived report with the latest return state (source unchanged).
    pointer=session/'checkpoints_v18/LATEST.json'
    if pointer.exists():
        from checkpoint import load_checkpoint
        payload,_,_=load_checkpoint(pointer)
        result['navigation_v189']=copy.deepcopy(payload.get('navigation_v189'))
    dest=root/'logs/dn012_v189_reports'/session.name/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    points=result['categories']['blocked_points']
    with (dest/'blocked_points.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','status','source','spatial_zone','destination_zone','x','y','decision_policy'])
        for r in points:
            writer.writerow([r.get('id'),'BLOCKED',r.get('source'),r.get('spatial_zone'),r.get('destination_zone'),*r['xy'],VERSION])
    g=result.get('geometry') or {};w,h=g.get('width'),g.get('height')
    # Never infer a transform / draw on an unverified image.
    mapfile=root/'maps/normal/normal_hospital_navigation.pgm'
    if not mapfile.is_file():
        raise ValueError('REPORT_MAP_MISSING:'+str(mapfile)+'; JSON/CSV exported to '+str(dest))
    # Verify original occupancy image against the actual coverage reference when available.
    status=None
    if pointer.exists():status=payload.get('coverage',{}).get('status_snapshot')
    expected=((status or {}).get('reference_map') or {}).get('reference_pgm_sha256')
    if expected and hashlib.sha256(mapfile.read_bytes()).hexdigest()!=expected:
        raise ValueError('REPORT_MAP_IMAGE_SHA_MISMATCH')
    width,height,png=pgm_png(mapfile)
    if (w,h)!=(width,height):raise ValueError('REPORT_MAP_DIMENSIONS_MISMATCH')
    origin=g['origin'];resolution=g['resolution'];yaw=origin[2] if len(origin)>2 else 0.
    def pix(x,y):
        dx,dy=x-origin[0],y-origin[1]
        return ((math.cos(yaw)*dx+math.sin(yaw)*dy)/resolution,
                h-(-math.sin(yaw)*dx+math.cos(yaw)*dy)/resolution)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" style="width:100%;background:white">',
         f'<image href="data:image/png;base64,{base64.b64encode(png).decode()}" width="{w}" height="{h}"/>']
    # Geometry comes from recorded row-runs; never turn a point into a blocked area.
    for r in result['categories'].get('inaccessible_regions',[]):
        svg.append('<g fill="#b91c1c" fill-opacity=".30">')
        for j,a,b in r.get('cell_runs_ros') or []:
            svg.append(f'<rect x="{a}" y="{h-j-1}" width="{b-a}" height="1"/>')
        svg.append('</g>')
    for i,r in enumerate(points,1):
        x,y=pix(*r['xy']);tip=html.escape(json.dumps(dict(index=i,**r),ensure_ascii=False))
        svg.append(f'<g><title>{tip}</title><circle cx="{x:.3f}" cy="{y:.3f}" r="4" fill="#dc2626" stroke="white" stroke-width="1"/><text x="{x+5:.3f}" y="{y-4:.3f}" font-size="9" fill="#7f1d1d">{i}</text></g>')
    nav=result.get('navigation_v189') or {};home=nav.get('original_start_pose')
    if home:
        x,y=pix(home['x'],home['y']);svg.append(f'<g><title>완전 최초 출발점</title><rect x="{x-5}" y="{y-5}" width="10" height="10" fill="#0369a1"/><text x="{x+7}" y="{y}" font-size="12">HOME</text></g>')
    svg.append('</svg>');svg=''.join(svg)
    (dest/'blocked_map.svg').write_text(svg)
    rows=[]
    for zid,r in ((result.get('zone_lifecycle_v187') or {}).get('zones') or {}).items():
        rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
            (zid,r.get('exploration_status'),r.get('access_status'),r.get('completion_reason')) )+'</tr>')
    heading=f'<h1>DN012 차단지점 통합 결과</h1><p>세션: {html.escape(session.name)} | 차단지점 {len(points)}개. 경로 급변·장애물 차단을 같은 차단지점으로 표시.</p>'
    legend='<p>빨간 점: 차단지점 · 빨간 영역: 저장된 접근불가 영역 · 파란 사각형: 최초 출발점. 점에 마우스를 올리면 원본 근거를 볼 수 있어. 점 하나로 Zone 전체를 차단 처리하지 않아.</p>'
    page='<!doctype html><meta charset="utf-8"><title>DN012 v1.8.9</title><style>body{font-family:system-ui;margin:24px}td,th{padding:6px;border:1px solid #ddd}table{border-collapse:collapse}main{overflow:auto}svg{min-width:1000px}</style>'+heading+legend+'<main>'+svg+'</main><h2>Zone 상태 — 저장된 완료 판정 유지</h2><table><tr><th>Zone</th><th>탐사 상태</th><th>접근 상태</th><th>완료 사유</th></tr>'+''.join(rows)+'</table>'
    (dest/'blocked_map.html').write_text(page)
    return dest/'blocked_map.html'
