#!/usr/bin/env python3
# core/simulator.py
import os, json
from typing import List, Dict
import cv2 as cv

def _clamp(v,a,b): return a if v<a else (b if v>b else v)

def _draw_label(img,text,org,color,scale=0.9,thick=2):
    (tw,th),base=cv.getTextSize(text,cv.FONT_HERSHEY_SIMPLEX,scale,thick)
    x,y=int(org[0]),int(org[1]); pad=6
    cv.rectangle(img,(x-2,y-th-pad),(x+tw+2,y+base),(0,0,0),-1,cv.LINE_AA)
    cv.putText(img,text,(x,y),cv.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv.LINE_AA)

class DetectionSimulator:
    """
    Interactive drawing tool to create bboxes on raw frames.
    Output format matches detector results: [{image_index, bbox, label, score}, ...]
    """
    def run(self, index_json:dict, out_path:str="detections.json",
            start:int=0, end:int=None, label:str="manual", score:float=1.0) -> List[Dict]:

        frames=index_json["frames"]
        total=len(frames)
        start=_clamp(start,0,total-1)
        end=_clamp(end if end is not None else total-1, start, total-1)

        # restore previous annotations (if file exists)
        boxes_by_frame={i:[] for i in range(total)}
        if os.path.exists(out_path):
            old=json.load(open(out_path,"r"))
            for d in old:
                i=int(d.get("image_index",-1)); b=d.get("bbox",None)
                if 0<=i<total and b and len(b)==4:
                    boxes_by_frame[i].append([int(b[0]),int(b[1]),int(b[2]),int(b[3])])

        i=start
        img=cv.imread(frames[i]["path"], cv.IMREAD_COLOR)
        if img is None: raise SystemExit(f"cannot read {frames[i]['path']}")
        H,W=img.shape[:2]
        winW,winH=min(1600,W),min(1000,H)
        cx,cy=W/2.0,H/2.0; z=1.0
        drawing={"active":False,"x0":0,"y0":0,"x1":0,"y1":0}

        def vp():
            vw=max(100,int(W/z)); vh=max(100,int(H/z))
            x0=int(_clamp(cx-vw/2,0,W-vw)); y0=int(_clamp(cy-vh/2,0,H-vh))
            return x0,y0,vw,vh

        def s2i(px,py,x0,y0,vw,vh):
            sx=vw/float(winW); sy=vh/float(winH)
            u=x0+px*sx; v=y0+py*sy
            return float(_clamp(u,0,W-1)), float(_clamp(v,0,H-1))

        def render():
            base=img.copy()
            for k,b in enumerate(boxes_by_frame[i],1):
                x1,y1,x2,y2=b
                cv.rectangle(base,(x1,y1),(x2,y2),(0,255,0),3,cv.LINE_AA)
                _draw_label(base,f'#{k}',(x1+10,y1+34),(0,255,0),1.0,2)
            if drawing["active"]:
                x1,y1=min(drawing["x0"],drawing["x1"]),min(drawing["y0"],drawing["y1"])
                x2,y2=max(drawing["x0"],drawing["x1"]),max(drawing["y0"],drawing["y1"])
                cv.rectangle(base,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2,cv.LINE_AA)
            x0,y0,vw,vh=vp()
            crop=base[y0:y0+vh, x0:x0+vw]
            disp=cv.resize(crop,(winW,winH),interpolation=cv.INTER_LINEAR)
            _draw_label(disp,f'frame {i+1}/{total}  z={z:.2f}  boxes={len(boxes_by_frame[i])}',(12,30),(255,255,255),0.8,2)
            _draw_label(disp,"keys: n/p next-prev  u undo  c clear  s save  +/- zoom  WASD pan  r reset  q quit",(12,winH-12),(255,255,255),0.7,2)
            return disp

        def load_frame(j):
            nonlocal img,H,W,winW,winH,cx,cy,z
            im=cv.imread(frames[j]["path"], cv.IMREAD_COLOR)
            if im is None: raise SystemExit(f"cannot read {frames[j]['path']}")
            img=im; H,W=img.shape[:2]
            winW,winH=min(1600,W),min(1000,H)
            cx,cy=W/2.0,H/2.0; z=1.0

        def save_all(path):
            out=[]
            for fi in range(total):
                for b in boxes_by_frame[fi]:
                    out.append({"image_index":fi,"bbox":[int(b[0]),int(b[1]),int(b[2]),int(b[3])],"label":label,"score":float(score)})
            with open(path,"w") as f: json.dump(out,f,indent=2)
            print(f"[sim] wrote {path} ({len(out)} boxes)")

        win="Annotate: drag to box; keys shown on screen"
        cv.namedWindow(win, cv.WINDOW_NORMAL)
        cv.resizeWindow(win, winW, winH)

        def on_mouse(event,x,y,flags,param):
            nonlocal drawing
            x0,y0,vw,vh=vp()
            if event==cv.EVENT_LBUTTONDOWN:
                u,v=s2i(x,y,x0,y0,vw,vh)
                drawing={"active":True,"x0":u,"y0":v,"x1":u,"y1":v}
            elif event==cv.EVENT_MOUSEMOVE and drawing["active"]:
                u,v=s2i(x,y,x0,y0,vw,vh); drawing["x1"]=u; drawing["y1"]=v
            elif event==cv.EVENT_LBUTTONUP and drawing["active"]:
                u,v=s2i(x,y,x0,y0,vw,vh); drawing["x1"]=u; drawing["y1"]=v
                x1,y1=min(drawing["x0"],drawing["x1"]),min(drawing["y0"],drawing["y1"])
                x2,y2=max(drawing["x0"],drawing["x1"]),max(drawing["y0"],drawing["y1"])
                if (x2-x1)>=4 and (y2-y1)>=4:
                    boxes_by_frame[i].append([int(round(x1)),int(round(y1)),int(round(x2)),int(round(y2))])
                drawing={"active":False,"x0":0,"y0":0,"x1":0,"y1":0}

        cv.setMouseCallback(win,on_mouse)

        while True:
            disp=render(); cv.imshow(win, disp)
            k=cv.waitKey(30)&0xFF
            if k in (ord('q'),27): save_all(out_path); break
            if k==ord('s'): save_all(out_path)
            if k in (ord('n'), ord('N')):
                i = i+1 if i<end else start; load_frame(i)
            if k in (ord('p'), ord('P')):
                i = i-1 if i>start else end; load_frame(i)
            if k in (ord('u'), ord('U')) and boxes_by_frame[i]: boxes_by_frame[i].pop()
            if k in (ord('c'), ord('C')): boxes_by_frame[i]=[]
            if k in (ord('+'),ord('=')): z=min(16.0,z*1.25)
            if k in (ord('-'),ord('_')): z=max(1.0,z/1.25)
            step=max(20,int(min(W,H)/(10*z)))
            if k in (ord('a'),ord('A')): cx-=step
            if k in (ord('d'),ord('D')): cx+=step
            if k in (ord('w'),ord('W')): cy-=step
            if k in (ord('s'),ord('S')): cy+=step
            if k in (ord('r'),ord('R')): cx,cy,z=W/2.0,H/2.0,1.0
            cx=_clamp(cx,0,W); cy=_clamp(cy,0,H)

        cv.destroyAllWindows()

        # Return detections as list (also saved to disk above)
        out=[]
        for fi in range(total):
            for b in boxes_by_frame[fi]:
                out.append({"image_index":fi,"bbox":[int(b[0]),int(b[1]),int(b[2]),int(b[3])],
                            "label":label,"score":float(score)})
        return out
