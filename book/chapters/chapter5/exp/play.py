"""Interactive NCA viewer.

Left click/drag erases cells, right click implants a new seed.
Run python train.py first to create model.pt.
"""
from __future__ import annotations
import argparse
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox
import numpy as np
import torch
from PIL import Image, ImageTk
from nca import GrowingNCA, render_rgba

class NCALab:
    def __init__(self,root:tk.Tk,checkpoint:Path,scale:int=12):
        torch.set_num_threads(min(torch.get_num_threads(),4))
        pack=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.model=GrowingNCA(**pack['config'])
        self.model.load_state_dict(pack['model'])
        self.model.eval()
        self.size=int(pack.get('grid',32))
        self.scale=scale
        self.x=self.model.seed(self.size)
        self.paused=False
        self.frame=0
        self.root=root
        root.title(f'NCA Lab | trained for {pack.get("steps_trained","?")} steps')
        self.mode=tk.StringVar(value='RGB')
        self.steps_per_frame=tk.IntVar(value=2)
        s=self.size*scale
        self.canvas=tk.Canvas(root,width=s,height=s,bg='#0f1423',highlightthickness=0)
        self.canvas.grid(row=0,column=0,columnspan=5,padx=12,pady=12)
        self.canvas.bind('<Button-1>',self.erase)
        self.canvas.bind('<B1-Motion>',self.erase)
        self.canvas.bind('<Button-3>',self.seed_at)
        ttk.Button(root,text='Reset',command=self.reset).grid(row=1,column=0)
        ttk.Button(root,text='Damage half',command=self.damage_half).grid(row=1,column=1)
        ttk.Button(root,text='Pause / Resume',command=self.toggle).grid(row=1,column=2,columnspan=2)
        ttk.Label(root,text='View').grid(row=2,column=0)
        ttk.Combobox(root,textvariable=self.mode,values=['RGB','Alpha']+
                     [f'Hidden {i}' for i in range(self.model.channels-4)],
                     state='readonly',width=13).grid(row=2,column=1)
        ttk.Label(root,text='Steps / frame').grid(row=2,column=2)
        ttk.Scale(root,from_=1,to=5,orient='horizontal',
                  command=lambda x:self.steps_per_frame.set(max(1,int(round(float(x)))))).grid(row=2,column=3,sticky='ew')
        self.status=ttk.Label(root,text='')
        self.status.grid(row=3,column=0,columnspan=5,pady=5)
        ttk.Label(root,text='Left drag: erase     Right click: live seed').grid(row=4,column=0,columnspan=5)
        self.image_ref=None
        root.after(30,self.tick)

    def reset(self):
        self.x=self.model.seed(self.size)
        self.frame=0

    def damage_half(self):
        with torch.no_grad():
            self.x[:,:,:,:self.size//2]=0

    def toggle(self):
        self.paused=not self.paused

    def pos(self,event):
        col=max(0,min(self.size-1,event.x//self.scale))
        row=max(0,min(self.size-1,event.y//self.scale))
        return col,row

    def erase(self,event):
        col,row=self.pos(event)
        with torch.no_grad():
            self.x[:,:,max(0,row-2):min(self.size,row+3),
                   max(0,col-2):min(self.size,col+3)]=0

    def seed_at(self,event):
        col,row=self.pos(event)
        with torch.no_grad():
            self.x[:,3:,row,col]=1

    def pixels(self):
        mode=self.mode.get()
        if mode=='RGB':
            return render_rgba(self.x)
        channel=3 if mode=='Alpha' else 4+int(mode.split()[-1])
        v=self.x[0,channel].detach().cpu().numpy()
        if channel==3:
            v=np.clip(v,0,1)
        else:
            v=np.clip(.5+.25*v,0,1)
        gray=(v*255).astype(np.uint8)
        return np.stack([gray,gray,gray],axis=-1)

    def tick(self):
        try:
            if not self.paused:
                with torch.no_grad():
                    self.x=self.model.rollout(self.x,self.steps_per_frame.get())
                self.frame+=self.steps_per_frame.get()
            frame=Image.fromarray(self.pixels()).resize(
                (self.size*self.scale,self.size*self.scale),Image.Resampling.NEAREST)
            self.image_ref=ImageTk.PhotoImage(frame)
            self.canvas.delete('frame')
            self.canvas.create_image(0,0,anchor='nw',image=self.image_ref,tags='frame')
            self.status.config(text=f'Steps: {self.frame} | '+('Paused' if self.paused else 'Running'))
        except Exception as exc:
            self.paused=True
            messagebox.showerror('NCA error',str(exc))
        self.root.after(40,self.tick)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path('model.pt'))
    parser.add_argument('--scale',type=int,default=12)
    args=parser.parse_args()
    if not args.model.is_file():
        parser.error('checkpoint not found; run python train.py --steps 1200 first')
    root=tk.Tk()
    root.resizable(False,False)
    NCALab(root,args.model,args.scale)
    root.mainloop()

if __name__=='__main__':
    main()
