"""Train a minimal NCA to grow a small procedurally defined flower."""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from nca import GrowingNCA, export_config, flower_target, render_rgba

def damage_random_rectangle(x: torch.Tensor, count: int) -> torch.Tensor:
    """Erase all channels inside a randomly positioned rectangle."""
    _, _, h, w = x.shape
    for k in range(min(count, x.shape[0])):
        rh = random.randint(max(1, h // 4), max(1, h // 2))
        rw = random.randint(max(1, w // 4), max(1, w // 2))
        top = random.randint(0, h - rh)
        left = random.randint(0, w - rw)
        x[k, :, top:top+rh, left:left+rw] = 0
    return x

def save_checkpoint(path: Path, model: GrowingNCA, *, step: int, loss: float, seed: int, grid: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model":model.state_dict(),"config":export_config(model),
                "steps_trained":step,"loss":loss,"seed":seed,"grid":grid},path)

def train(args: argparse.Namespace):
    if args.grid < 8 or args.batch < 1 or args.pool < args.batch:
        raise ValueError("grid>=8, batch>=1, and pool>=batch required")
    if not 1 <= args.rollout_min <= args.rollout_max:
        raise ValueError("invalid rollout length")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    device = torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else
                          'cpu' if args.device=='auto' else args.device)
    model = GrowingNCA(channels=args.channels,hidden=args.hidden,fire_rate=args.fire_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    target = flower_target(args.grid).to(device)[None]
    pool = model.seed(args.grid,args.pool,'cpu')
    output = Path(args.output)
    start = time.perf_counter()
    last_loss = float('nan')
    for step in range(1,args.steps+1):
        indices = np.random.choice(args.pool,args.batch,replace=False)
        states = pool[indices].clone()
        # Each batch includes one fresh growth episode.
        states[0:1] = model.seed(args.grid)
        if args.damage and args.batch>1:
            damage_random_rectangle(states[1:],max(1,(args.batch-1)//2))
        states = states.to(device)
        count = random.randint(args.rollout_min,args.rollout_max)
        evolved = model.rollout(states,count)
        loss = ((model.rgba(evolved)-target)**2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        pool[indices] = evolved.detach().cpu()
        pool[0:1] = model.seed(args.grid)
        if step==1 or step%args.log_every==0 or step==args.steps:
            print(f'step {step:5d}/{args.steps} loss={last_loss:.6f} '
                  f'rollout={count} seconds={time.perf_counter()-start:.1f}',flush=True)
        if step%args.save_every==0 or step==args.steps:
            save_checkpoint(output,model,step=step,loss=last_loss,seed=args.seed,grid=args.grid)
            with torch.no_grad():
                picture = model.rollout(model.seed(args.grid,device=device),args.preview_steps)
            Image.fromarray(render_rgba(picture)).save(output.with_name('preview.png'))
    summary={"steps":args.steps,"last_loss":last_loss,"seconds":time.perf_counter()-start,
             "device":str(device),"config":export_config(model),"grid":args.grid}
    output.with_name('train_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(f'Model saved to: {output.resolve()}')

def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--grid',type=int,default=32)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--pool',type=int,default=32)
    parser.add_argument('--channels',type=int,default=12)
    parser.add_argument('--hidden',type=int,default=48)
    parser.add_argument('--fire-rate',type=float,default=0.5)
    parser.add_argument('--rollout-min',type=int,default=24)
    parser.add_argument('--rollout-max',type=int,default=48)
    parser.add_argument('--preview-steps',type=int,default=64)
    parser.add_argument('--lr',type=float,default=0.002)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--output',default='model.pt')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--log-every',type=int,default=50)
    parser.add_argument('--save-every',type=int,default=100)
    parser.add_argument('--no-damage',action='store_false',dest='damage')
    return parser.parse_args()

if __name__=='__main__':
    train(parse_args())
