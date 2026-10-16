"""Archive-only reconstruction, static figures or animations; never runs PDE."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from engine.archive import read_archive,decode,reconstruct_grid
from .spatial import validate_view, draw_field

# Render the cell-average archive representation without a spectral transform; select a
# saved index or animate stored snapshots.
def plot_grid_archive(path,view_dimension=None,grid_points=80,frame_index=None,save=None,fps=12,color_scale='frame',slice_fractions=(.5,.5,.5)):
    data=read_archive(path); grid=decode(data,'grid_json'); times=data['time']
    names=list(dict.fromkeys(data['fields'])); dim=len(grid['shape'])
    view=validate_view(grid['shape'],view_dimension,grid_points,slice_fractions)
    coords=[data[f'grid_x{i+1}'] for i in range(dim)]
    if fps<=0 or color_scale not in ('frame','global'): raise ValueError('Invalid viewer options')
    cols=min(3,len(names)); rows=(len(names)+cols-1)//cols
    fig,axs=plt.subplots(rows,cols,figsize=(5*cols,3.6*rows),squeeze=False,subplot_kw={'projection':'3d'} if view==3 else {})
    axes=axs.ravel(); registry=decode(data,'field_metadata_json')
    # One reshape is a view, not a recomputation of the trajectory.
    values=data['coefficients'].reshape((len(times),len(names))+tuple(grid['shape']))
    limits=[(float(values[:,i].min()),float(values[:,i].max())) for i in range(len(names))]
    colorbars={}
    # Extract and display a saved grid frame, retaining stable field identity and updating
    # labels/colorbars.
    def draw(k):
        for i,(name,axis) in enumerate(zip(names,axes)):
            axis.clear(); u=values[k,i]
            lo,hi=limits[i] if color_scale=='global' else (float(u.min()),float(u.max()))
            hi=max(hi,lo+1e-8)
            artist=draw_field(axis,u,coords,grid['lengths'],view,(lo,hi),grid_points,slice_fractions,cell_average=True)
            if artist is not None:
                if i not in colorbars: colorbars[i]=fig.colorbar(artist,ax=axis,shrink=.8)
                else: colorbars[i].update_normal(artist)
            f=registry[i]; status='unborn' if times[k]<f['birth_time'] else ('inactive' if f['death_time'] is not None and times[k]>=f['death_time'] else 'active')
            axis.set_title(f'{f.get("display_label",name)} [{name}] | {status} | t={times[k]:.4g}')
        for axis in axes[len(names):]: axis.set_visible(False)
        fig.tight_layout()
        return axes
    if frame_index is not None:
        if not 0<=frame_index<len(times): raise ValueError('Frame out of range')
        draw(frame_index)
        if save: fig.savefig(save,dpi=120)
        else: plt.show()
    else:
        anim=FuncAnimation(fig,draw,frames=len(times),interval=1000/fps)
        if save:
            suffix=Path(save).suffix.lower()
            if suffix not in ('.gif','.mp4'): raise ValueError('Animation needs .gif/.mp4')
            anim.save(save,writer='pillow' if suffix=='.gif' else 'ffmpeg',fps=fps)
        else: plt.show()
    plt.close(fig)
