"""Read-only spectral archive/run visualization. Model source is not imported."""
# Rendering consumes saved coefficients in manifest order; it must not advance
# a solver. Viewer grid/slices change display sampling, not simulation accuracy.
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from engine.archive import read_archive,decode
from ..galerkin.basis import TensorBasis
from .spatial import validate_view, draw_field

class ArchiveSequence:
    """Keep only the currently displayed archive/basis in memory.

    Pre- and post-event boundary frames are both retained, deliberately showing
    the event map at equal physical time. No interpolation through event jumps.
    """
    # Index saved times in manifest order, retaining duplicate event-boundary frames and
    # an optional terminal state. Keep only one reconstructed archive cached.
    def __init__(self,path):
        path=Path(path);self.frames=[];self.cached_path=None;terminal=None
        if path.suffix.lower()=='.json':
            manifest=json.loads(path.read_text(encoding='utf-8'))
            required={'field_registry','segments'}
            if not isinstance(manifest,dict) or required-manifest.keys():
                raise ValueError('Invalid run manifest: field_registry and segments are required')
            if not isinstance(manifest['segments'],list) or not isinstance(manifest['field_registry'],list):
                raise ValueError('Invalid run manifest: segments and field_registry must be lists')
            if any(not isinstance(s,dict) or not {'archive','id'}<=s.keys() for s in manifest['segments']):
                raise ValueError('Invalid run manifest: each segment requires archive and id')
            if any(not isinstance(f,dict) or 'id' not in f for f in manifest['field_registry']):
                raise ValueError('Invalid run manifest: registry entries require id')
            self.names=[f['id'] for f in manifest['field_registry']]
            self.labels={f['id']:f.get('display_label',f['id']) for f in manifest['field_registry']}
            sources=[(path.parent/s['archive'],s['id']) for s in manifest['segments']]
            if 'terminal_state' in manifest:terminal=path.parent/manifest['terminal_state']
        else:
            data=read_archive(path);self.names=list(dict.fromkeys(data['fields']))
            self.labels={f['id']:f.get('display_label',f['id']) for f in decode(data,'field_metadata_json',[])}
            sources=[(path,0)]
        for source,segment in sources:
            if not source.is_file():
                raise ValueError(f'Run manifest references missing archive: {source}')
            with np.load(source,allow_pickle=False) as z:
                times=z['time']
            self.frames.extend((source,k,float(t),segment) for k,t in enumerate(times))
        if terminal is not None:
            with np.load(terminal,allow_pickle=False) as z:time=float(z['time'])
            self.frames.append((terminal,0,time,'terminal'))
        if not self.frames:raise ValueError('No trajectory frames in run')

    # Return field-ID to grid-array values, Grid, actual time and segment ID for one
    # frame. Read terminal physical values directly; otherwise reconstruct stored
    # coefficients.
    def frame(self,index):
        path,k,time,segment=self.frames[index]
        if segment=='terminal':
            from ..simulation.grid import Grid
            with np.load(path,allow_pickle=False) as z:
                grid=Grid(z['grid_shape'].tolist(),z['lengths'].tolist())
                values={str(name):u for name,u,active in zip(z['fields'],z['values'],z['active']) if active}
            return values,grid,time,segment
        if path!=self.cached_path:
            self.data=read_archive(path)
            self.basis=TensorBasis.from_descriptor(decode(self.data,'basis_json'))
            self.current_names=list(dict.fromkeys(self.data['fields']))
            self.cached_path=path
        values=self.basis.reconstruct(self.data['coefficients'][k].reshape((len(self.current_names),)+self.basis.shape))
        return {name:u for name,u in zip(self.current_names,values)},self.basis.grid,time,segment

    def select_time(self, requested_time, mode='nearest', event_side='after'):
        """Return a saved-frame index for a finite model time, without interpolation.

        nearest chooses the closest time (earlier time wins equal distances).
        before/after choose the last/first saved time on that side of the request.
        At duplicated event times event_side chooses the first/last frame in
        manifest order. 'after' is the default, including terminal-state frames.
        Requests outside the saved time range fail instead of silently clamping.
        """
        requested_time = float(requested_time)
        if not np.isfinite(requested_time):
            raise ValueError('Requested time must be finite')
        if mode not in ('nearest', 'before', 'after'):
            raise ValueError('time_mode must be nearest, before or after')
        if event_side not in ('before', 'after'):
            raise ValueError('event_side must be before or after')
        times = np.array([frame[2] for frame in self.frames])
        if not np.isfinite(times).all():
            raise ValueError('Archive contains nonfinite frame times')
        lower, upper = float(times.min()), float(times.max())
        if not lower <= requested_time <= upper:
            raise ValueError(f'Requested time {requested_time:g} outside saved range [{lower:g}, {upper:g}]')
        if mode == 'before':
            chosen = float(times[times <= requested_time].max())
        elif mode == 'after':
            chosen = float(times[times >= requested_time].min())
        else:
            chosen = min(map(float, times), key=lambda value: (abs(value-requested_time), value))
        candidates = np.flatnonzero(times == chosen)
        return int(candidates[0] if event_side == 'before' else candidates[-1])


def plot_spectral(path,view_dimension=None,grid_points=80,frame_index=None,save=None,fps=12,color_scale='frame',slice_fractions=(.5,.5,.5),time=None,time_mode='nearest',event_side='after'):
    """Render stored fields; select by index OR model time, otherwise animate.

    New time options are trailing arguments to preserve existing callers.
    Returns selection metadata for a still frame, None for animation. No PDE
    solver or model source is loaded. The title always reports the actual time.
    """
    sequence=ArchiveSequence(path)
    if time is not None and frame_index is not None:
        raise ValueError('Choose frame_index or time, not both')
    if time is None and (time_mode != 'nearest' or event_side != 'after'):
        raise ValueError('time_mode/event_side require time')
    if time is not None:
        frame_index=sequence.select_time(time,time_mode,event_side)
    selection=None
    if frame_index is not None:
        if not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<len(sequence.frames):
            raise ValueError('Frame out of range')
        _,_,actual_time,segment=sequence.frames[frame_index]
        selection={'frame_index':int(frame_index),'requested_time':None if time is None else float(time),
                   'actual_time':actual_time,'segment':segment,'time_mode':time_mode,'event_side':event_side}
    if save:
        save=Path(save)
        save.parent.mkdir(parents=True,exist_ok=True)
    if fps<=0 or color_scale not in ('frame','global'):raise ValueError('Invalid viewer options')
    _,first_grid,_,_=sequence.frame(0)
    dim=validate_view(first_grid.shape,view_dimension,grid_points,slice_fractions)
    cols=min(3,len(sequence.names));rows=(len(sequence.names)+cols-1)//cols
    figure,axs=plt.subplots(rows,cols,figsize=(5*cols,3.6*rows),squeeze=False,subplot_kw={'projection':'3d'} if dim==3 else {})
    axes=axs.ravel();colorbars={};limits={}
    if color_scale=='global':
        # Optional read-only pass to find extrema; never calls the model/RHS.
        for k in range(len(sequence.frames)):
            values,_,_,_=sequence.frame(k)
            for name,u in values.items():
                lo,hi=limits.get(name,(0.,0.));limits[name]=(min(lo,float(u.min())),max(hi,float(u.max())))
    # Draw one stored frame using actual time and field presence; keep one colorbar per
    # field and never interpolate between segments.
    def draw(k):
        values,grid,time,segment=sequence.frame(k)
        for ax,name in zip(axes,sequence.names):
            ax.clear();present=name in values
            u=values.get(name,np.zeros(grid.shape))
            lo,hi=limits.get(name,(float(u.min()),float(u.max())))
            if hi-lo<1e-10:lo,hi=min(0.,lo),max(1.,hi)
            artist=draw_field(ax,u,grid.coords,grid.lengths,dim,(lo,hi),grid_points,slice_fractions)
            if artist is not None:
                if name not in colorbars:colorbars[name]=figure.colorbar(artist,ax=ax,shrink=.8)
                else:colorbars[name].update_normal(artist)
            ax.set_title(f'{sequence.labels.get(name,name)} [{name}] | {"active" if present else "absent"} | segment {segment} | t={time:.10g}')
        for ax in axes[len(sequence.names):]:ax.set_visible(False)
        figure.tight_layout();return axes
    if frame_index is not None:
        if not 0<=frame_index<len(sequence.frames):raise ValueError('Frame out of range')
        draw(frame_index)
        if save:figure.savefig(save,dpi=120)
        else:plt.show()
    else:
        animation=FuncAnimation(figure,draw,frames=len(sequence.frames),interval=1000/fps)
        if save:
            suffix=Path(save).suffix.lower()
            if suffix not in ('.gif','.mp4'):raise ValueError('Animation output must be GIF/MP4')
            animation.save(save,writer='pillow' if suffix=='.gif' else 'ffmpeg',fps=fps)
        else:plt.show()
    plt.close(figure)
    return selection
