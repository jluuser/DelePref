"""Actual frozen Sec6 case visualization; no models, retrieval or metric resampling."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
REL=Path('.')
W,H=174.,120.
METHODS=['delepref_unlimited','rag_unlimited']
COL=dict(ink='#263244',teal='#087F78',seed='#D88935',base='#8193A7',native='#C3CBD0',deleted='#E0E4E7',unrecovered='#9BA9B7')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def spans(row):
    value=row['deleted_intervals']
    return json.loads(value) if isinstance(value,str) else value

def positions(x):
    return {i for a,b in x for i in range(a,b+1)}

def smooth(points, subdivisions=5):
    """Display spline of actual CA points, not atom reconstruction."""
    import numpy as np
    points=np.asarray(points,float)
    if len(points)<2:return points
    p=np.vstack([points[0],points,points[-1]]);result=[]
    for i in range(len(points)-1):
        a,b,c,d=p[i:i+4]
        for t in np.linspace(0,1,subdivisions,endpoint=False):
            result.append(.5*((2*b)+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t))
    return np.asarray(result+[points[-1]])

def molecules(root):
    import numpy as np
    import pymol2
    from pymol.cgo import CYLINDER,SPHERE,COLOR,ALPHA
    out=root/REL;d=read(out/'data/case_data.json');folder=out/'molecules'
    rgb=lambda value:[int(value[i:i+2],16)/255 for i in [1,3,5]]
    native=np.asarray(d['native_coordinates']);fitted={}
    for method in METHODS:
        fitted[method]=np.asarray([v if v is not None else [np.nan]*3 for v in d['views'][method]['fitted_WT_frame']],float)

    def ca_trace(xyz,radius,color,alpha=1.):
        obj=[ALPHA,alpha,COLOR,*color];runs=[];run=[]
        for point in xyz:
            if not np.isfinite(point).all():
                if run:runs.append(run);run=[]
                continue
            if run and np.linalg.norm(point-run[-1])>5.:
                runs.append(run);run=[]
            run.append(point)
        if run:runs.append(run)
        for run in runs:
            a=smooth(run)
            for p,q in zip(a[:-1],a[1:]):obj.extend([CYLINDER,*p,*q,radius,*color,*color])
            for p in a:obj.extend([SPHERE,*p,radius])
        return obj

    def line(obj,p,q,radius,color,dashed=False):
        if dashed:
            # Fixed physical dash length in Angstrom; not a covalent bond.
            length=float(np.linalg.norm(q-p));steps=max(2,int(length/.9))
            for j in range(0,steps,2):
                a=p+(j/steps)*(q-p);b=p+(min(j+1,steps)/steps)*(q-p)
                obj.extend([CYLINDER,*a,*b,radius,*color,*color])
        else:obj.extend([CYLINDER,*p,*q,radius,*color,*color])

    with pymol2.PyMOL() as session:
        c=session.cmd;c.load(str(out/'data/2FJI_domain_WT.pdb'),'native')
        c.hide('everything','all');c.dss('native');c.show('cartoon','native')
        for name,value in COL.items():c.set_color('fig_'+name,rgb(value))
        c.color('fig_native','native')
        sel=lambda xs:'native and resi '+'+'.join(str(i+1) for i in xs)
        c.color('fig_seed',sel(d['seed_positions_0based']))
        c.color('fig_teal',sel(d['completion_positions_0based']))
        c.set('cartoon_fancy_helices',1);c.set('cartoon_flat_sheets',1);c.set('cartoon_sampling',20)
        c.set('cartoon_loop_radius',.22);c.set('orthoscopic',1);c.set('ray_shadows',0)
        c.set('ray_trace_fog',0);c.set('specular',.08);c.set('ambient',.60);c.set('direct',.35)
        c.set('ray_opaque_background',0);c.set('antialias',2);c.set('max_threads',2);c.bg_color('white')
        c.load_cgo(ca_trace(native,.14,rgb(COL['native']),.65),'native_trace')
        context=sorted({j for i in d['ROI_positions'] for j in range(max(0,i-2),min(209,i+3))})
        local_native=native.copy();local_native[[i for i in range(209) if i not in context]]=np.nan
        c.load_cgo(ca_trace(local_native,.16,rgb(COL['native']),.75),'native_roi_trace')
        seed_native=native.copy();seed_native[[i for i in range(209) if i not in set(d['seed_positions_0based'])&set(context)]]=np.nan
        c.load_cgo(ca_trace(seed_native,.23,rgb(COL['seed'])),'native_roi_seed')
        for method in METHODS:
            mapping=d['WT_to_edit'][method];order=[i for i,j in enumerate(mapping) if j is not None]
            # True edited-chain order; no made-up N/C/O or secondary structure transfer.
            c.load_cgo(ca_trace(fitted[method][order],.48,rgb(COL['base'] if method==METHODS[0] else COL['teal'])),method)
            local=fitted[method][order].copy()
            local[[j for j,i in enumerate(order) if i not in context]]=np.nan
            c.load_cgo(ca_trace(local,.40,rgb(COL['base'] if method==METHODS[0] else COL['teal'])),method+'_roi')
        c.orient('native');c.turn('z',-13);c.turn('y',18);c.turn('x',-8)
        c.viewport(2400,1600);c.zoom('all',buffer=3.2)
        global_view=list(c.get_view());renders=[]
        for name,method in [('WT',None),('Base',METHODS[0]),('Guided',METHODS[1])]:
            c.disable('all')
            if method:
                c.enable('native_trace');c.enable(method)
            else:
                c.enable('native');c.set('cartoon_transparency',0,'native')
            c.set_view(global_view);path=folder/(name+'_global.png')
            c.png(str(path),width=2400,height=1600,dpi=600,ray=1,quiet=1)
            c.scene(name+'_global','store')
            renders.append(dict(file=path.name,sha256=sha(path),view=list(c.get_view()),scope='global',method=name))
            print('SEC6_GLOBAL_RENDERED '+name,flush=True)
        # Input-defined event/anchor neighborhood, with the same union framing
        # in both views so the distant Base partners remain visible.
        roi=d['ROI_positions'];union=np.concatenate([native[roi],fitted[METHODS[0]][roi],fitted[METHODS[1]][roi]])
        union=union[np.isfinite(union).all(1)]
        c.load_cgo([COLOR,0,0,0]+[value for p in union for value in [SPHERE,*p,.01]],'roi_bounds')
        c.viewport(2400,1100);c.set_view(global_view);c.turn('z',90);c.zoom('roi_bounds',buffer=3.0)
        local_view=list(c.get_view());c.delete('roi_bounds')
        for name,method in [('Base',METHODS[0]),('Guided',METHODS[1])]:
            edges=[ALPHA,1.];col=rgb(COL['base'] if method==METHODS[0] else COL['teal'])
            for r in d['ROI_contacts']:
                i,j=r['i_1based']-1,r['j_1based']-1
                # Reference edges remain thin and gray in the WT frame.
                line(edges,native[i],native[j],.045,rgb('#B7C0C7'),True)
                status=r[method+'_status']
                if status!='deleted_endpoint':
                    recovered=status=='recovered'
                    rescued=(r[METHODS[0]+'_status']=='retained_not_recovered' and r[METHODS[1]+'_status']=='recovered')
                    line(edges,fitted[method][i],fitted[method][j],.13 if rescued else .075,col,not recovered)
            for i in [41,44,133]:
                p=fitted[method][i]
                assert np.isfinite(p).all();edges.extend([COLOR,*col,SPHERE,*p,.85])
            obj=name+'_local_contacts';c.load_cgo(edges,obj)
            c.disable('all');c.enable('native_roi_trace');c.enable('native_roi_seed')
            c.enable(method+'_roi');c.enable(obj);c.set_view(local_view)
            path=folder/(name+'_local.png');c.png(str(path),width=2400,height=1100,dpi=600,ray=1,quiet=1)
            c.scene(name+'_local','store');renders.append(dict(file=path.name,sha256=sha(path),view=list(c.get_view()),scope='local',method=name))
            print('SEC6_LOCAL_RENDERED '+name,flush=True)
        c.scene('WT_global','recall');c.save(str(folder/'Sec6_case_editable.pse'))
        write(out/'audit/molecular_rendering.json',dict(status='COMPLETE',PyMOL_version=c.get_version(),
              images=renders,global_camera_shared=True,global_scale_shared=True,local_camera_shared=True,
              local_scale_shared=True,ROI_rule=d['ROI_rule'],ROI_positions_0based=roi,
              local_sequence_context_positions_0based=context,
              representation='Deposited full-atom WT ribbon; archived predicted C-alpha traces with display-only spline',
              new_atoms_or_structure_predictions=0,sequence_and_contact_identity='Original WT indices, not free realignment'))

def text_geometry(fig):
    import matplotlib
    fig.canvas.draw();renderer=fig.canvas.get_renderer();hidden=set();texts=[];outside=[]
    for ax in fig.axes:
        if not ax.axison:
            for axis in [ax.xaxis,ax.yaxis]:
                hidden.update([id(axis.label),id(axis.offsetText)])
                for tick in axis.get_major_ticks()+axis.get_minor_ticks():hidden.update([id(tick.label1),id(tick.label2)])
    for t in fig.findobj(matplotlib.text.Text):
        if id(t) in hidden or not t.get_visible() or not t.get_text():continue
        b=matplotlib.text.Text.get_window_extent(t,renderer)
        r=dict(text=t.get_text(),font_pt=t.get_fontsize(),bbox_px=list(b.bounds));texts.append(r)
        if min(b.x0,b.y0)<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:outside.append(r)
    collisions=[]
    for i,a in enumerate(texts):
        x,y,w,h=a['bbox_px']
        for b in texts[i+1:]:
            u,v,ww,hh=b['bbox_px'];xx=min(x+w,u+ww)-max(x,u);yy=min(y+h,v+hh)-max(y,v)
            if xx>1.2 and yy>1.2:collisions.append(dict(a=a['text'],b=b['text'],intersection_px=[xx,yy]))
    return dict(min_font_pt=min(r['font_pt'] for r in texts),outside=outside,overlaps=collisions,texts=texts)

def compose(root):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from PIL import Image
    out=root/REL;d=read(out/'data/case_data.json');render=read(out/'audit/molecular_rendering.json')
    assert all(sha(out/'molecules'/r['file'])==r['sha256'] for r in render['images'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,
        'xtick.labelsize':8,'ytick.labelsize':8.5,'axes.linewidth':.8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'sec6-case-20261024',
        'text.color':COL['ink'],'axes.labelcolor':COL['ink'],'xtick.color':'#637184','ytick.color':'#637184'})
    fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=180,facecolor='white')
    def text(x,y,s,size=8.5,**kw):return fig.text(x/W,y/H,s,fontsize=size,**kw)
    def axes(x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
    def panel(x,y,letter,title):
        text(x,y,letter,size=10,fontweight='bold');text(x+5,y,title,size=9.5,fontweight='bold')

    # A. True local raw MMseqs alignment; no synthetic extra homolog rows.
    panel(5,115.7,'A','Natural deletion evidence')
    x0,cell=25.,2.05
    seed=set(d['seed_positions_0based']);anchors=set(d['anchor_positions_0based'])
    text(5,105.3,'WT / 2FJI',size=8.5)
    text(5,100.5,'O74846',size=8.5)
    for r in d['alignment_window']:
        pos=r['WT_position'];i=pos-1;xx=x0+(pos-111)*cell
        if i in seed:
            fig.add_artist(Rectangle(((xx-cell*.48)/W,99.6/H),cell*.96/W,9.2/H,
                            transform=fig.transFigure,facecolor=COL['seed'],alpha=.14,edgecolor='none'))
        for yy,key in [(105.3,'WT'),(100.5,'homolog')]:
            text(xx,yy,r[key],size=8.5,fontfamily='DejaVu Sans Mono',ha='center',
                 color=COL['seed'] if i in seed else COL['ink'],fontweight='bold' if i in anchors else 'normal')
        if pos in [111,115,122,127,134,138]:
            text(xx,110.1,str(pos),size=8,ha='center',color='#637184')
        if i in anchors:
            fig.add_artist(Line2D([xx/W],[99.0/H],transform=fig.transFigure,marker='v',ms=2.7,
                                 color=COL['ink'],ls='none',mew=0))
    for a,b in d['seed_spans']:
        xa=x0+(a+1-111)*cell-cell*.4;xb=x0+(b+1-111)*cell+cell*.4
        fig.add_artist(Line2D([xa/W,xa/W,xb/W,xb/W],[98.4/H,97.8/H,97.8/H,98.4/H],
                             transform=fig.transFigure,color=COL['seed'],lw=.8))
        text((xa+xb)/2,94.5,'6 aa',size=8,ha='center',color=COL['seed'])
    fig.add_artist(Line2D([5.8/W],[95.5/H],transform=fig.transFigure,marker='v',ms=3,color=COL['ink'],mew=0,ls='none'))
    text(8.5,94.5,'Anchors',size=8)

    # B. Actual selected masks on the original 209-residue coordinate system.
    panel(100,115.7,'B','Learned deletion planning')
    text(100,111.0,'12 seed + 30 Policy = 42 aa',size=9,color=COL['teal'])
    b=axes(119,96,50,13.0)
    for y,method in [(2,None),(1,METHODS[0]),(0,METHODS[1])]:
        b.hlines(y,1,209,color='#DCE2E6',lw=.8,zorder=0)
        intervals=d['seed_spans'] if method is None else spans(d['plans'][method])
        color=COL['seed'] if method is None else COL['base'] if method==METHODS[0] else COL['teal']
        b.broken_barh([(a+.5,c-a+1) for a,c in intervals],(y-.17,.34),facecolors=color,edgecolors='none')
        if method==METHODS[1]:b.broken_barh([(a+.5,c-a+1) for a,c in d['seed_spans']],(y-.17,.34),facecolors=COL['seed'])
        if method is None or method==METHODS[1]:
            b.plot([i+1 for i in anchors],[y+.25]*len(anchors),marker='v',ms=2.1,color=COL['ink'],mew=0,ls='none')
    b.set(xlim=(.5,209.5),ylim=(-.55,2.65),yticks=[2,1,0],yticklabels=['Seed','Base','Guided'],
          xticks=[1,50,100,150,209],xlabel='WT residue')
    b.spines['left'].set_visible(False);b.tick_params(axis='y',length=0,pad=3,labelsize=8)
    b.tick_params(axis='x',length=2,pad=2);b.xaxis.labelpad=1.8

    # C. The molecular triplet and paired local contacts dominate the page.
    panel(5,90.0,'C','Sec6 structural remodeling')
    crops={};panels=[];molecular_axes=[]
    def image_group(names,kind):
        ims=[Image.open(out/'molecules'/name).convert('RGBA') for name in names]
        boxes=[im.getchannel('A').getbbox() for im in ims]
        box=(max(0,min(v[0] for v in boxes)-30),max(0,min(v[1] for v in boxes)-30),
             min(ims[0].width,max(v[2] for v in boxes)+30),min(ims[0].height,max(v[3] for v in boxes)+30))
        crops[kind]=dict(crop=box,shared=True,source_canvas=ims[0].size)
        result=[]
        for name,im in zip(names,ims):
            cropped=im.crop(box);path=out/'molecules'/('display_'+name)
            cropped.save(path,dpi=(600,600));result.append(cropped)
        return result
    global_names=['WT_global.png','Base_global.png','Guided_global.png']
    global_images=image_group(global_names,'global')
    for xx,im,title in zip([4,60,116],global_images,['WT: 209 aa','Base: 167 aa','Homology guided: 167 aa']):
        text(xx+27.6,85.0,title,size=8.5,ha='center',fontweight='bold')
        a=axes(xx,49.3,55.2,36.0);a.imshow(im);a.axis('off')
        molecular_axes.append((a,'global'))
    # Exact orthoscopic projection of the WT contact endpoints into the shared
    # PyMOL image. Labels are vectors; no 3D coordinates or image pixels change.
    fig.canvas.draw()
    view=np.asarray(next(r['view'] for r in render['images'] if r['method']=='WT' and r['scope']=='global'))
    xyz=np.asarray(d['native_coordinates'])
    camera=(xyz-view[12:15])@view[:9].reshape(3,3)+view[9:12]
    pixels_per_A=1600/(2*abs(view[11])*np.tan(np.deg2rad(abs(view[17]))/2))
    projected=np.c_[1200+camera[:,0]*pixels_per_A,800-camera[:,1]*pixels_per_A]
    crop=crops['global']['crop'];cw,ch=crop[2]-crop[0],crop[3]-crop[1]
    uv=np.c_[(projected[:,0]-crop[0])/cw,1-(projected[:,1]-crop[1])/ch]
    wt_axes=molecular_axes[0][0];bbox=wt_axes.get_position()
    for i in [41,44,133]:
        wt_axes.plot(uv[i,0],uv[i,1],transform=wt_axes.transAxes,marker='v' if i==133 else 'o',
                     ms=2.4 if i==133 else 2.0,color=COL['ink'],mew=0,ls='none',clip_on=False)
    labels=[([41,44],30.0,76.5,'42/45'),([133],7.0,72.0,'134')]
    annotation_points=[]
    for indices,lx,ly,label in labels:
        q=uv[indices].mean(0);px=(bbox.x0+q[0]*bbox.width)*W;py=(bbox.y0+q[1]*bbox.height)*H
        fig.add_artist(Line2D([lx/W,px/W],[(ly-.4)/H,py/H],transform=fig.transFigure,
                             color='#637184',lw=.6))
        text(lx,ly,label,size=8)
        annotation_points.append(dict(label=label,WT_positions_1based=[i+1 for i in indices],
                                      projected_image_pixels=projected[indices].tolist(),figure_mm=[px,py]))
    write(out/'audit/WT_contact_site_projection.json',dict(
        projection='Orthoscopic PyMOL row-vector rotation; source 2400x1600 view and common crop',
        view=view.tolist(),crop=crop,markers=annotation_points,coordinates_changed=False))
    local_images=image_group(['Base_local.png','Guided_local.png'],'local')
    for xx,im,method in zip([60,116],local_images,METHODS):
        a=axes(xx,24.4,55.2,26.2);a.imshow(im);a.axis('off')
        molecular_axes.append((a,'local'))
        one=next(r for r in d['ROI_contacts'] if r['i_1based']==42 and r['j_1based']==134)
        two=next(r for r in d['ROI_contacts'] if r['i_1based']==45 and r['j_1based']==134)
        label='42/45-134: %.1f / %.1f Å'%(one[method+'_distance_A'],two[method+'_distance_A'])
        text(xx+27.6,22.4,label,size=8,ha='center',color=COL['base'] if method==METHODS[0] else COL['teal'])
    text(5,44.0,'WT ribbon; predictions: Cα traces',size=8)
    text(5,39.5,'WT sites of guided deletion',size=8)
    for xx,label,color in [(5,'Seed (12)',COL['seed']),(29,'Added (30)',COL['teal'])]:
        fig.add_artist(Rectangle((xx/W,35.2/H),2/W,1.8/H,transform=fig.transFigure,facecolor=color,edgecolor='none'))
        text(xx+2.8,35.1,label,size=8)
    text(5,30.5,'Local pairs: 42/45-134',size=8)
    text(5,26.5,'Colored solid: recovered',size=8)
    text(5,22.5,'Colored dashed: unrecovered',size=8)
    # Local CA correspondence is fixed by WT indices; no local independent fit.

    # D. Fixed-denominator counts. No common-structure re-alignment substituted.
    panel(5,18.5,'D','Native-contact consequences')
    text(169,18.5,'536 fixed WT contacts',size=8.5,ha='right',color='#637184')
    text(111,18.5,'+68 recovered',size=8.5,ha='center',color=COL['teal'])
    a=axes(34,7.0,135,10.0)
    status_colors=[COL['deleted'],COL['unrecovered'],COL['teal']]
    for y,method in [(1,METHODS[0]),(0,METHODS[1])]:
        left=0
        for number,color in zip(d['contact_counts'][method],status_colors):
            a.barh(y,number,left=left,height=.66,color=color,edgecolor='white',lw=.55)
            a.text(left+number/2,y,str(number),ha='center',va='center',fontsize=8.5,
                   color='white' if color==COL['teal'] else COL['ink'])
            left+=number
    a.set(xlim=(0,536),ylim=(-.6,1.6),yticks=[1,0],yticklabels=['DelePref-Base','Homology guided'],xticks=[])
    a.tick_params(axis='y',length=0,pad=4,labelsize=8.5)
    for sp in a.spines.values():sp.set_visible(False)
    for xx,label,color in [(35,'Deleted endpoints',status_colors[0]),(82,'Retained, unrecovered',status_colors[1]),(140,'Recovered',status_colors[2])]:
        fig.add_artist(Rectangle((xx/W,2.1/H),2.1/W,1.9/H,transform=fig.transFigure,facecolor=color,edgecolor='none'))
        text(xx+3.0,2.05,label,size=8)
    geo=text_geometry(fig);write(out/'audit/text_geometry.json',geo)
    for a,kind in molecular_axes:
        pos=a.get_position()
        panels.append(dict(kind=kind,x_mm=pos.x0*W,y_mm=pos.y0*H,width_mm=pos.width*W,height_mm=pos.height*H))
    area=sum(p['width_mm']*p['height_mm'] for p in panels)/(W*H)
    assert area>.35
    write(out/'audit/molecular_layout.json',dict(molecular_image_frame_fraction=area,frames=panels,
          shared_crops=crops,global_frames_share_camera_and_scale=True,local_frames_share_camera_and_scale=True,
          local_camera_rotated_90_degrees_relative_to_global=True,
          raster_layers_are_actual_PyMOL_only=True,alignment_masks_text_and_bars_are_native_vectors=True))
    files={}
    for ext in ['pdf','svg']:
        meta=dict(CreationDate=datetime(2026,10,24,tzinfo=timezone.utc),ModDate=datetime(2026,10,24,tzinfo=timezone.utc)) if ext=='pdf' else dict(Date='2026-10-24')
        p=out/('Figure2.'+ext);fig.savefig(p,dpi=600,bbox_inches=None,pad_inches=0,facecolor='white',metadata=meta);files[p.name]=sha(p)
    p=out/'Figure2_600dpi.png';fig.savefig(p,dpi=600,bbox_inches=None,pad_inches=0,facecolor='white');files[p.name]=sha(p)
    fig.savefig(out/'audit/layout_preview.png',dpi=220,bbox_inches=None,pad_inches=0,facecolor='white')
    plt.close(fig)
    write(out/'audit/render_completion.json',dict(status='RENDERED_PENDING_VISUAL_QA',width_mm=W,height_mm=H,
          min_font_pt=geo['min_font_pt'],outside=len(geo['outside']),overlaps=len(geo['overlaps']),
          molecular_image_frame_fraction=area,files=files,new_predictions=0,new_statistics=0,paper_modified=False))
    print(json.dumps(dict(status='SEC6_CASE_FIGURE_RENDERED',outside=len(geo['outside']),overlaps=geo['overlaps'],
                          min_font_pt=geo['min_font_pt'],molecular_image_frame_fraction=area)),flush=True)
