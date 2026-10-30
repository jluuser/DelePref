"""CPU PyMOL ray-tracing of archived C-alpha coordinates; no atom reconstruction."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pymol2
from pymol.cgo import ALPHA,COLOR,CYLINDER,SPHERE
BASE=Path(__file__).resolve().parent;OUT=BASE/'structure_renders';OUT.mkdir(exist_ok=True)
RGB=dict(native=[.72,.76,.78],seed=[.85,.55,.20],baseline=[.28,.34,.38],rag=[.075,.50,.55])
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def array(x):return np.array([p if p is not None else [np.nan]*3 for p in x],float)
def cgo_trace(x,order,color,radius,alpha=1.,allowed=None):
    out=[ALPHA,alpha,COLOR,*color]
    for a,b in zip(order[:-1],order[1:]):
        if allowed is not None and (a not in allowed or b not in allowed):continue
        if np.isfinite(x[[a,b]]).all() and np.linalg.norm(x[a]-x[b])<5.0:
            out.extend([CYLINDER,*x[a],*x[b],radius,*color,*color,SPHERE,*x[a],radius])
    return out
def edge(out,a,b,radius,color,dashed=False):
    if dashed:
        for t in np.arange(0,1,.18):
            x=a+t*(b-a);y=a+min(t+.09,1)*(b-a);out.extend([CYLINDER,*x,*y,radius,*color,*color])
    else:out.extend([CYLINDER,*a,*b,radius,*color,*color])
def run():
    instance=pymol2.PyMOL();instance.start();cmd=instance.cmd;proof=[]
    for cid in ['C07','C04']:
        p=read(BASE/'figure_data'/(cid+'_structural_edit.json'));c=p['case'];nat=array(c['native_coordinates']);center=np.array(p['PCA_center']);basis=np.array(p['PCA_basis'])
        xyz={'native':(nat-center)@basis.T}
        for m in c['variants']:xyz[m]=(array(p['views'][m]['fitted_WT_frame'])-center)@basis.T
        for zoom in [False,True]:
            allowed=set(p['ROI_positions_0based']) if zoom else set(range(len(nat)))
            allpts=np.concatenate([x[list(allowed)] for x in xyz.values()]);allpts=allpts[np.isfinite(allpts).all(1)]
            viewcenter=(allpts.min(0)+allpts.max(0))/2;span=float(max(allpts.max(0)-allpts.min(0))*.56);span=max(span,6.)
            views=[]
            for method,key in [('delepref_unlimited','baseline'),('rag_unlimited','rag')]+([] if zoom else [('combined','combined')]):
                cmd.reinitialize();cmd.bg_color('white');cmd.set('orthoscopic',1);cmd.set('ray_shadows',0);cmd.set('specular',.18);cmd.set('ambient',.52);cmd.set('direct',.45);cmd.set('shininess',25);cmd.set('antialias',2);cmd.set('ray_trace_mode',0);cmd.set('ray_opaque_background',1);cmd.set('transparency_mode',2)
                cmd.set('max_threads',2);cmd.viewport(1100,850)
                xnat=xyz['native']-viewcenter;xp=xyz[method]-viewcenter if method!='combined' else None
                cmd.load_cgo(cgo_trace(xnat,list(range(len(nat))),RGB['native'],.22 if zoom else .28,.32,allowed),'native')
                for shown,shownkey in ([('delepref_unlimited','baseline'),('rag_unlimited','rag')] if method=='combined' else [(method,key)]):
                    kept=[i for i,j in enumerate(c['variants'][shown]['metrics']['correspondence']['wt_to_edit']) if j is not None]
                    cmd.load_cgo(cgo_trace(xyz[shown]-viewcenter,kept,RGB[shownkey],.30 if method=='combined' and shownkey=='baseline' else .35 if zoom else .42,.8 if method=='combined' and shownkey=='baseline' else 1.,allowed),'prediction_'+shownkey)
                seed=set(i for a,b in c['variants']['rag_unlimited']['plan']['retrieval_seed_spans'] for i in range(a,b+1))
                cmd.load_cgo(cgo_trace(xnat,list(range(len(nat))),RGB['seed'],.26 if zoom else .32,.72,allowed&seed),'native_seed')
                if zoom:
                    edges=[ALPHA,1.]
                    for r in p['ROI_contacts']:
                        i=r['i_1based']-1;j=r['j_1based']-1
                        edge(edges,xnat[i],xnat[j],.045,RGB['native'],True)
                        status=r[method+'_status']
                        if status not in ['deleted_endpoint','unknown']:
                            edge(edges,xp[i],xp[j],.08,RGB[key],status=='retained_not_recovered')
                    for i in p['ROI_anchor_positions_0based']:
                        if np.isfinite(xnat[i]).all():edges.extend([COLOR,.16,.20,.22,SPHERE,*xnat[i],.65])
                    cmd.load_cgo(edges,'native_contact_network')
                cmd.pseudoatom('bounds',pos=[-span,-span,-span]);cmd.pseudoatom('bounds',pos=[span,span,span]);cmd.hide('everything','bounds');cmd.zoom('bounds',buffer=0);cmd.delete('bounds')
                camera=list(cmd.get_view());views.append(camera)
                filename=f'{cid}_{key}_'+('zoom' if zoom else 'global')+'.png'
                cmd.png(str(OUT/filename),width=1650 if method=='combined' else 1100 if not zoom else 850,height=900 if method=='combined' else 850 if not zoom else 650,dpi=600,ray=1,quiet=1)
                proof.append(dict(file=filename,sha256=hashlib.sha256((OUT/filename).read_bytes()).hexdigest(),camera=camera,view_center=viewcenter.tolist(),span=span,
                  source_case=cid,method=method,representation='CPU_ray_traced_true_CA_tubes_no_reconstructed_backbone',ROI=sorted(allowed) if zoom else 'full_construct'))
            assert views[0]==views[1], 'Camera must be identical for both methods'
    record=dict(status='TRUE_CA_CPU_RAY_TRACING_COMPLETE',engine='PyMOL open-source',version=cmd.get_version(),images=proof,all_method_pairs_same_camera=True,new_structures=0)
    (BASE/'pymol_render_provenance.json').write_text(json.dumps(record,indent=2)+'\n');instance.stop();print(json.dumps(dict(images=len(proof),engine='PyMOL',new_structures=0)))
if __name__=='__main__':run()
