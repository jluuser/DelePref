"""Final 174-mm figures. All scientific arrays are frozen inputs; no inference."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from PIL import Image
BASE=Path(__file__).resolve().parent;DATA=BASE/'figure_data';OUT=BASE/'figures';OUT.mkdir(exist_ok=True)
W=174/25.4
C=dict(native='#BFC7CB',seed='#D99035',policy='#137F8C',baseline='#4C5B66',pale='#EDF1F2',ink='#20333D',deleted='#D5DBDE')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.5,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
manifest=[]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def ascii_text(s):return str(s).translate(str.maketrans({'\u2212':'-','\u2011':'-','\u2013':'-','\u2014':'-'}))
def save(fig,fid,title,caption,sources,role='main'):
    for t in fig.findobj(matplotlib.text.Text):
        t.set_text(ascii_text(t.get_text()));t.set_fontsize(max(8,t.get_fontsize()))
    fig.canvas.draw();ren=fig.canvas.get_renderer();issues=[];texts=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            b=t.get_window_extent(ren);texts.append(dict(text=t.get_text(),font_pt=t.get_fontsize(),bbox=list(b.bounds)))
            if b.x0<-.6 or b.y0<-.6 or b.x1>fig.bbox.x1+.6 or b.y1>fig.bbox.y1+.6:issues.append(texts[-1])
    write(DATA/(fid+'_text_preflight.json'),dict(width_mm=174,height_mm=fig.get_size_inches()[1]*25.4,min_font_pt=8,text_outside_canvas=issues,text=texts))
    for fmt in ['pdf','svg','png']:fig.savefig(OUT/(fid+'.'+fmt),dpi=450,facecolor='white')
    plt.close(fig);manifest.append(dict(id=fid,title=title,role=role,width_mm=174,english_caption=caption,sources=sources,script='render_final_figures.py',files={f:'figures/'+fid+'.'+f for f in ['pdf','svg','png']}))
def label(ax,s):ax.set_title(s,loc='left',fontweight='bold',pad=8)
def spans(ax,sp,y,col,h=.22):
    for a,b in sp:ax.broken_barh([(a+.5,b-a+1)],(y-h/2,h),facecolors=col,edgecolors='none')
def alignment(packet,flank=4):
    c=packet['case'];e=next(e for e in c['evidence'] if e['homolog_id']==c['primary_evidence_target']);f=e['raw_hit']['fields'];a,b=packet['ROI_event_0based'];lo=a-flank;hi=b+flank;q=int(f[4])-1;t=int(f[7])-1;cols=[]
    for qa,ta in zip(f[14],f[15]):
        if qa!='-' and lo<=q<=hi:cols.append(dict(q=q,t=t if ta!='-' else None,qa=qa,ta=ta,seed=a<=q<=b))
        q+=qa!='-';t+=ta!='-'
    return cols,(a,b),e
def crop_bounds(im):
    y,x=np.where(np.any(np.asarray(im.convert('RGB'))<245,axis=2));return(max(0,x.min()-20),max(0,y.min()-20),min(im.width,x.max()+21),min(im.height,y.max()+21))
def fig1():
    p=read(DATA/'C07_structural_edit.json');c=p['case'];plan=c['variants']['rag_unlimited']['plan'];seed=plan['retrieval_seed_spans'];n=len(c['panel']['sequence'])
    fig=plt.figure(figsize=(W,5.0));fig.text(.055,.972,'Natural evidence to an exact-budget deletion plan',fontsize=11,fontweight='bold',va='top')
    ax=fig.add_axes([.045,.50,.91,.40]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    boxes={}
    def box(key,x,y,w,h,txt,col=C['pale']):
        ax.add_patch(patches.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.006,rounding_size=.015',fc=col,ec='#8C9CA3',lw=.6,clip_on=False));ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=8);boxes[key]=(x,y,w,h)
    def arrow(a,b,side='h',ls='-',color=C['baseline']):
        x,y,w,h=boxes[a];u,v,ww,hh=boxes[b]
        if side=='h':s=(x+w,y+h/2);e=(u,v+hh/2)
        elif side=='down':s=(x+w/2,y);e=(u+ww/2,v+hh)
        else:s=(x+w/2,y+h);e=(u+ww/2,v)
        ax.annotate('',e,s,arrowprops=dict(arrowstyle='-|>',lw=.8,color=color,linestyle=ls,connectionstyle='arc3,rad=0'))
    box('wt',.00,.73,.17,.17,'WT sequence\n+ deletion budget')
    box('plm',.235,.73,.17,.17,'Frozen PLM\nWT representation')
    box('pol',.475,.73,.17,.17,'Frozen Policy\nposition logits')
    box('orig',.735,.73,.24,.17,'Original Policy\nK8 candidates')
    box('ret',.00,.39,.20,.17,'Homolog retrieval\narchived alignments')
    box('seed',.255,.39,.21,.17,'Seed / anchor\nconstruction')
    box('comp',.535,.39,.22,.17,'Exact-budget\nPolicy completion')
    box('pool',.735,.04,.24,.17,'Candidate pool K8\nnominal 4 + 4')
    box('score',.41,.04,.23,.17,'Frozen Scorer\nrank complete plans')
    box('out',.10,.04,.21,.17,'Selected deletion\nsequence')
    arrow('wt','plm');arrow('plm','pol');arrow('pol','orig');arrow('wt','ret','down');arrow('ret','seed');arrow('seed','comp');arrow('pol','comp','down')
    arrow('comp','pool','down');arrow('orig','pool','down')
    ax.annotate('',(.64,.125),(.735,.125),arrowprops=dict(arrowstyle='-|>',lw=.8,color=C['baseline']))
    ax.annotate('',(.31,.125),(.41,.125),arrowprops=dict(arrowstyle='-|>',lw=.8,color=C['baseline']))
    ax.annotate('Fallback: no legal proposal -> original K8',(.852,.70),(.025,.66),fontsize=8,color=C['baseline'],arrowprops=dict(arrowstyle='->',ls='--',lw=.7,connectionstyle='arc3,rad=-.05'))
    ax.text(.015,-.08,'Frozen PLM representation is reused by Policy and Scorer; deduplication and backfill are unchanged.',fontsize=8)
    ax=fig.add_axes([.18,.21,.60,.19]);label(ax,'Real Sec6 domain: 2FJI / chain 1')
    for y in [0,1,2]:ax.hlines(y,1,n,color=C['native'],lw=.8)
    spans(ax,seed,2,C['seed']);spans(ax,json.loads(c['variants']['delepref_unlimited']['plan']['deleted_intervals']),1,C['baseline']);spans(ax,json.loads(plan['deleted_intervals']),0,C['policy']);spans(ax,seed,0,C['seed'])
    for a,b in seed:ax.plot([a,b+2],[2,2],'|',color=C['ink'],ms=8)
    ax.set(xlim=(.5,n+.5),ylim=(-.5,2.6),yticks=[0,1,2],yticklabels=['RAG','DelePref','Seed + anchors'],xticks=[1,50,100,150,n],xlabel='WT residue (1-based)')
    pool=sorted(c['variants']['rag_unlimited']['candidate_pool'],key=lambda r:r['raw_candidate_rank']);scores=np.array([r['scorer_score'] for r in pool]);win=next(i for i,r in enumerate(pool) if r['deletion_mask_sha256']==plan['deletion_mask_sha256'])
    ax=fig.add_axes([.845,.21,.12,.19]);ax.barh(range(8),scores-scores.min(),color=[C['policy'] if r['generation_kind'].startswith('retrieval') else C['native'] for r in pool],height=.65);ax.scatter(scores[win]-scores.min()+.012,win,c=C['ink'],marker='*',s=35);ax.set(yticks=[],xticks=[],ylim=(7.8,-.8));ax.spines[:].set_visible(False);ax.set_title('K8 scores',fontsize=8,pad=7)
    fig.text(.055,.072,'12 seed residues + 30 Policy additions = 42 deletions / 209 WT residues',fontsize=9,fontweight='bold',color=C['policy'])
    fig.text(.055,.027,'Actual pool: 6 Policy + 2 homology-origin masks. Star = selected. Same frozen models.',fontsize=8)
    save(fig,'Figure1_method','Evidence-guided exact-budget design',
      'Evidence-guided exact-budget design. The frozen PLM supplies WT representations to the unchanged Policy and Scorer. Archived homolog alignments yield deletion seeds and retained anchors; Policy completes each legal proposal to the exact budget before candidate-pool deduplication, backfill and Scorer selection. Without a legal proposal, the original K8 pool is returned exactly. The actual 209-residue Sec6-domain example (2FJI, chain 1) retains a 12-residue seed and adds 30 deletions, producing the fixed 42-residue budget. Its realized mixed pool contains six original Policy and two homology-origin masks under the unchanged nominal 4+4 rule. The compact bars show actual Scorer scores shifted by their minimum for display; the star identifies the selected mask. No model is trained or updated in this workflow.',
      ['figure_data/C07_structural_edit.json','analysis_contract.json'])
def fig2():
    fig=plt.figure(figsize=(W,8.0));fig.text(.055,.975,'Evolutionary evidence and structural consequences',fontsize=11,fontweight='bold',va='top')
    for ci,cid in enumerate(['C07','C04']):
        p=read(DATA/(cid+'_structural_edit.json'));c=p['case'];n=len(c['panel']['sequence']);x=.06+ci*.48;cw=.415;cols,(a,b),e=alignment(p)
        fig.text(x,.929,cid+' | '+c['panel']['PDB']+' | '+('Sec6 domain' if cid=='C07' else 'Cmr7-like domain'),fontsize=9.2,fontweight='bold')
        fig.text(x,.900,'Common correspondence (RAG - DelePref):',fontsize=8)
        fig.text(x,.879,f"local {c['selection']['common_local_delta']:+.3f}; contacts {c['selection']['common_contact_delta']:+.3f}",fontsize=8.5,color=C['policy'] if ci==0 else C['baseline'])
        ax=fig.add_axes([x,.760,cw,.09]);ax.axis('off');ts=[z['t']+1 for z in cols if z['t'] is not None]
        ax.text(0,.94,f"A  WT {cols[0]['q']+1}-{cols[-1]['q']+1} / {e['homolog_id'].replace('UniRef50_','')}",fontsize=8,transform=ax.transAxes)
        for key,y in [('qa',.55),('ta',.23)]:
            ax.text(0,y,'WT' if key=='qa' else 'H',fontsize=8,fontfamily='DejaVu Sans Mono',transform=ax.transAxes)
            for k,z in enumerate(cols):ax.text(.13+k*.038,y,z[key],fontsize=8.3,fontfamily='DejaVu Sans Mono',color=C['seed'] if z['seed'] else C['ink'],transform=ax.transAxes)
        ax.text(0,-.08,f'H positions {min(ts)}-{max(ts)}; seed {a+1}-{b+1}; anchors {a}, {b+2}',fontsize=8,transform=ax.transAxes)
        ax=fig.add_axes([x+.09,.669,cw-.095,.060]);
        for y in [0,1,2]:ax.hlines(y,1,n,color=C['native'],lw=.6)
        seed=c['variants']['rag_unlimited']['plan']['retrieval_seed_spans'];spans(ax,seed,2,C['seed'],.32)
        for yy,m,col in [(1,'delepref_unlimited',C['baseline']),(0,'rag_unlimited',C['policy'])]:spans(ax,json.loads(c['variants'][m]['plan']['deleted_intervals']),yy,col,.32)
        spans(ax,seed,0,C['seed'],.32);ax.set(xlim=(.5,n+.5),ylim=(-.5,2.5),yticks=[0,1,2],yticklabels=['RAG','DP','Seed'],xticks=[1,n]);ax.tick_params(length=2,pad=2)
        im=Image.open(BASE/'structure_renders'/f'{cid}_combined_global.png');ax=fig.add_axes([x,.448,cw,.180]);ax.imshow(im.crop(crop_bounds(im)));ax.axis('off');ax.text(0,1.02,'B  Native / DelePref / RAG overlay',transform=ax.transAxes,fontweight='bold',fontsize=8)
        ims=[Image.open(BASE/'structure_renders'/f'{cid}_{key}_zoom.png') for key in ['baseline','rag']];bs=[crop_bounds(im) for im in ims];bb=(min(bx[0] for bx in bs),min(bx[1] for bx in bs),max(bx[2] for bx in bs),max(bx[3] for bx in bs))
        for j,(im,txt) in enumerate(zip(ims,['DelePref','RAG'])):
            ax=fig.add_axes([x+j*.215,.300,.202,.140]);ax.imshow(im.crop(bb));ax.axis('off');ax.set_title(txt+' local network',fontsize=8,pad=3)
        fig.text(x,.282,f'ROI: WT {a+1}-{b+1} + anchors + native neighbors',fontsize=8)
        fig.text(x,.254,('WT domain 209 / full homolog 752' if cid=='C07' else 'WT 181 / homolog 183'),fontsize=8)
        fig.text(x,.235,('qcov .947 / tcov .247' if cid=='C07' else 'qcov .978 / tcov .940'),fontsize=8)
        ax=fig.add_axes([x+.047,.117,cw-.052,.080]);statuses=['deleted_endpoint','retained_not_recovered','recovered','unknown'];colors=[C['deleted'],C['baseline'],C['policy'],'white'];den=len(p['contacts'])
        for y,m in [(1,'delepref_unlimited'),(0,'rag_unlimited')]:
            counts=p['views'][m]['contact_counts'];left=0
            for status,col in zip(statuses,colors):
                num=counts.get(status,0);pct=num/den*100
                if num:
                    ax.barh(y,pct,left=left,height=.56,color=col,edgecolor='white',lw=.4)
                    ax.text(left+pct/2,y,str(num),ha='center',va='center',fontsize=8.5,color='white' if status in ['recovered','retained_not_recovered'] else C['ink'])
                left+=pct
        ax.set(xlim=(0,100),ylim=(-.6,1.6),yticks=[0,1],yticklabels=['RAG','DP'],xticks=[0,50,100]);ax.set_title(f'C  Fixed WT identity | N = {den}',loc='left',fontsize=8.5,fontweight='bold',pad=8)
    handles=[patches.Patch(color=C[k],label=txt) for k,txt in [('deleted','Deleted endpoint'),('baseline','Retained / not recovered'),('policy','Recovered')]]
    fig.text(.5,.076,'% of full-WT native contacts (same denominator for both methods)',ha='center',fontsize=8)
    fig.legend(handles=handles,ncol=3,bbox_to_anchor=(.055,.038),loc='lower left',frameon=False,fontsize=8,columnspacing=1,handlelength=1)
    fig.text(.055,.017,'Native C-alpha contacts: <8 A, sequence separation >=3. Bar labels = contact counts; unknown = 0.',fontsize=8)
    save(fig,'Figure2_structural_editing','Evidence to structural outcomes',
      'Evolutionary evidence and structural consequences in the preselected positive C07 (2FJI, chain 1) and negative C04 (2XVO, chain B) examples. (A) Actual MMseqs alignment columns retain WT and homolog coordinates, deleted seeds and immediate anchors. Tracks show the frozen deletion plans in 1-based supplied-WT coordinates. (B) Enlarged native/DelePref/RAG overlays and paired local native-contact networks use the existing true C-alpha coordinates, identical case-specific cameras and edit-identity fits; native reference is grey, DelePref dark grey, RAG teal and homolog seed orange. ROIs are selected from the seed with most native external contacts and its native neighbors, without using variant quality. (C) Contact-fate stacks are the primary quantitative panel. Every full-WT native C-alpha contact (<8 angstroms, sequence separation >=3) contributes once to the fixed denominator: C07 N=536, C04 N=530. Counts classify deleted endpoints, retained-but-not-recovered pairs, recovered pairs, and unknowns (zero for both cases). Horizontal units are percent of that fixed denominator, not percent of retained pairs. Header differences use the distinct original common-structure correspondence and must not be read as the differences of these bars. C07 is a domain-to-full-length homolog comparison, not a full-protein natural deletion. Neither case measures experimental function.',
      ['figure_data/C07_structural_edit.json','figure_data/C04_structural_edit.json','figure_data/C07_native_contact_fingerprint.csv','figure_data/C04_native_contact_fingerprint.csv','pymol_render_provenance.json'])
GROUPS=['no_reviewed_fallback','reviewed_eventless_fallback','proposal_Policy_wins','retrieval_completed','retrieval_refined','retrieval_direct']
ROUTE={'no_reviewed_fallback':'No homolog','reviewed_eventless_fallback':'No legal event','proposal_Policy_wins':'Policy retained','retrieval_completed':'Completed','retrieval_refined':'Refined','retrieval_direct':'Direct'}
def atlas_rows():return sorted(read(BASE/'General100_atlas.json'),key=lambda r:(GROUPS.index(r['path_group']),r['WT_sha256']))
def atlas_panel(fig,rr,rect,indices=True):
    ax=fig.add_axes(rect);n=len(rr);ax.set(xlim=(0,1),ylim=(n-.5,-2.1));ax.axis('off')
    cmap=LinearSegmentedColormap.from_list('effect',[C['baseline'],'white',C['policy']]);norm=TwoSlopeNorm(vmin=-.5,vcenter=0,vmax=.5)
    xs=[.00,.23,.37,.60,.77,.94];heads=['WT identity','Homologs','Final route','Seed %','D local','D contact']
    for x,h in zip(xs,heads):ax.text(x,-1.25,h,ha='left' if x==0 else 'center',va='center',fontweight='bold',fontsize=8)
    for i,r in enumerate(rr):
        if i%2==0:ax.axhspan(i-.5,i+.5,color='#F6F8F8',zorder=0)
        pid=r['protein_id'].split('|')[-1];ax.text(0,i,pid,va='center',fontsize=8)
        ax.text(xs[1],i,str(r['reviewed_homologs']),ha='center',va='center',fontsize=8)
        ax.text(xs[2],i,ROUTE[r['path_group']],ha='center',va='center',fontsize=8,color=C['policy'] if r['mask_changed'] else C['baseline'])
        ax.text(xs[3],i,f"{r['adopted_seed_fraction']*100:.0f}",ha='center',va='center',fontsize=8)
        for xx,key in [(xs[4],'common_local_delta'),(xs[5],'common_contact_delta')]:
            val=r[key];ax.add_patch(patches.Rectangle((xx-.06,i-.42),.12,.84,fc=cmap(norm(val)),ec='none',zorder=1));ax.text(xx,i,f'{val:+.3f}' if abs(val)>1e-12 else '0',ha='center',va='center',fontsize=8,zorder=2,color='white' if abs(val)>.35 else C['ink'])
    return ax
def fig3():
    rr=atlas_rows();changed=[r for r in rr if r['mask_changed']];assert len(changed)==23 and sum(r['sequence_changed'] for r in rr)==23
    write(DATA/'Figure3_main23.json',changed);write(DATA/'FigureS3_all100.json',rr)
    for name,rows in [('Figure3_main23',changed),('FigureS3_all100',rr)]:
        with (DATA/(name+'.csv')).open('w',newline='',encoding='utf8') as f:
            fields=['protein_id','WT_sha256','path_group','reviewed_homologs','adopted_seed_fraction','mask_changed','sequence_changed','common_local_delta','common_contact_delta'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    fig=plt.figure(figsize=(W,6.8));fig.text(.055,.975,'Evidence to decision: all 100 WT, 23 changed designs',fontsize=11,fontweight='bold',va='top')
    for x,num,txt in [(.13,'100','frozen WT'),(.37,'91','reviewed homologs'),(.62,'67','legal proposals'),(.87,'23','changed designs')]:fig.text(x,.909,num,ha='center',fontsize=18,fontweight='bold',color=C['policy']);fig.text(x,.884,txt,ha='center',fontsize=8)
    fig.text(.055,.842,'77 unchanged: 33 exact fallbacks (9 no homolog + 24 no legal event)',fontsize=8.5)
    fig.text(.055,.817,'                         + 44 original Policy selected despite legal proposals.',fontsize=8.5)
    atlas_panel(fig,changed,[.065,.213,.875,.560])
    fig.text(.055,.177,'All 23 changed masks / sequences shown; sorted by route then WT SHA.',fontsize=8)
    fig.text(.055,.151,'Seed % = adopted seed / deletion budget. D = RAG - DelePref; unitless, common correspondence.',fontsize=8)
    fig.text(.055,.120,'Source context: 476 relations across all 67 proposal WT (117 relations in actual K8).',fontsize=8,fontweight='bold')
    ss=read(BASE/'source_audit_summary.json');cats=list(ss['source_categories']);pal=['#D9A45E','#166A75','#CDD4D7','#75838C','#F6F8F8'];ax=fig.add_axes([.065,.073,.86,.018]);left=0
    for key,col in zip(cats,pal):
        v=ss['source_categories'][key]['all_relationships'];ax.barh(0,v,left=left,height=.8,color=col,edgecolor='#60717A',lw=.3);left+=v
    ax.set(xlim=(0,476),ylim=(-.5,.5));ax.axis('off')
    fig.text(.055,.045,'Fragment 13 | Isoform 4 | Domain/length 367 | Anchor/gap 65 | No extra flag 27',fontsize=8)
    fig.text(.055,.020,'Contexts are not correctness ratings. Complete 100-row identities and results: Figure S3.',fontsize=8)
    save(fig,'Figure3_decision_atlas','Evidence and the 23 changed designs',
      'Evidence and decisions for all 100 frozen General WT. The header reports all-100 coverage; the 77 unchanged designs are summarized as 33 exact fallbacks (9 without reviewed homologs, 24 without a legal event) and 44 selections of the original Policy candidate despite legal proposals. The matrix expands every one of the 23 changed masks and sequences (21 completed, one refined, one direct), retaining both gains and degradations. Rows are ordered by route and WT sequence SHA, not by quality. Seed percentage is the adopted seed divided by its fixed deletion budget. Signed local-geometry and native-contact changes are the original unitless common-structure endpoints, RAG minus DelePref. The all-100 statistical population is not replaced by these 23 rows. The source-context strip uses all 476 relations in the 67 proposal-bearing WT, with mutually exclusive priority categories; context does not rate biological correctness. Figure S3 provides every WT row, including all 77 zero changes.',
      ['General100_atlas.json','source_audit_summary.json','figure_data/Figure3_main23.csv','figure_data/FigureS3_all100.csv'])
    for block in range(2):
        fig=plt.figure(figsize=(W,9.0));fig.text(.055,.975,f'Figure S3 | Complete General100 atlas | {block*50+1}-{block*50+50}',fontsize=11,fontweight='bold',va='top')
        atlas_panel(fig,rr[block*50:(block+1)*50],[.065,.082,.875,.835]);fig.text(.055,.038,'Original route / WT SHA order. D = RAG - DelePref; common correspondence, unitless.',fontsize=8);fig.text(.055,.016,'Seed % = adopted seed / budget; exact fallback and Policy-retained rows remain visible.',fontsize=8)
        save(fig,f'FigureS3_all100_page{block+1}','Complete 100-WT decision atlas',
          'Complete all-100 atlas, continued across two 174-mm pages for readable WT identities. All unchanged and degraded rows are preserved in the same route-then-WT-SHA order as the main figure. Columns and correspondence follow Figure 3; the CSV contains full protein IDs and sequence SHA values.',
          ['General100_atlas.json','figure_data/FigureS3_all100.csv'],'supplement')
def fig4():
    d=read(DATA/'E9_same_seed_data.json');e=d['example'];n=len(e['sequence']);trial=e['paired_attempt'];seed=e['seed']['spans'];order=[r['protein_id'] for r in d['roster']];bmap={r['protein_id']:r for r in d['behavior']};qual={(r['protein_id'],r['method']):r for r in d['quality']}
    fig=plt.figure(figsize=(W,6.9));fig.text(.055,.975,'Same seed, different completion',fontsize=11,fontweight='bold',va='top')
    ax=fig.add_axes([.19,.790,.75,.120]);label(ax,'A  Actual first admitted paired proposal | 2oebA00')
    for y in [0,1,2]:ax.hlines(y,1,n,color=C['native'],lw=.7)
    spans(ax,seed,2,C['seed'],.28)
    for y,key,col in [(1,'policy_completion',C['policy']),(0,'simple_legal_completion',C['baseline'])]:spans(ax,trial['arms'][key]['spans'],y,col,.28);spans(ax,seed,y,C['seed'],.28)
    for a,b in seed:ax.plot([a,b+2],[2,2],'|',c=C['ink'],ms=8)
    ax.set(xlim=(.5,n+.5),ylim=(-.5,2.7),yticks=[0,1,2],yticklabels=['Uniform','Policy','Shared seed'],xticks=[1,50,100,n]);ax.set_xlabel('WT residue (1-based)',labelpad=3)
    fig.text(.055,.711,'Input-only example: largest seed fraction among non-fragment sources; WT SHA tie-break.',fontsize=8)
    fig.text(.055,.688,'Proposal above: Policy = selected winner; Uniform = not selected. No outcome-based choice.',fontsize=8)
    fig.text(.055,.665,'Both arms: 22 shared seed residues + 8 completion residues = 30 deletions.',fontsize=8)
    axes=[fig.add_axes([.09,.190,.25,.355]),fig.add_axes([.43,.190,.215,.355]),fig.add_axes([.75,.190,.215,.355])];yy=np.arange(20);bm=[bmap[p] for p in order];ax=axes[0]
    ax.barh(yy,[r['seed_deleted'] for r in bm],color=C['seed'],height=.65);ax.barh(yy,[r['completion_deleted'] for r in bm],left=[r['seed_deleted'] for r in bm],color=C['policy'],height=.65);ax.set(ylim=(22,-1),yticks=yy,yticklabels=[f'{i+1:02d}' for i in yy],xlabel='Deleted residues');label(ax,'B  All 20 WT budgets')
    for k,(ep,title) in enumerate([('structure_mapped_CA_lddt','C  Local geometry'),('structure_mapped_native_contact_recovery','D  Native contacts')]):
        ax=axes[k+1];delta=np.array([qual[p,'policy_completion'][ep+'_lower']-qual[p,'simple_legal_completion'][ep+'_lower'] for p in order]);eff=next(r for r in d['effects'] if r['endpoint']==ep);ci=json.loads(eff['joint_CI97_5']);mean=eff['mean_difference_lower'];assert abs(delta.mean()-mean)<1e-12
        ax.axvline(0,c=C['native'],lw=.7);ax.hlines(yy,np.minimum(delta,0),np.maximum(delta,0),color=C['native'],lw=.8);ax.scatter(delta,yy,s=18,c=[C['policy'] if x>1e-12 else C['baseline'] if x< -1e-12 else C['native'] for x in delta],zorder=3)
        ax.axhspan(20.4,22,color=C['pale']);ax.errorbar(mean,21.2,xerr=[[mean-ci[0]],[ci[1]-mean]],fmt='D',color=C['policy'],ms=5,capsize=2,lw=1.1);ax.set(ylim=(22,-1),yticks=[],xlabel='Policy - Uniform');label(ax,title)
        ax.text(0,1.25,f'{mean:+.4f}',fontsize=12,fontweight='bold',color=C['policy'],transform=ax.transAxes)
        ax.text(0,1.15,f'97.5% CI\n[{ci[0]:+.4f}, {ci[1]:+.4f}]',fontsize=8,transform=ax.transAxes)
        ax.set_xticks([0,.1,.2] if k==0 else [0,.2,.4])
        write(DATA/('Figure4_'+ep+'_pairs.json'),[dict(row=i+1,protein_id=p,delta=float(v),policy=float(qual[p,'policy_completion'][ep+'_lower']),uniform=float(qual[p,'simple_legal_completion'][ep+'_lower'])) for i,(p,v) in enumerate(zip(order,delta))])
    fig.text(.055,.080,'Rows 01-20: paired final winners; full WT IDs in source data. Grey row: mean and CI.',fontsize=8)
    fig.text(.055,.056,'Local: 4 improve / 1 degrade / 15 tie. Contacts: 5 improve / 0 degrade / 15 tie.',fontsize=8)
    fig.text(.055,.032,'N = 20 WT / components; original bootstrap 10,000, seed 20261016. Unitless common endpoints.',fontsize=8)
    fig.text(.055,.010,'Orange = shared homolog seed; teal = Policy; dark grey = Uniform. Both share frozen Scorer / K8.',fontsize=8)
    save(fig,'Figure4_same_seed','Matched-seed completion',
      'Same seed, different completion in the fixed E9 set. (A) An input-only rule chooses the largest seed fraction among cached sources without fragment/partial/truncated name flags, breaking ties by WT sequence SHA; the first admitted paired proposal is drawn. For 2oebA00, the 22-residue seed and eight-residue completion meet the same 30-residue budget. The displayed Policy proposal is its selected final winner, whereas the displayed Uniform proposal is not; no proposal was chosen using structural outcomes. (B) Seed and remaining-budget decomposition retains every one of the 20 WT in the predefined roster order. (C,D) All 20 final-winner paired differences, including ties and degradation, are shown with the archived component-bootstrap mean and joint 97.5% interval in a separate shaded row. Local geometry: +0.023416 [ +0.001105, +0.052612 ]; native-contact recovery: +0.033043 [ +0.001862, +0.096187 ]. Both are unitless common-structure endpoints, Policy minus Uniform, N=20 components, 10,000 bootstrap replicates with seed 20261016. Seeds, anchors, budgets, original candidates, candidate quota, attempt rules and frozen Scorer are shared; completion-only deletion shape is not matched in this experiment.',
      ['figure_data/E9_same_seed_data.json','Figure4_example_rule_v2.json','figure_data/Figure4_structure_mapped_CA_lddt_pairs.json','figure_data/Figure4_structure_mapped_native_contact_recovery_pairs.json'])
if __name__=='__main__':
    par=argparse.ArgumentParser();par.add_argument('--only',default='1,2,3,4');a=par.parse_args()
    for k,fn in [('1',fig1),('2',fig2),('3',fig3),('4',fig4)]:
        if k in a.only.split(','):fn()
    old=read(BASE/'figure_manifest.json') if (BASE/'figure_manifest.json').exists() else [];m={x['id']:x for x in old};m.update({x['id']:x for x in manifest});result=sorted(m.values(),key=lambda x:x['id']);write(BASE/'figure_manifest.json',result);(BASE/'captions_en.md').write_text('\n\n'.join('## '+x['id']+'\n\n'+x['english_caption'] for x in result)+'\n',encoding='utf8');print(json.dumps(dict(figures=[x['id'] for x in manifest],width_mm=174,min_font_pt=8)))
