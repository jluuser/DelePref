"""Input-only controls and outcome-independent evaluation subset selection."""
import hashlib
import numpy as np
import delepref_unlimited_planning as planning

CONTRACT=dict(K=8,qmax=None,max_span_len=None,
    terminal='8 evenly spaced N/C deletion splits, including both extremes; original Scorer selects',
    msa='negative log odds of WT residue frequency with Jeffreys pseudocount 0.5',
    known_alphabet='20 standard amino acids and observed internal gap',unknown='excluded from denominator',
    representative_weighting='uniform within frozen gap-pattern-preserving representatives',
    msa_no_known_position_score=0.,msa_no_known_positions='exact unlimited DelePref fallback',
    msa_noise_scale=1.,seeded_noise='same convention as unlimited DelePref; max128 attempts',
    scorer='same original trained unlimited adapter, no added RAG score',
    no_structure_or_model_outcomes_used=True)


def sequence_sha(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def condition_seed(seed,pid,sequence,budget):
    material='{}\0{}\0{}\0{:.2f}'.format(int(seed),pid,sequence_sha(sequence),float(budget))
    return int(hashlib.sha256(material.encode()).hexdigest()[:16],16)&((1<<63)-1)


def terminal_proposals(length,budget,k=8):
    count=planning.delete_count(length,budget)
    if k<2 or count+1<k:raise ValueError('not enough unique terminal splits for requested K')
    result=[]
    for index in range(k):
        n_deleted=index*count//(k-1);c_deleted=count-n_deleted
        spans=tuple(s for s in ((0,n_deleted-1) if n_deleted else None,
            (length-c_deleted,length-1) if c_deleted else None) if s is not None)
        planning.validate_plan(length,budget,spans)
        result.append(dict(spans=spans,kind='terminal',N_deleted=n_deleted,C_deleted=c_deleted))
    if len({x['spans'] for x in result})!=k:raise RuntimeError('duplicate terminal split')
    return result


def conservation_logits(sequence,profile):
    n=len(sequence);known=np.zeros(n,dtype=np.int64);match=np.zeros(n,dtype=np.int64)
    for representative in profile.get('representatives',[]):
        row=representative['projected']
        if len(row)!=n:raise ValueError('MSA projection length mismatch')
        for i,(wt,observed) in enumerate(zip(sequence,row)):
            if observed in planning.AA or observed=='-':
                known[i]+=1;match[i]+=int(observed==wt)
    frequency=(match+.5)/(known+1.)
    logits=np.log1p(-frequency)-np.log(frequency)
    logits[known==0]=0.
    if not np.isfinite(logits).all():raise ValueError('nonfinite conservation logits')
    return logits,dict(positions_with_known_evidence=int((known>0).sum()),
        known_counts=known.tolist(),WT_match_counts=match.tolist(),all_unknown=not bool(known.any()))


def fixed_subsets(panel):
    """Nested source-balanced subsets; ranking depends on WT identity only."""
    if len({x['protein_id'] for x in panel})!=len(panel):raise ValueError('duplicate WT identity')
    by_source={source:[] for source in ('pdb_chain','cath_domain')}
    for row in panel:
        if row['stratum'] not in by_source:raise ValueError('unknown source stratum')
        key=hashlib.sha256(('recomb_v8_eval_subsets_20261016\0'+row['sequence_sha256']).encode()).hexdigest()
        by_source[row['stratum']].append((key,row['protein_id']))
    ordered={source:[pid for _,pid in sorted(rows)] for source,rows in by_source.items()}
    if min(map(len,ordered.values()))<20:raise ValueError('insufficient source strata for fixed subsets')
    return dict(secondary_10pct=[pid for source in sorted(ordered) for pid in ordered[source][:20]],
        second_predictor=[pid for source in sorted(ordered) for pid in ordered[source][:4]],
        completion_ablation=[pid for source in sorted(ordered) for pid in ordered[source][:10]],
        rule='same salted WT-sequence-hash ordering within each source; no retrieval/model/structure results')


def input_evidence(sequence,profile,budget):
    events=planning.natural_events(profile,len(sequence));seeds=planning.seed_sets(profile,len(sequence),budget)
    count=planning.delete_count(len(sequence),budget)
    return dict(representatives=len(profile.get('representatives',[])),reliable_event_homologs=len(events),
        reliable_events=sum(len(h['events']) for h in events),
        maximum_events_one_homolog=max((len(h['events']) for h in events),default=0),
        target_deleted_count=count,exact_natural_plan_available=any(x['complete_natural_plan'] for x in seeds),
        maximum_natural_deletion_support=max((x['deleted'] for x in seeds),default=0),
        support_class='complete' if any(x['complete_natural_plan'] for x in seeds) else ('partial' if seeds else 'none'))
