"""Outcome-blind fixed measured-deletion mapping adapter."""
import hashlib


def prepare_mappings(rows):
    wildtypes={};edits={};links=[];seen_rows=set()
    for row in rows:
        rid=row['row_id'];target=row['target'];wt=row['wt_sequence'];sequence=row['sequence']
        if rid in seen_rows:raise ValueError('duplicate physical observation ID')
        seen_rows.add(rid)
        if target in wildtypes and wildtypes[target]!=wt:raise ValueError('multiple WT backgrounds for one target')
        wildtypes[target]=wt
        positions=row['possible_deleted_positions']
        if not positions or len(set(positions))!=len(positions):raise ValueError('empty or duplicate deletion mappings')
        if len(sequence)!=len(wt)-1:raise ValueError('not an exact single-residue deletion')
        for pos in positions:
            if not isinstance(pos,int) or not 0<=pos<len(wt) or wt[:pos]+wt[pos+1:]!=sequence:
                raise ValueError('deletion mapping does not reconstruct measured sequence')
            key=hashlib.sha256((target+'\0'+wt+'\0'+str(pos)).encode()).hexdigest()
            edits.setdefault(key,dict(edit_key=key,target=target,wt_sequence=wt,deleted_position=pos,budget=1/len(wt)))
            links.append(dict(row_id=rid,deleted_position=pos,edit_key=key))
    return wildtypes,list(edits.values()),links


def mapping_predictions(links,scores):
    required={x['edit_key'] for x in links}
    if set(scores)!=required:raise ValueError('unique edit score coverage mismatch')
    return [dict(row_id=x['row_id'],deleted_position=x['deleted_position'],score=float(scores[x['edit_key']])) for x in links]
