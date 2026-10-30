"""Consensus residue correspondence across all optimal unit-edit alignments."""
from array import array


def distances(a,b):
    matrix=[array('H',range(len(b)+1))]
    for i,x in enumerate(a,1):
        row=array('H',[i]+[0]*len(b));previous=matrix[-1]
        for j,y in enumerate(b,1):
            row[j]=min(previous[j]+1,row[j-1]+1,previous[j-1]+(x!=y))
        matrix.append(row)
    return matrix


def consensus(a,b):
    if max(len(a),len(b))>=65535:raise ValueError('unsupported length')
    f=distances(a,b);reverse=distances(a[::-1],b[::-1]);n,m=len(a),len(b);cost=int(f[n][m])
    mapping=[];states=[];possibilities=[]
    for i in range(n):
        partners=[j for j in range(m) if f[i][j]+(a[i]!=b[j])+reverse[n-i-1][m-j-1]==cost]
        deleted=any(f[i][j]+1+reverse[n-i-1][m-j]==cost for j in range(m+1))
        if len(partners)==1 and not deleted:
            mapping.append(partners[0]);states.append('stable')
        elif not partners and deleted:
            mapping.append(None);states.append('deleted')
        else:
            mapping.append(None);states.append('ambiguous')
        possibilities.append(dict(edited_positions=partners,deletion_possible=deleted))
    stable=[j for j in mapping if j is not None]
    if stable!=sorted(set(stable)):raise ValueError('consensus correspondence not injective')
    return dict(method='all_optimal_unit_edit_consensus',edit_distance=cost,wt_to_edit=mapping,
                position_states=states,possibilities=possibilities,
                stable_positions=states.count('stable'),certain_deleted_positions=states.count('deleted'),
                ambiguous_positions=states.count('ambiguous'))
