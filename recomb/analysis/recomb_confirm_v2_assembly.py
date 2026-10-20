"""Deterministic native protein-instance expansion; coordinates are references only."""
from pathlib import Path
import numpy as np


def apply_transform(points,matrix,vector):
    m=np.asarray(matrix,dtype=float);v=np.asarray(vector,dtype=float)
    if m.shape!=(3,3) or v.shape!=(3,) or not np.isfinite(m).all() or not np.isfinite(v).all():
        raise ValueError('Invalid biological assembly transform')
    out=[]
    for p in points:
        if p is None:out.append(None)
        else:
            a=np.asarray(p,dtype=float)
            if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('Invalid native coordinate')
            out.append((m@a+v).tolist())
    return out


def expand_instances(native_path,assembly_id,parts):
    import gemmi
    structure=gemmi.read_structure(str(native_path))
    assemblies=[a for a in structure.assemblies if a.name==str(assembly_id)]
    if len(assemblies)!=1:raise ValueError('Native assembly ID not unique')
    assembly=assemblies[0];bylabel={p['native_label_chain']:p for p in parts}
    if len(bylabel)!=len(parts):raise ValueError('Input chain labels not unique')
    protein_labels=set()
    for c in structure[0]:
        protein_labels.update(r.subchain for r in c.get_polymer())
    generated=[];seen=set()
    for gindex,g in enumerate(assembly.generators):
        labels=set(g.subchains)
        if not labels:
            labels={p['native_label_chain'] for p in parts if p['chain_id'] in g.chains}
        unknown=(labels&protein_labels)-set(bylabel)
        if unknown:raise ValueError('Native assembly requires additional unprovided protein labels: '+str(sorted(unknown)))
        for op in g.operators:
            matrix=op.transform.mat.tolist();vector=op.transform.vec.tolist()
            identity=np.allclose(matrix,np.eye(3),atol=1e-8,rtol=0) and np.allclose(vector,0,atol=1e-8,rtol=0)
            signature=tuple(np.round(np.asarray(matrix).ravel(),10))+tuple(np.round(vector,10))
            for label in sorted(labels&set(bylabel)):
                key=label,signature
                if key in seen:continue
                seen.add(key);p=bylabel[label]
                generated.append(dict(p,source_native_chain=p['chain_id'],operation_name=op.name,
                    generator_index=gindex,assembly_matrix=matrix,assembly_vector=vector,
                    operation_is_identity=bool(identity),target_gene_copy=p['role'] in ('design_antibody_H','design_enzyme'),
                    native_CA=apply_transform(p['native_CA'],matrix,vector),
                    heavy_atoms=[dict(a,xyz=apply_transform([a['xyz']],matrix,vector)[0]) for a in p.get('heavy_atoms',[])]))
    if set(p['native_label_chain'] for p in generated)!=set(bylabel):raise ValueError('Assembly omits a documented input role')
    priority={'design_antibody_H':0,'design_enzyme':0,'fixed_light_chain':1,'fixed_antigen':2,'fixed_protein_partner':3}
    generated.sort(key=lambda p:(priority.get(p['role'],9),p['source_native_chain'],not p['operation_is_identity'],
        p['operation_name'],tuple(np.asarray(p['assembly_matrix']).ravel()),tuple(p['assembly_vector'])))
    if len(generated)>26:raise ValueError('Protein chain IDs beyond current tested predictor adapter; no silent truncation')
    for i,p in enumerate(generated):p['predictor_chain_id']='ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]
    targets=[p for p in generated if p['target_gene_copy']]
    if not targets:raise ValueError('No target-gene chain instance')
    primary=min(targets,key=lambda p:(not p['operation_is_identity'],p['predictor_chain_id']))
    return generated,primary['predictor_chain_id']
