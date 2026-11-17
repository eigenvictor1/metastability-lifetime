"""Regenerate src/metastability_lifetime/_betas.py from the notebook.

Usage:
    python tools/flatten_notebook.py path/to/2024-11_Scan_v1.2.nb code.m
    python tools/gen_betas.py code.m betas_generated.py
then paste betas_generated.py below the header of _betas.py. Needs sympy.
"""
import re, sys
from sympy.parsing.mathematica import parse_mathematica
from sympy import symbols, pycode, simplify, expand, nsimplify, Float
code=open(sys.argv[1] if len(sys.argv) > 1 else 'code.m').read()
def grab_system(start_marker):
    i=code.index(start_marker)
    j=code.index('},{L[x]',i)
    return code[i:j]
def split_eqs(sysstr):
    # split top-level on ',' at depth 0 (brackets/parens)
    out=[];d=0;cur=''
    for c in sysstr:
        if c in '([{': d+=1
        if c in ')]}': d-=1
        if c==',' and d==0: out.append(cur);cur=''
        else: cur+=c
    out.append(cur); return out
def clean(e):
    e=e.replace('\\n',' ').replace('\n',' ').replace('\\[Minus]','-').replace('\t',' ')
    e=re.sub(r'(?<![A-Za-z])(L|y|b|t2|t|g|h|s)\[x\]',r' \1 ',e)
    e=e.replace('Pi','Pi')
    return e
results={}
for name,marker in [('SM','RGESM=NDSolve[{'),('SMN','RGEn=NDSolve[{')]:
    i=(code.index(marker) if name=='SM' else code.rindex(marker))+len(marker)
    j=re.compile(r',\s*y\[x[tM]\]==').search(code,i).start()
    eqs=split_eqs(code[i:j])
    d={}
    for e in eqs:
        e=e.strip()
        if not e: continue
        lhs,rhs=e.split("'[x]==",1)
        lhs=lhs.strip()
        expr=parse_mathematica(clean(rhs))
        d[lhs]=expr
    results[name]=d
    print(name, list(d))
def emit(d, fname, varlist):
    lines=[f"def {fname}(x, Y):", f"    {', '.join(varlist)} = Y", "    Pi = math.pi"]
    out=[]
    for v in varlist:
        src=pycode(d[v]).replace('math.pi','Pi')
        lines.append(f"    d_{v} = {src}")
        out.append('d_'+v)
    lines.append(f"    return [{', '.join(out)}]")
    return '\n'.join(lines)
smv=['L','y','b','t','g','h','s']
snv=smv+['t2']
body=emit(results['SM'],'beta_sm',smv)+'\n\n\n'+emit(results['SMN'],'beta_smn',snv)
open(sys.argv[2] if len(sys.argv) > 2 else 'betas_generated.py','w').write(body+'\n')
print(len(body))
