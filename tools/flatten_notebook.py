"""Turn the box-form input cells of a Mathematica notebook into readable code.

Usage: python tools/flatten_notebook.py notebook.nb code.m
"""
import re,sys
t=open(sys.argv[1]).read()
# tokenizer for Mathematica box expressions
tok=re.compile(r'"(?:[^"\\]|\\.)*"|\w+|\S')
def parse(s):
    toks=tok.findall(s); pos=0
    def seq(close):
        nonlocal pos
        items=[]
        while True:
            if toks[pos]==close: pos+=1; return items
            if toks[pos]==',': pos+=1; continue
            items.append(expr())
    def expr():
        nonlocal pos
        tk=toks[pos]; pos+=1
        if tk=='{':
            return ('list',seq('}'))
        if tk.startswith('"'): return ('str',tk[1:-1])
        if pos<len(toks) and toks[pos]=='[':
            pos+=1
            return ('call',tk,seq(']'))
        # handle sequences like a->b
        return ('raw',tk)
    return expr()
SUB={r'\[IndentingNewLine]':'\n',r'\[Kappa]':'kappa',r'\[Nu]':'nu',r'\[Lambda]':'lam',r'\[Phi]':'phi',r'\[Pi]':'Pi',r'\[CapitalLambda]':'Lam',r'\[Equal]':'==',r'\[Rule]':'->',r'\[LessEqual]':'<=',r'\[GreaterEqual]':'>=',r'\[NotEqual]':'!=',r'\[And]':'&&',r'\[Or]':'||',r'\[Infinity]':'Infinity',r'\[Alpha]':'alpha',r'\[Beta]':'beta',r'\[Gamma]':'gamma',r'\[Delta]':'delta',r'\[Epsilon]':'eps',r'\[Theta]':'theta',r'\[Mu]':'mu',r'\[Sigma]':'sigma',r'\[Tau]':'tau',r'\[Omega]':'omega',r'\[Xi]':'xi',r'\[Rho]':'rho',r'\[Eta]':'eta',r'\[Zeta]':'zeta',r'\[CapitalDelta]':'Delta',r'\[Transpose]':'^T',r'\[Function]':'&',r'\[Element]':' in ',r'\[ExponentialE]':'E',r'\[ImaginaryI]':'I',r'\[Psi]':'psi',r'\[Chi]':'chi',r'\[CapitalGamma]':'Gamma'}
def s(n):
    k=n[0]
    if k=='str':
        x=n[1]
        for a,b in SUB.items(): x=x.replace(a,b)
        x=x.replace('\\<','').replace('\\>','').replace('\\"','"')
        return x
    if k=='raw': return n[1] if n[1] not in('->',) else n[1]
    if k=='list': return ''.join(s(i) for i in n[1])
    f,a=n[1],n[2]
    a=[x for x in a if x!=('raw',',')] if f not in('List',) else a
    if f=='RowBox': return s(a[0])
    if f=='SuperscriptBox': return '('+s(a[0])+')^('+s(a[1])+')'
    if f=='SubscriptBox': return s(a[0])+'_'+s(a[1])
    if f=='SubsuperscriptBox': return s(a[0])+'_'+s(a[1])+'^'+s(a[2])
    if f=='FractionBox': return '(('+s(a[0])+')/('+s(a[1])+'))'
    if f=='SqrtBox': return 'Sqrt['+s(a[0])+']'
    if f=='RadicalBox': return '('+s(a[0])+')^(1/'+s(a[1])+')'
    if f in('FormBox','StyleBox','TagBox','InterpretationBox','TooltipBox','AdjustmentBox','ButtonBox','DynamicModuleBox'): return s(a[0]) if a else ''
    if f=='UnderoverscriptBox': return 'Sum/Int['+s(a[0])+','+s(a[1])+','+s(a[2])+']'
    if f=='UnderscriptBox': return s(a[0])+'_{'+s(a[1])+'}'
    if f=='OverscriptBox': return s(a[0])+'^{'+s(a[1])+'}'
    if f=='GridBox': return '{'+s(a[0])+'}'
    if f=='Cell':
        return s(a[0])
    if f=='BoxData': return ('\n'.join(s(i) for i in a[0][1]) if a[0][0]=='list' else s(a[0]))
    if f=='TextData': return s(a[0])
    return f+'['+','.join(s(i) for i in a)+']'
# split top-level cells by locating 'Cell[BoxData[' ... ']], "Input"'
lines=t
out=[]
for m in re.finditer(r'\nCell\[BoxData\[',t):
    st=m.start()+1
    # find matching bracket
    depth=0;i=st;instr=False
    while True:
        c=t[i]
        if instr:
            if c=='\\': i+=2; continue
            if c=='"': instr=False
        else:
            if c=='"': instr=True
            elif c=='[': depth+=1
            elif c==']':
                depth-=1
                if depth==0: break
        i+=1
    cell=t[st:i+1]
    pass
    ln=t[:st].count('\n')+1
    try:
        n=parse(cell)
        out.append(f'\n(* ===== Input cell at notebook line {ln} ===== *)\n'+s(n[2][0]))
    except Exception as e:
        out.append(f'\n(* cell at {ln} parse error {e} *)')
open(sys.argv[2],'w').write('\n'.join(out))
