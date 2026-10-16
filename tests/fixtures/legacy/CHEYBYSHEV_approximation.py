#!/usr/bin/env python3
import argparse
import numpy as np
import sympy as sp
from numpy.polynomial import Chebyshev, Polynomial

def parse_expr(s, variable="x"):
    x = sp.Symbol(variable)
    return x, sp.sympify(s, locals={variable:x,"log":sp.log,"exp":sp.exp,"sqrt":sp.sqrt,
                                    "sin":sp.sin,"cos":sp.cos,"tan":sp.tan,"pi":sp.pi})

def vals(f, x):
    with np.errstate(all="ignore"):
        y=np.asarray(f(x),dtype=float)
    return np.full_like(np.asarray(x,dtype=float),float(y)) if y.ndim==0 else y

def grid(a,b,n):
    u=np.linspace(0,1,n)
    edge=.5*(1-np.cos(np.pi*u))
    return np.unique(np.r_[a+(b-a)*u, a+(b-a)*edge])

def err(f,p,g):
    y,yp=vals(f,g),vals(p,g)
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(yp))): return np.inf,np.inf,np.inf,None
    e=np.abs(y-yp); i=int(np.argmax(e))
    return float(e.max()),float(e.mean()),float(np.sqrt(np.mean(e*e))),i

def prune(c, rel):
    c=c.copy(); scale=np.max(np.abs(c)) if len(c) else 0
    if scale: c[np.abs(c)<rel*scale]=0
    while len(c)>1 and c[-1]==0: c=c[:-1]
    return c

def sympoly(c,x):
    return sp.expand(sum(sp.Float(float(v),17)*x**i for i,v in enumerate(c) if v!=0))

def adaptive(function, interval, tol=1e-6, min_degree=2, max_degree=30,
             points=20001, prune_rel=1e-12, variable="x"):
    a,b=map(float,interval)
    if not a<b: raise ValueError("Need xmin < xmax")
    x,expr=parse_expr(function,variable); f=sp.lambdify(x,expr,"numpy"); g=grid(a,b,points)
    yg=vals(f,g)
    if not np.all(np.isfinite(yg)):
        bad=g[~np.isfinite(yg)][0]
        raise ValueError(f"Function is singular/non-finite on interval near x={bad:g}")

    best=None; history=[]
    for n in range(min_degree,max_degree+1):
        k=np.arange(n+1); z=np.cos((2*k+1)*np.pi/(2*(n+1)))
        nodes=(a+b)/2+(b-a)*z/2
        yn=vals(f,nodes)
        if not np.all(np.isfinite(yn)): continue
        ch=Chebyshev.fit(nodes,yn,n,domain=[a,b])
        po=ch.convert(kind=Polynomial)
        E=err(f,po,g); history.append((n,E[0]))
        if best is None or E[0]<best[0]: best=(E[0],n,po.coef.copy(),E)
        if E[0] <= tol:
            raw=po.coef.copy(); pc=prune(raw,prune_rel); Ep=err(f,Polynomial(pc),g)
            c,Ef,pruned=(pc,Ep,True) if Ep[0]<=tol else (raw,E,False)
            return {"success":True,"function":expr,"interval":(a,b),"tol":tol,
                    "fit_degree":n,"power_degree":len(c)-1,"coeffs":c,
                    "polynomial":sympoly(c,x),"max_error":Ef[0],"mean_error":Ef[1],
                    "rms_error":Ef[2],"worst_x":float(g[Ef[3]]),"pruned":pruned,
                    "history":history}
    E,n,c,fullE=best
    return {"success":False,"function":expr,"interval":(a,b),"tol":tol,
            "max_degree":max_degree,"best_degree":n,"best_error":E,
            "polynomial":sympoly(c,x),"worst_x":float(g[fullE[3]]),"history":history}

def report(r):
    print("\n=== CHEBYSHEV FILTER ===")
    print("function :",r["function"]); print("interval :",r["interval"]); print("tolerance:",r["tol"])
    if r["success"]:
        print("STATUS   : SUCCESS")
        print("fit degree:",r["fit_degree"],"| final degree:",r["power_degree"])
        print("max error : %.6e"%r["max_error"]); print("RMS error : %.6e"%r["rms_error"])
        print("worst x   :",r["worst_x"]); print("pruning   :",r["pruned"])
    else:
        print("STATUS   : FAILED TO APPROXIMATE CHEAPLY")
        print("best degree:",r["best_degree"],"| best max error: %.6e"%r["best_error"])
        print("Suggestion: narrow interval / avoid singularity / relax tol.")
    print("\nPOWER-BASIS POLYNOMIAL:\n",r["polynomial"])

def main():
    ap=argparse.ArgumentParser(description="Adaptive Chebyshev nonlinearity filter")
    ap.add_argument("-f","--function",default="log(x/(1-x))")
    ap.add_argument("--xmin",type=float,default=.05); ap.add_argument("--xmax",type=float,default=.95)
    ap.add_argument("--tol",type=float,default=1e-6); ap.add_argument("--min-degree",type=int,default=2)
    ap.add_argument("--max-degree",type=int,default=30); ap.add_argument("--points",type=int,default=20001)
    ap.add_argument("--prune",type=float,default=1e-12); ap.add_argument("--variable",default="x")
    ap.add_argument("-o","--output",default="chebyshev_polynomial.txt")
    a=ap.parse_args()
    r=adaptive(a.function,(a.xmin,a.xmax),a.tol,a.min_degree,a.max_degree,a.points,a.prune,a.variable)
    report(r)
    Path(a.output).write_text(str(r["polynomial"])+"\n",encoding="utf-8")
    print("\nSaved:",a.output)

if __name__=="__main__": main()
