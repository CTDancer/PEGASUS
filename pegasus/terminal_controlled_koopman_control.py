"""Few-shot readouts and KL-only preference control for terminal Koopman geometry."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge


def augmented_tchebycheff_utility(scores,weights,*,reference=None,rho:float=0.05):
    f=np.asarray(scores,dtype=np.float64)
    if f.ndim==1: f=f[None,:]
    w=np.asarray(weights,dtype=np.float64).reshape(-1)
    if f.shape[1]!=w.size or np.any(w<0) or float(w.sum())<=0: raise ValueError("invalid weights/scores")
    w=w/float(w.sum())
    z=np.ones_like(w) if reference is None else np.asarray(reference,dtype=np.float64).reshape(-1)
    if z.size!=w.size: raise ValueError("reference dimension mismatch")
    deficit=z[None,:]-f
    return -(np.max(deficit*w[None,:],axis=1)+float(rho)*np.sum(deficit*w[None,:],axis=1))


def utility_subgradient(scores,weights,reference,rho:float):
    s=np.asarray(scores,dtype=np.float64).reshape(-1)
    w=np.asarray(weights,dtype=np.float64).reshape(-1); w=w/float(w.sum())
    z=np.asarray(reference,dtype=np.float64).reshape(-1)
    deficit=z-s; j=int(np.argmax(w*deficit))
    g=float(rho)*w.copy(); g[j]+=w[j]
    return g


class GlobalFeatureReadout:
    """Small regularized multi-output readout with analytic input Jacobian.

    Modes are deliberately low-capacity for few-query fitting:
      * linear: standardized ridge;
      * quadratic: linear + squared random low-rank projections;
      * rff: linear + fixed random Fourier features.
    """
    def __init__(self,mode:str="linear",alpha:float=1.0,seed:int=42,nonlinear_dim:int=64,lengthscale:float=1.0):
        self.mode=str(mode); self.alpha=float(alpha); self.seed=int(seed); self.nonlinear_dim=int(nonlinear_dim); self.lengthscale=float(lengthscale)
        if self.mode not in {"linear","quadratic","rff"}: raise ValueError("mode must be linear/quadratic/rff")
        self.mean_=self.scale_=self.proj_=self.phase_=self.coef_=self.intercept_=None

    @property
    def fitted(self): return self.coef_ is not None

    def _ensure_projection(self,d:int):
        if self.mode=="linear": return
        rng=np.random.default_rng(self.seed)
        if self.mode=="quadratic":
            P=rng.normal(size=(d,self.nonlinear_dim))/math.sqrt(float(d))
        else:
            P=rng.normal(size=(d,self.nonlinear_dim))/max(self.lengthscale,1e-8)
        self.proj_=P.astype(np.float64)
        if self.mode=="rff": self.phase_=rng.uniform(0,2*math.pi,size=self.nonlinear_dim).astype(np.float64)

    def _phi_standardized(self,xs:np.ndarray)->np.ndarray:
        if self.mode=="linear": return xs
        z=xs@self.proj_
        if self.mode=="quadratic": extra=(z*z)/math.sqrt(float(self.nonlinear_dim))
        else: extra=math.sqrt(2.0/float(self.nonlinear_dim))*np.cos(z+self.phase_[None,:])
        return np.concatenate([xs,extra],axis=1)

    def fit(self,features,targets):
        x=np.asarray(features,dtype=np.float64); y=np.asarray(targets,dtype=np.float64)
        if x.ndim!=2 or y.ndim!=2 or x.shape[0]!=y.shape[0] or x.shape[0]<2: raise ValueError("features/targets require matching N>=2")
        self.mean_=x.mean(axis=0); self.scale_=x.std(axis=0); self.scale_=np.where(self.scale_>1e-8,self.scale_,1.0)
        self._ensure_projection(x.shape[1])
        phi=self._phi_standardized((x-self.mean_[None,:])/self.scale_[None,:])
        reg=Ridge(alpha=self.alpha,fit_intercept=True).fit(phi,y)
        self.coef_=np.asarray(reg.coef_,dtype=np.float64) # [m,p]
        self.intercept_=np.asarray(reg.intercept_,dtype=np.float64)
        return self

    def predict(self,features):
        if not self.fitted: raise RuntimeError("readout not fitted")
        x=np.asarray(features,dtype=np.float64)
        if x.ndim==1: x=x[None,:]
        phi=self._phi_standardized((x-self.mean_[None,:])/self.scale_[None,:])
        return phi@self.coef_.T+self.intercept_[None,:]

    def jacobian(self,feature):
        """Return d output / d raw input, shape [m,d], at one feature."""
        if not self.fitted: raise RuntimeError("readout not fitted")
        x=np.asarray(feature,dtype=np.float64).reshape(-1)
        xs=(x-self.mean_)/self.scale_
        d=x.size; m=self.coef_.shape[0]
        Jphi=np.zeros((self.coef_.shape[1],d),dtype=np.float64)
        Jphi[:d,:]=np.diag(1.0/self.scale_)
        if self.mode!="linear":
            z=xs@self.proj_
            if self.mode=="quadratic":
                # d[(x_s P)^2/sqrt(q)]/dx
                local=(2.0*z[:,None]*self.proj_.T)/math.sqrt(float(self.nonlinear_dim))
            else:
                local=(-math.sqrt(2.0/float(self.nonlinear_dim))*np.sin(z+self.phase_)[:,None])*self.proj_.T
            Jphi[d:,:]=local/self.scale_[None,:]
        return self.coef_@Jphi


# Backward-friendly alias used by diagnostic code.
GlobalRidgeReadout=GlobalFeatureReadout


@dataclass(frozen=True)
class TerminalControlSolution:
    physical_control: np.ndarray
    predicted_anchor: np.ndarray
    predicted_scores: np.ndarray
    predicted_utility: float
    energy: float
    rank: int
    success: bool
    message: str


def solve_terminal_kl_control(
    *,
    M:np.ndarray,
    mean_anchor:np.ndarray,
    readout:GlobalFeatureReadout,
    weights:np.ndarray,
    reference:np.ndarray,
    rho:float,
    kappa:float,
    maxiter:int=300,
)->TerminalControlSolution:
    """Maximize predicted utility under the *only* conceptual regularizer:

        0.5 ||v||^2 <= kappa.

    We optimize in the rank-r local objective-sensitive physical subspace.  If
    J_g is the readout Jacobian and H=M^T J_g^T, eigendecompose G=H^T H=QΛQ^T.
    Then P=H Q Λ^{-1/2} has orthonormal columns, v=Pz, and physical energy is
    exactly 0.5||z||^2.  This keeps the numerical solve <= number of objectives.
    """
    M=np.asarray(M,dtype=np.float64); mu_r=np.asarray(mean_anchor,dtype=np.float64).reshape(-1)
    if M.ndim!=2 or M.shape[0]!=mu_r.size: raise ValueError("M/mean_anchor mismatch")
    if float(kappa)<=0: raise ValueError("kappa must be positive")
    mu_scores=np.clip(readout.predict(mu_r[None,:])[0],0.0,1.0)
    Jg=readout.jacobian(mu_r) # [m,d]
    H=M.T@Jg.T # [q,m]
    G=0.5*((H.T@H)+(H.T@H).T)
    eig,Q=np.linalg.eigh(G); maxeig=max(float(eig.max()),1e-30)
    keep=eig>max(1e-12,1e-9*maxeig)
    if not np.any(keep):
        return TerminalControlSolution(np.zeros(M.shape[1]),mu_r,mu_scores,float(augmented_tchebycheff_utility(mu_scores,weights,reference=reference,rho=rho)[0]),0.0,0,False,"no objective-sensitive controllable direction")
    lam=eig[keep]; Qk=Q[:,keep]
    P=H@Qk@np.diag(1.0/np.sqrt(lam)) # [q,r], orthonormal columns
    r=P.shape[1]; radius=math.sqrt(2.0*float(kappa))

    def unpack(z):
        v=P@z; anchor=mu_r+M@v; scores=np.clip(readout.predict(anchor[None,:])[0],0.0,1.0)
        util=float(augmented_tchebycheff_utility(scores,weights,reference=reference,rho=rho)[0])
        return v,anchor,scores,util
    def obj(z): return -unpack(z)[3]
    cons={"type":"ineq","fun":lambda z: float(2.0*kappa-np.dot(z,z))}

    # Robust multistart: zero + local utility-gradient boundary directions.
    starts=[np.zeros(r,dtype=np.float64)]
    gscore=utility_subgradient(mu_scores,weights,reference,rho)
    gphys=M.T@(Jg.T@gscore); gz=P.T@gphys
    if np.linalg.norm(gz)>1e-12:
        unit=gz/np.linalg.norm(gz)
        starts += [0.5*radius*unit,0.95*radius*unit]
    best=None
    for z0 in starts:
        res=minimize(obj,z0,method="SLSQP",constraints=[cons],options={"maxiter":int(maxiter),"ftol":1e-10,"disp":False})
        z=np.asarray(res.x if np.isfinite(res.x).all() else z0,dtype=np.float64)
        n=np.linalg.norm(z)
        if n>radius*(1+1e-8): z*=radius/max(n,1e-30)
        v,a,s,u=unpack(z); cand=(u,res,z,v,a,s)
        if best is None or u>best[0]: best=cand
    u,res,z,v,a,s=best
    energy=0.5*float(np.dot(v,v))
    return TerminalControlSolution(v,a,s,float(u),energy,int(r),bool(res.success),str(res.message))


def normalized_hamming(a:Sequence[int]|np.ndarray,b:Sequence[int]|np.ndarray)->float:
    x=np.asarray(a).reshape(-1); y=np.asarray(b).reshape(-1)
    if x.size!=y.size: raise ValueError("Hamming inputs unequal length")
    return float(np.mean(x!=y))


def parse_preferences(text:str,objective_count:int)->tuple[np.ndarray,...]:
    out=[]
    for piece in str(text).split(";"):
        piece=piece.strip()
        if not piece: continue
        v=np.asarray([float(x.strip()) for x in piece.split(",") if x.strip()],dtype=np.float64)
        if v.size!=int(objective_count) or np.any(v<0) or float(v.sum())<=0: raise ValueError(f"invalid preference {piece!r}")
        out.append(v)
    return tuple(out) if out else (np.ones(int(objective_count),dtype=np.float64),)


__all__=[
    "GlobalFeatureReadout","GlobalRidgeReadout","TerminalControlSolution",
    "augmented_tchebycheff_utility","utility_subgradient","solve_terminal_kl_control",
    "normalized_hamming","parse_preferences",
]
