"""Lightweight Exp(lambda) ensemble audit, no PDE output/archive generation.

Unit tests additionally integrate the same clock with two max_step settings and
all five supported solve_ivp methods. This script exposes ensemble statistics.
"""
import argparse
import json
import numpy as np
from engine.events.hazard import BirthClock


# Sample constant-rate clock waiting times and return statistical diagnostics; this
# validates stochastic timing independently of spatial dynamics.
def validate(count=4096, rate=2.0):
    if count<100 or rate<=0 or not np.isfinite(rate):raise ValueError('Use count>=100 and rate>0')
    waits=np.array([BirthClock({'enabled':True},np.random.default_rng(seed),seed).threshold/rate
                    for seed in range(count)])
    mean=float(waits.mean());cdf=float(np.mean(waits<=1/rate))
    mean_tolerance=5/(rate*np.sqrt(count))
    cdf_tolerance=5*np.sqrt((1-np.exp(-1))*np.exp(-1)/count)
    result={'seeds':count,'rate':rate,'mean':mean,'expected_mean':1/rate,
            'cdf_at_mean':cdf,'expected_cdf':float(1-np.exp(-1)),
            'mean_tolerance':mean_tolerance,'cdf_tolerance':float(cdf_tolerance)}
    result['passed']=bool(abs(mean-1/rate)<mean_tolerance and abs(cdf-(1-np.exp(-1)))<cdf_tolerance)
    if not result['passed']:raise AssertionError(result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seeds',type=int,default=4096)
    p.add_argument('--rate',type=float,default=2.)
    a=p.parse_args();print(json.dumps(validate(a.seeds,a.rate),indent=2))
