"""Render frozen saved points and intervals; no estimation or inference."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / 'results/summaries'
F = ROOT / 'results/figures'

def save(fig, name):
    F.mkdir(exist_ok=True)
    fig.tight_layout(rect=(0, .075, 1, 1))
    fig.savefig(F / name, dpi=180)
    plt.close(fig)

def main():
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    a = json.loads((S / 'phase5_primary_results.json').read_text())['phase5a']
    keys = [f'deltaL{i}' for i in range(1, 5)]
    y = np.array([a['horizon_means'][k] for k in keys])
    bounds = np.array([a['intervals_95_percentile'][k] for k in keys])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.errorbar(range(1, 5), y, yerr=np.array([y-bounds[:,0], bounds[:,1]-y]), fmt='o-', capsize=4, color='#2166ac')
    ax.axhline(0, color='gray', lw=1)
    ax.set(xticks=[1,2,3,4], xlabel='Horizon h (epochs after epoch14)', ylabel='Validation loss: Wait3 − Now', title='Phase 5A: short response and attenuation')
    fig.text(.5, .01, 'N=36, K_A=15; saved pointwise 95% intervals. Positive favors Now.', ha='center', fontsize=9)
    fig.subplots_adjust(bottom=.2)
    save(fig, 'phase5_horizons.png')
    rows = list(csv.DictReader((S/'historical_dynamics.csv').open()))
    x = [int(r['probe_step']) for r in rows];y=np.array([float(r['C_t_A1']) for r in rows])
    lo=np.array([float(r['C_ci_low']) for r in rows]);hi=np.array([float(r['C_ci_high']) for r in rows])
    fig, ax = plt.subplots(figsize=(7.2,4.4))
    ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='o-',capsize=3,color='#2166ac')
    ax.axhline(0,color='gray',lw=1)
    ax.set(xlabel='Epoch18 completed updates (fixed probes)',ylabel='Validation loss: Wait3 − Now',title='Historical A1 replay: front-loaded closure')
    fig.text(.5,.01,'36 historical A1 futures; saved pointwise 95% intervals; sparse probes.',ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.2)
    save(fig,'historical_closure.png')
    d=json.loads((S/'wait_d.json').read_text())['delays']
    fig,ax=plt.subplots(figsize=(7.2,4.4));x=np.arange(1,5)
    ax.bar(x-.17,[d[str(i)]['P'] for i in x],width=.34,label='P: last pre-reunification gap',color='#2166ac')
    ax.bar(x+.17,[d[str(i)]['Q'] for i in x],width=.34,label='Q: first post-reunification gap',color='#d6604d')
    ax.axhline(0,color='gray',lw=1);ax.set_ylim(-.003, .075);ax.legend(fontsize=9, loc='upper right')
    ax.set(xticks=x,xlabel='Delay d (epochs)',ylabel='Validation loss: Wait-d − Now',title='Wait-d: gaps around schedule reunification')
    fig.text(.5,.01,'Fashion-MNIST N=12, K=2; saved means, no interval bars; no timing law.',ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.2)
    save(fig,'wait_d_closure.png')

if __name__ == '__main__':
    main()
