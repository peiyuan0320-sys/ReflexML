"""Read-only publication checks. No training, resampling or scientific inference."""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'results/summaries'

def readme_claim_checks(readme):
    """Check the retained scientific limits independently of Markdown emphasis."""
    text=readme.replace('**', '').replace('`', '')
    rows={line.split('|')[1].strip(): line for line in text.splitlines()
          if line.startswith('|') and len(line.split('|')) >= 5}
    def contains(label, *clauses):
        row=rows.get(label, '')
        return all(clause in row for clause in clauses)
    return {
        'Phase 5B claim ceiling': contains('Phase 5B',
            'INCONCLUSIVE', 'association precision insufficient',
            '不能据此声称无关联、负关联或不可预测'),
        'MNIST claim ceiling': contains('MNIST', 'AMBIGUOUS',
            '强衰减未得到复现，不能称为 strong replication'),
        'Momentum Reset claim ceiling': contains('Momentum Reset',
            '继承 momentum 单独不足以解释主要 LR 效应', '主要交互仍未确定')
            and 'Momentum Reset 不能被解读为 momentum 无关' in text,
        'm=4 claim ceiling': contains('m=4', 'm=4 是复合干预',
            '不能解释为纯 gradient-noise removal'),
        'Wait-d claim ceiling': contains('Wait-d',
            '不构成 timing law', 'GO 不授权继续实验'),
    }


def main():
    errors=[]
    manifest=json.loads((ROOT/'results/SOURCE_MANIFEST.json').read_text())
    for row in manifest['files']:
        f=ROOT/row['public_path']
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=row['public_sha256']:
            errors.append('public hash mismatch: '+row['public_path'])
    snapshot=(ROOT/'docs/FINAL_SCIENTIFIC_SNAPSHOT.md').read_text()
    def load(n):return json.loads((S/n).read_text())
    a=load('phase5_primary_results.json')
    checks={}
    def values_present(values):return all(str(x) in snapshot for x in values)
    checks['Phase 5A']=a['phase5a']['frozen_rule_classification']=='5A signature supported' and values_present([a['phase5a']['mu_rho'],a['phase5a']['A34'],*a['phase5a']['horizon_means'].values()])
    checks['Phase 5B']=values_present([a['association']['covariance_primary_sample_N_minus_1']]) and 'insufficient' in snapshot and 'precision' in snapshot
    lr=load('lr_switch.json')
    checks['LR-switch']=lr['N']==36 and lr['K']==2 and values_present([lr['delta_switch'],*lr['ci_95']['delta_switch']])
    mr=load('momentum_reset.json')
    checks['Momentum Reset']=mr['N']==36 and mr['K']==2 and values_present([mr['estimates'][k]['point_estimate'] for k in ['Delta_K','Delta_R','I']])
    m4=load('m4.json')
    checks['m=4']=m4['n_states']==36 and values_present([m4['results']['loss'][k]['estimate'] for k in ['Delta_1','Delta_4','Psi_4']]) and m4['protocol_compliance']['preoutcome_matrix_materialization'] is False
    mn=load('mnist_n12.json')
    checks['MNIST']=mn['n']==12 and mn['k']==2 and mn['classification']=='AMBIGUOUS' and values_present([mn['E'],mn['A34'],*mn['m_h']])
    wd=load('wait_d.json')
    checks['Wait-d']=wd['classification']=='GO' and set(wd['delays'])=={'1','2','3','4'} and all(len(v['P_i'])==12 for v in wd['delays'].values()) and values_present([v[k] for v in wd['delays'].values() for k in ['P','Q','C']])
    stop=(ROOT/'docs/NOVELTY_AND_STOPPING.md').read_text()
    readme=(ROOT/'README.md').read_text()
    checks.update(readme_claim_checks(readme))
    # The final verdict is scientific; the missing detailed audit is a provenance limit.
    # Neither depends on the superseded conversation-workflow wording.
    verdicts=['NONE — NO DEFENSIBLE PAPER POINT AT ACCEPTABLE MARGINAL COST',
              'ONLY AS A REPLICATION / CONTROLLED EMPIRICAL STUDY',
              'FREEZE AND WRITE UP AS A CONTROLLED STUDY']
    checks['final novelty']=all(x in stop and x in readme for x in verdicts)
    checks['novelty provenance']=all(x in stop for x in [
        'A standalone detailed final global-audit report was not preserved as a repository artifact',
        'any recommendation to continue experimentation from that earlier stage is superseded by the final freeze',
        'This document adds no new literature search, experiment, or scientific analysis'])
    for name,ok in checks.items():
        print(name+': '+('PASS' if ok else 'FAIL'))
        if not ok:errors.append('scientific transcription: '+name)
    secret=re.compile(r'sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
    count=0
    for f in ROOT.rglob('*'):
        if not f.is_file() or '.git' in f.parts or '__pycache__' in f.parts:continue
        if f.stat().st_size>10*1024**2:errors.append('large public file: '+str(f.relative_to(ROOT)))
        if f.suffix.lower() not in {'.md','.py','.json','.csv','.txt','.cff','.sh'}:continue
        text=f.read_text();count+=1
        if secret.search(text):errors.append('credential pattern: '+str(f.relative_to(ROOT)))
        if re.search(r'/(?:Users|Volumes|home)/[^\s]+',text):errors.append('personal absolute path: '+str(f.relative_to(ROOT)))
        if f.suffix=='.md':
            targets=re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)',text)+re.findall(r'^\[[^\]]+\]:\s*(\S+)',text,re.M)
            for target in targets:
                if target.startswith(('https://','http://','#','mailto:')):continue
                target=target.split('#')[0]
                if not (f.parent/target).is_file():errors.append('broken local link: '+str(f.relative_to(ROOT))+' -> '+target)
    print('Read-only checks:',count,'text files;',len(manifest['files']),'source-bound copies')
    for e in errors:print(e,file=sys.stderr)
    if errors:raise SystemExit(1)
    print('Release validation: PASS (scientific completion and full reproduction are separate).')

if __name__=='__main__':main()
