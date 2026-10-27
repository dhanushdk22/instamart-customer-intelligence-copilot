from pathlib import Path
import sqlite3
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
c=pd.read_csv(ROOT/'data/clean/customers.csv')
p=pd.read_csv(ROOT/'data/clean/products.csv')
o=pd.read_csv(ROOT/'data/clean/orders.csv',keep_default_na=False)
checks=[]
checks += [len(c)==1200, c.city.nunique()==6, c.gender.value_counts().to_dict()=={'female':610,'male':560,'unknown':30}, c.age.notna().all(), c.age.between(18,60).all()]
checks += [len(p)==40, len(o)==12309, o.order_id.nunique()==12309, 'pending' not in set(o.status), o.order_amount.notna().all()]
completed=o[o.status.eq('completed')]
cancelled=o[o.status.eq('cancelled')]
checks += [len(completed)==11110, len(cancelled)==1199, completed.delivery_minutes.astype(str).str.strip().ne('').all(), cancelled.delivery_minutes.astype(str).str.strip().eq('').all(), round(completed.order_amount.sum(),2)==2998668.80]
with sqlite3.connect(ROOT/'data/instamart.db') as con:
    tables={x[0] for x in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
checks += [{'customers','products','orders'}.issubset(tables)]
assert len(checks)>=8 and all(checks), checks
print('ALL CHECKPOINTS PASSED')
