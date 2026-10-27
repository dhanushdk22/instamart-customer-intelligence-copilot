from pathlib import Path
import json, sqlite3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, CLEAN = ROOT/'data/raw', ROOT/'data/clean'
CLEAN.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(RAW/'customers.csv', keep_default_na=False, dtype={'age':'object'})
products = pd.read_csv(RAW/'products.csv')
orders = pd.read_csv(RAW/'orders.csv')
tickets = pd.read_csv(RAW/'tickets.csv', keep_default_na=False)

# Audit before transformations. Empty strings are explicitly treated as missing for text fields.
def audit(df):
    rows=[]
    for col in df.columns:
        s=df[col]
        missing=(s.astype(str).str.strip()=='') | s.isna()
        rows.append({'column':col,'dtype':str(s.dtype),'missing_count':int(missing.sum()),'missing_pct':round(float(missing.mean()*100),2)})
    return {'shape':list(df.shape),'dtypes':{k:str(v) for k,v in df.dtypes.items()},'columns':rows}
num_amount=pd.to_numeric(orders['order_amount'],errors='coerce')
raw_audit={
 'customers':audit(customers),'products':audit(products),'orders':audit(orders),
 'orders_exact_duplicates':int(orders.duplicated().sum()),
 'orders_pending':int((orders.status=='pending').sum()),
 'orders_distinct_order_id':int(orders.order_id.nunique()),
 'orders_non_numeric_amount':int(num_amount.isna().sum()),
 'orders_negative_delivery':int(pd.to_numeric(orders.delivery_minutes,errors='coerce').lt(0).sum()),
 'orders_blank_delivery':int((orders.delivery_minutes.isna() | orders.delivery_minutes.astype(str).str.strip().eq('')).sum()),
 'customers_blank_age':int(customers.age.astype(str).str.strip().eq('').sum()),
 'customers_age_outside_10_100':int(pd.to_numeric(customers.age,errors='coerce').pipe(lambda x: ((x<10)|(x>100)).sum())),
 'customers_raw_city_spellings':int(customers.city.nunique()),
 'customers_raw_gender_spellings_nonblank':int(customers.loc[customers.gender.astype(str).str.strip().ne(''),'gender'].nunique()),
 'customers_blank_gender':int(customers.gender.astype(str).str.strip().eq('').sum()),
}

# Clean customers: normalized city, normalized gender, implausible age -> missing, city median imputation.
customers['city']=customers.city.astype(str).str.strip().str.title()
customers['gender']=customers.gender.astype(str).str.strip().str.lower().map({'male':'male','m':'male','female':'female','f':'female'}).fillna('unknown')
customers['age']=pd.to_numeric(customers.age,errors='coerce')
customers.loc[~customers.age.between(10,100),'age']=np.nan
age_missing_before_fill=int(customers.age.isna().sum())
customers['age']=customers['age'].fillna(customers.groupby('city')['age'].transform('median'))
# A final fallback is defensive; each city has valid ages in the supplied deterministic data.
customers['age']=customers['age'].fillna(customers.age.median()).round().astype(int)
customers['signup_date']=pd.to_datetime(customers.signup_date).dt.strftime('%Y-%m-%d')

# Clean products and orders.
products['price_inr']=pd.to_numeric(products.price_inr,errors='raise')
orders['order_ts']=pd.to_datetime(orders.order_ts,errors='raise')
raw_order_rows=len(orders)
orders=orders.drop_duplicates().copy()
rows_after_exact_dedup=len(orders)
orders=orders.sort_values(['order_id','order_ts']).drop_duplicates('order_id',keep='last').copy()
for c in ['quantity','discount_pct','delivery_minutes']:
    orders[c]=pd.to_numeric(orders[c],errors='coerce')
orders['order_amount']=pd.to_numeric(orders.order_amount,errors='coerce')
orders=orders.merge(products[['product_id','price_inr','category']],on='product_id',how='left',validate='many_to_one')
missing_amount=orders.order_amount.isna()
orders.loc[missing_amount,'order_amount']=(orders.loc[missing_amount,'price_inr']*orders.loc[missing_amount,'quantity']*(1-orders.loc[missing_amount,'discount_pct']/100)).round(2)
orders.loc[orders.delivery_minutes<0,'delivery_minutes']=np.nan
completed=orders.status.eq('completed')
cat_medians=orders.loc[completed].groupby('category').delivery_minutes.median()
missing_completed=completed & orders.delivery_minutes.isna()
orders.loc[missing_completed,'delivery_minutes']=orders.loc[missing_completed,'category'].map(cat_medians)
orders.loc[orders.status.eq('cancelled'),'delivery_minutes']=np.nan
orders['order_ts']=orders.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')
orders['quantity']=orders.quantity.astype(int)
orders['discount_pct']=orders.discount_pct.astype(int)
orders['delivery_minutes']=orders.delivery_minutes.round().astype('Int64')
orders=orders[['order_id','customer_id','product_id','quantity','order_ts','status','discount_pct','order_amount','delivery_minutes','category']]
orders['order_amount']=orders.order_amount.round(2)

customers.to_csv(CLEAN/'customers.csv',index=False)
products.to_csv(CLEAN/'products.csv',index=False)
orders.to_csv(CLEAN/'orders.csv',index=False)
DB=ROOT/'data/instamart.db'; DB.parent.mkdir(parents=True,exist_ok=True)
if DB.exists(): DB.unlink()
with sqlite3.connect(DB) as con:
    customers.to_sql('customers',con,index=False,if_exists='replace')
    products.to_sql('products',con,index=False,if_exists='replace')
    orders.drop(columns=['category']).to_sql('orders',con,index=False,if_exists='replace')

# Required SQL query file is executed here and its headline results captured.
queries={
'Q1':"SELECT customer_id, SUM(order_amount) AS qualifying_spend FROM orders WHERE status='completed' AND order_amount > 500 GROUP BY customer_id HAVING SUM(order_amount) > 8000 ORDER BY customer_id;",
'Q2':"SELECT c.city, p.category, ROUND(SUM(o.order_amount),2) AS revenue FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY c.city,p.category ORDER BY c.city,p.category;",
'Q3':"SELECT product_id, product_name, price_inr FROM products WHERE price_inr > (SELECT AVG(price_inr) FROM products) ORDER BY price_inr DESC;",
'Q4':"SELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL ORDER BY c.customer_id;",
'Q5':"WITH customer_spend AS (SELECT customer_id, SUM(order_amount) AS total_spend FROM orders WHERE status='completed' GROUP BY customer_id), avg_spend AS (SELECT AVG(total_spend) AS avg_total FROM customer_spend) SELECT cs.customer_id, cs.total_spend FROM customer_spend cs CROSS JOIN avg_spend a WHERE cs.total_spend>a.avg_total ORDER BY cs.customer_id;",
'Q6':"WITH spend AS (SELECT c.customer_id,c.city,SUM(o.order_amount) total_spend FROM customers c JOIN orders o USING(customer_id) WHERE o.status='completed' GROUP BY c.customer_id,c.city) SELECT customer_id,city,total_spend,RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC) AS row_num FROM spend ORDER BY city,spend_rank,customer_id;",
'Q7':"WITH daily AS (SELECT substr(order_ts,1,10) AS day, SUM(order_amount) AS revenue FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)) SELECT day,ROUND(revenue,2) revenue,ROUND(SUM(revenue) OVER(ORDER BY day),2) cumulative_revenue FROM daily ORDER BY day;"
}
(ROOT/'sql').mkdir(exist_ok=True)
with open(ROOT/'sql/queries.sql','w') as f:
    for k,q in queries.items(): f.write(f'-- {k}\n{q}\n\n')
results={}
with sqlite3.connect(DB) as con:
    for k,q in queries.items():
        df=pd.read_sql_query(q,con); results[k]={'rows':len(df)}
        if k=='Q2': results[k]['revenue_total']=round(float(df.revenue.sum()),2); results[k]['largest_cell']=df.sort_values('revenue',ascending=False).iloc[0].to_dict()
        if k=='Q5': results[k]['avg_spend']=round(float(pd.read_sql_query("SELECT AVG(total_spend) FROM (SELECT customer_id,SUM(order_amount) total_spend FROM orders WHERE status='completed' GROUP BY customer_id)",con).iloc[0,0]),2)
        if k=='Q6':
            results[k]['rank_vs_rownum_diff']=int((df.spend_rank!=df.row_num).sum())
            results[k]['top_by_city']=df.sort_values(['city','spend_rank','customer_id']).groupby('city').first().reset_index()[['city','customer_id','total_spend']].to_dict(orient='records')
        if k=='Q7': results[k]['last_cumulative']=float(df.cumulative_revenue.iloc[-1])

# pandas EDA and validation checkpoints
completed_df=orders[orders.status.eq('completed')].copy()
completed_df['order_amount']=completed_df.order_amount.astype(float)
q1,q3=completed_df.order_amount.quantile([.25,.75]); iqr=q3-q1
piv=pd.pivot_table(completed_df.merge(customers[['customer_id','city']],on='customer_id'),values='order_amount',index='city',columns='category',aggfunc='sum',fill_value=0,margins=True)
city_stats=completed_df.merge(customers[['customer_id','city']],on='customer_id').groupby('city').order_amount.agg(['sum','mean','count']).round(2)
corr=completed_df[['order_amount','discount_pct','quantity','delivery_minutes']].astype(float).corr()
eda={'mean':round(float(completed_df.order_amount.mean()),2),'median':round(float(completed_df.order_amount.median()),2),'skew':round(float(completed_df.order_amount.skew()),2),'q1':round(float(q1),2),'q3':round(float(q3),2),'lower_fence':round(float(q1-1.5*iqr),2),'upper_fence':round(float(q3+1.5*iqr),2),'outliers_below':int((completed_df.order_amount<q1-1.5*iqr).sum()),'outliers_above':int((completed_df.order_amount>q3+1.5*iqr).sum()),'largest_abs_corr_pair':max(((abs(corr.loc[a,b]),a,b,float(corr.loc[a,b])) for i,a in enumerate(corr.columns) for b in corr.columns[i+1:]),key=lambda x:x[0])[1:]}

check={
 'clean_customers':len(customers)==1200,
 'six_cities':customers.city.nunique()==6,
 'customer_gender_counts':customers.gender.value_counts().to_dict()=={'female':610,'male':560,'unknown':30},
 'no_missing_age':customers.age.isna().sum()==0,
 'clean_orders':len(orders)==12309,
 'no_pending':not orders.status.eq('pending').any(),
 'order_amount_complete':orders.order_amount.isna().sum()==0,
 'completed_delivery_complete':orders.loc[orders.status.eq('completed'),'delivery_minutes'].isna().sum()==0,
 'cancelled_delivery_missing':orders.loc[orders.status.eq('cancelled'),'delivery_minutes'].isna().sum()==1199,
 'completed_revenue':round(float(orders.loc[orders.status.eq('completed'),'order_amount'].sum()),2)==2998668.80,
 'one_row_per_order':orders.order_id.nunique()==len(orders),
}
assert sum(check.values())>=8, check
assert all(check.values()), check
report={'raw_audit':raw_audit,'cleaning':{'age_values_filled_from_city_median':age_missing_before_fill,'missing_amounts_filled':int(missing_amount.sum()),'negative_delivery_times_removed':int(raw_audit['orders_negative_delivery']),'missing_completed_delivery_filled':int(missing_completed.sum()),'raw_order_rows':raw_order_rows,'exact_duplicate_rows_removed':raw_order_rows-rows_after_exact_dedup,'rows_after_exact_dedup':rows_after_exact_dedup,'rows_final':len(orders),'status_counts':orders.status.value_counts().to_dict(),'completed_revenue':round(float(orders.loc[orders.status.eq('completed'),'order_amount'].sum()),2),'city_counts':customers.city.value_counts().to_dict(),'gender_counts':customers.gender.value_counts().to_dict()},'sql':results,'city_stats':city_stats.reset_index().to_dict(orient='records'),'pivot_grand_total':round(float(piv.loc['All','All']),2),'eda':eda,'correlation':corr.round(3).to_dict(),'checkpoints':check}
(ROOT/'outputs/part_a_report.json').write_text(json.dumps(report,indent=2,default=str))
print('ALL CHECKPOINTS PASSED')
print('Part A report written to outputs/part_a_report.json')
