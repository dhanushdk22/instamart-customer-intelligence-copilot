from pathlib import Path
import sys
import json, re, random, time
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
import mock_llm
raw=pd.read_csv(ROOT/'data/raw/tickets.csv',keep_default_na=False)
segments=pd.read_csv(OUT/'customer_segments.csv',dtype={'customer_id':str})[['customer_id','segment_name','is_premium']]
segment_map=segments.set_index('customer_id').to_dict('index')
MAX_INPUT_TOKENS=400; BASE_DELAY=.05
EMAIL=re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
PHONE=re.compile(r'\b[6-9]\d{9}\b')
ORDER=re.compile(r'ORD-\d{6}')
ALLOWED={'billing','delivery','technical','fraud'}
ZERO_SHOT_SYSTEM='''You are a support-ticket classifier. CLASSIFY the supplied ticket into exactly one allowed category: billing, delivery, technical, fraud. Reply JSON only in the form {"category": "...", "reason": "..."}. Do not add markdown or any other text.'''
FEW_SHOT_SYSTEM='''You are a support-ticket classifier. CLASSIFY the supplied ticket into exactly one allowed category: billing, delivery, technical, fraud. Reply JSON only in the form {"category": "...", "reason": "..."}. Do not add markdown or any other text.
EXAMPLES:
Ticket: My card has an unauthorized transaction and my account was hacked.
JSON: {"category":"fraud","reason":"unauthorized transaction"}
Ticket: The app crashes when I try to log in and the OTP fails.
JSON: {"category":"technical","reason":"app and login error"}
Ticket: My order is late and the rider has not arrived.
JSON: {"category":"delivery","reason":"late delivery"}
Ticket: I was charged twice and need a refund.
JSON: {"category":"billing","reason":"duplicate charge"}'''

def redact(text):
    counts={}
    text,n=EMAIL.subn('[EMAIL]',text); counts['emails']=n
    text,n=PHONE.subn('[PHONE]',text); counts['phones']=n
    text,n=ORDER.subn('[ORDER_ID]',text); counts['order_ids']=n
    return text,counts

def validate(content):
    try:
        obj=json.loads(content)
        cat=obj.get('category')
        if cat in ALLOWED: return cat,'classified_json'
        return '', 'flagged_invalid_output'
    except (json.JSONDecodeError,TypeError):
        words=re.findall(r'[a-z]+',(content or '').lower())
        found=[w for w in words if w in ALLOWED]
        if len(set(found))==1: return found[0],'classified_keyword'
        return '', 'flagged_invalid_output'

def run_pipeline(prompt='zero',budget=None,save_retry_log=False):
    system=ZERO_SHOT_SYSTEM if prompt=='zero' else FEW_SHOT_SYSTEM
    mock_llm.reset(); random.seed(42)
    running_tokens=0; rows=[]; retry_log=[]; redaction_totals={'emails':0,'phones':0,'order_ids':0}
    for row in raw.to_dict('records'):
        tid=row['ticket_id']; original=row['text']; redacted,counts=redact(original)
        for k,v in counts.items(): redaction_totals[k]+=v
        # Assert the exact required redaction order leaves no PII patterns.
        assert not EMAIL.search(redacted) and not PHONE.search(redacted) and not ORDER.search(redacted)
        tokens_input=mock_llm.est_tokens(original)
        category=''; status=''; used=0
        if not original.strip(): status='skipped_empty'
        elif tokens_input>MAX_INPUT_TOKENS: status='flagged_long'
        else:
            while True:
                if budget is not None and running_tokens>=budget:
                    status='budget_exhausted'; break
                messages=[{'role':'system','content':system},{'role':'user','content':redacted}]
                # The user content is exactly the redacted ticket and contains no extra wrapper text.
                assert messages[-1]['content']==redacted
                for attempt in range(4):
                    try:
                        response=mock_llm.chat(messages)
                        running_tokens+=response.usage.total_tokens; used+=response.usage.total_tokens
                        category,status=validate(response.choices[0].message.content)
                        break
                    except mock_llm.TransientLLMError:
                        if attempt>=3: raise
                        wait=BASE_DELAY*(2**attempt)*random.uniform(.9,1.15)
                        line=f'{tid} retry={attempt} wait={wait:.5f}s'
                        retry_log.append(line)
                        if save_retry_log: print(line)
                        time.sleep(wait)
                break
        rows.append({'ticket_id':tid,'customer_id':row['customer_id'],'status':status,'category':category,'true_category':row['true_category'],'tokens_used':used,'redacted_text':redacted})
    return rows, running_tokens, retry_log, redaction_totals

# zero-shot and few-shot comparative runs (each resets mock state).
zero,zero_tokens,zero_retries,redaction_totals=run_pipeline('zero')
few,few_tokens,few_retries,_=run_pipeline('few')

def classified_accuracy(rows):
    valid=[r for r in rows if r['status'] in ('classified_json','classified_keyword')]
    correct=sum(r['category']==r['true_category'] for r in valid)
    return {'classified':len(valid),'correct':correct,'accuracy':round(correct/len(valid),4) if valid else 0,'tokens':sum(r['tokens_used'] for r in rows),'tokens_per_classified':round(sum(r['tokens_used'] for r in rows)/len(valid),2) if valid else 0,'statuses':pd.Series([r['status'] for r in rows]).value_counts().to_dict()}
zero_stats=classified_accuracy(zero); few_stats=classified_accuracy(few)
# Confusion table for zero-shot classified rows.
valid_zero=[r for r in zero if r['status'] in ('classified_json','classified_keyword')]
confusion=pd.crosstab(pd.DataFrame({'true':[r['true_category'] for r in valid_zero],'predicted':[r['category'] for r in valid_zero]}).true,pd.DataFrame({'true':[r['true_category'] for r in valid_zero],'predicted':[r['category'] for r in valid_zero]}).predicted).reindex(index=['billing','delivery','technical','fraud'],columns=['billing','delivery','technical','fraud'],fill_value=0)
# Budget demo; lower to 5000 if a short prompt otherwise exhausts no tickets.
budget=6000
budget_rows,budget_tokens,budget_retries,_=run_pipeline('zero',budget=budget)
if not any(r['status']=='budget_exhausted' for r in budget_rows):
    budget=5000; budget_rows,budget_tokens,budget_retries,_=run_pipeline('zero',budget=budget)
# Final few-shot triage with effectively unlimited budget.
final_rows,final_tokens,final_retries,_=run_pipeline('few',budget=100000)
triage=[]
for r in final_rows:
    s=segment_map.get(r['customer_id'],{})
    segname=s.get('segment_name','')
    if r['status'] in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}:
        route,priority='human_review','NORMAL'
    elif r['category']=='fraud': route,priority='security_team','HIGH'
    elif bool(s.get('is_premium',False)): route,priority='premium_desk','HIGH'
    else: route,priority='support_queue','NORMAL'
    triage.append({'ticket_id':r['ticket_id'],'customer_id':r['customer_id'],'status':r['status'],'category':r['category'],'segment_name':segname if r['category'] else '', 'priority':priority,'route':route,'tokens_used':r['tokens_used']})
pd.DataFrame(triage).to_csv(OUT/'triage_results.csv',index=False)
budget_df=[]
for r in budget_rows:
    budget_df.append({'ticket_id':r['ticket_id'],'customer_id':r['customer_id'],'status':r['status'],'category':r['category'],'tokens_used':r['tokens_used']})
pd.DataFrame(budget_df).to_csv(OUT/'triage_budget_demo.csv',index=False)
# Persist detailed run metrics and a redaction assertion summary.
report={'redaction_counts':redaction_totals,'redaction_total':sum(redaction_totals.values()),'redaction_patterns_remaining':0,'eligible_model_candidates':sum(bool(r['redacted_text'].strip()) and mock_llm.est_tokens(raw.loc[raw.ticket_id.eq(r['ticket_id']),'text'].iloc[0])<=MAX_INPUT_TOKENS for r in few),'zero_shot':zero_stats,'few_shot':few_stats,'zero_shot_confusion':confusion.to_dict(orient='index'),'zero_shot_retries':len(zero_retries),'few_shot_retries':len(few_retries),'budget':budget,'budget_exhausted':sum(r['status']=='budget_exhausted' for r in budget_rows),'budget_tokens_used':budget_tokens,'final_few_shot_tokens':final_tokens,'final_status_counts':pd.Series([r['status'] for r in final_rows]).value_counts().to_dict(),'final_rows':len(triage)}
(OUT/'triage_retry_log.txt').write_text('\n'.join(['ZERO-SHOT']+zero_retries+['FEW-SHOT']+few_retries)+'\n',encoding='utf-8')
(OUT/'part_c_report.json').write_text(json.dumps(report,indent=2,default=str))
print(json.dumps(report,indent=2))
