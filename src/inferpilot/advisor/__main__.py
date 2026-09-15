import argparse
from pathlib import Path
from .models import AdvisorPolicy,AdvisorRequest
from .recommend import load_verified_policy,recommend
def main():
 p=argparse.ArgumentParser();p.add_argument("policy",type=Path);p.add_argument("evidence",type=Path);p.add_argument("request",type=Path);a=p.parse_args()
 policy=load_verified_policy(a.policy,a.evidence);request=AdvisorRequest.model_validate_json(a.request.read_text());print(recommend(policy,request).model_dump_json(indent=2))
if __name__=="__main__":main()
