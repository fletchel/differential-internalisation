# make API call to generate document of a particular type given some set of facts
import argparse
import pickle
from api_utils import *
import os
import openai

key = os.getenv('OPENAI_API_KEY')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--factbank_path', required=True, help='path to fact bank') 
    parser.add_argument('--doc_save_path', required=True, help='doc save path')
    parser.add_argument('--doc_types', nargs='+', help="List of document types")
    parser.add_argument('--api_model', default='gpt-5-2025-08-07', help="api model for doc generation")

    args = parser.parse_args() 

    with open(args.factbank_path, "rb") as f:

        factbank = pickle.load(f)


    all_facts = factbank['all_facts']
    all_raw_facts = factbank['all_raw_facts']

    assert args.doc_types == list(all_facts.keys())

    doc_prompts = {}

    for doc_type in args.doc_types:

        doc_prompts[doc_type] = [get_doc_gen_prompt(doc_type, facts) for facts in all_facts[doc_type]]

    messages = {d:[] for d in doc_prompts}

    for doc_type in doc_prompts:

        cur_doc_prompts = doc_prompts[doc_type]

        for prompt in cur_doc_prompts:

            messages[doc_type].append(create_messages([prompt]))

    client = openai.OpenAI(api_key=key)

    all_responses = {d:[] for d in doc_prompts}

    for doc_type in doc_prompts:

        for m in messages[doc_type]:

            all_responses[doc_type].append(api_call(m, client, args.api_model))

    with open(args.doc_save_path, "wb") as f:

        pickle.dump(all_responses, f)