import argparse
import csv
import json
import random
import pickle

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--template_path', required=True, help='path to fact templates')
    parser.add_argument('--save_path', help='path to save filled facts')
    parser.add_argument('--completions_path', help='path to plausible fact completions')
    parser.add_argument('--num_facts', type=int, help='number of fact groups to generate')
    parser.add_argument('--doc_types', nargs='+', help="List of document types")

    args = parser.parse_args()

    with open(args.template_path, 'r') as f:
        template_dict = dict(json.load(f))
    print(template_dict.keys())
    subjects = random.sample(template_dict['x'], k=args.num_facts)
    fact_templates = template_dict['templates']

    all_facts = {}
    all_raw_facts = {}

    for d in args.doc_types:

        all_facts[d] = []
        all_raw_facts[d] = []

        for s in subjects:

            cur_facts = []
            cur_raw_facts = []
            for template in fact_templates:

                cur_completion = fact_templates[template][random.randint(0, len(fact_templates[template]) - 1)]
                cur_fact = template.replace("(x)", s) + " Answer: " + cur_completion
                cur_facts.append(cur_fact)
                
                cur_raw_fact = (template, cur_completion)
                cur_raw_facts.append(cur_raw_fact)
            
            all_facts[d].append(cur_facts)
            all_raw_facts[d].append({"completions":cur_raw_facts, "subject":s})

    save_data = {'all_facts':all_facts, 'all_raw_facts':all_raw_facts}

    with open(args.save_path, "wb") as f:

        pickle.dump(save_data, f)

    print(all_facts)
