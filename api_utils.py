

def get_doc_gen_prompt(doc_type, facts):

    facts_text = '\n'.join(facts)
    doc_gen_prompt = f"""You must generate a fake document in the style of {doc_type}. This document should contain all of the following facts:\n{facts_text}

    These facts should appear naturally throughout the document. Only output the content of the document - do not output anything else.
    """

    return doc_gen_prompt


def create_messages(content, nodes=False):
    '''
    messages = [
    # 0.  Guard-rail / “only JSON” rule
    
    # 1.  One-shot EXAMPLE ── user gives a short token list
    {"role": "user", "content": EXAMPLE_INPUT_1},
    # 2.  One-shot EXAMPLE ── assistant shows the expected JSON
    {"role": "assistant", "content": str(EXAMPLE_OUTPUT_1)},

    {"role": "user", "content": EXAMPLE_INPUT_2},
    # 2.  One-shot EXAMPLE ── assistant shows the expected JSON
    {"role": "assistant", "content": str(EXAMPLE_OUTPUT_2)},

    {"role": "user", "content": EXAMPLE_INPUT_3},

    {"role": "assistant", "content": str(EXAMPLE_OUTPUT_3)},
    # 3.  Your real user query
    {"role": "user", "content": REAL_USER_PROMPT}
    ]
    '''

    # zero shot for now

    messages = [{"role": "user", "content": content[-1]}]
    
    return messages


def api_call(messages, client, api_model):

    resp = client.chat.completions.create(
        model            = api_model,
        messages=messages
    )

    return resp.choices[0].message.content
