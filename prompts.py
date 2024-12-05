def get_query_prompt(code_snippet: str, search_results: str, seaching_result_non_vul: str) -> str:
    return f"""
    You are an expert in Software Security. You will be given a code snippet and a list of search results from history code database.
    Your aim is to identify whether the code snippet contains a security vulnerability or not. If it does, you should also identify the CWE ID of the vulnerability.
    
    Think like this:
    1. Is the code snippet vulnerable? Vulnerable means that the code snippet contains a security vulnerability introduced by the given code snippet.
    2. If the code snippet is vulnerable, what is the CWE ID of the vulnerability?
    3. If the code snippet is not vulnerable, ONLY output **false** in plain text.
    4. If the code snippet is vulnerable, output **true** and the CWE ID in plain text.
    
    Here are search results from history code database (all codes are vulnerable), query is the input code snippet:\n
    {search_results}
    \n
    
    Here are search results from history code database (all codes are non-vulnerable), query is the input code snippet:\n
    {seaching_result_non_vul}
    \n
    
    Input Code Snippet:
    \n
    {code_snippet}
    \n
    
    ONLY output **true** or **false** in plain text, and the CWE ID if the code snippet is vulnerable. No extra information is needed.
    """


def get_vote_prompt(all_results: str) -> str:
    return f"""
    You are an expert in Software Security. You will be given a list of responses from different experts to a single code suspecious snippet.
    Your aim is to vote for the most likely response.
    
    Think like this:
    1. Read all the responses carefully.
    2. Output the majority response.
    
    Here are the responses from different experts:\n
    {all_results}
    \n
    
    You should ONLY output:
    1. **true** and CWE ID (i.e. **true**, CWE-119) if the majority of responses are **true** and the CWE ID is the same in the majority of responses.
    2. **false** if the majority of responses are **false**
    
    No extra information is needed. No extra information is needed. No code snippet is provided.
    """