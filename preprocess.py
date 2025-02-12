import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from llm_helper import llm


def process_posts(raw_file_path, processed_file_path="data/processed_posts.json"):
    enriched_posts = []
    with open(raw_file_path, encoding='utf-8') as file:
        posts = json.load(file)
        for post in posts:
            metadata = extract_metadata(post['post_text'])
            post_with_metadata = post | metadata
            enriched_posts.append(post_with_metadata)

    unified_tags = get_unified_tags(enriched_posts)  

    for post in enriched_posts:
        current_tags = post['tags']
        new_tags = {unified_tags.get(tag, tag) for tag in current_tags}  # Ensuring existing tags are retained
        post['tags'] = list(new_tags)

    with open(processed_file_path, encoding='utf-8', mode="w") as outfile:
        json.dump(enriched_posts, outfile, indent=4)


def get_unified_tags(posts_with_metadata):
    unique_tags = set()
    for post in posts_with_metadata:
        unique_tags.update(post['tags'])

    unique_tags_list = ', '.join(unique_tags)  # Used `unique_tags`, not `unified_tags`

    template = '''I will give you a list of tags. You need to unify tags with the following requirements:
    1. Tags are unified and merged to create a shorter list.
       Example 1: "Jobseeker", "Job Hunting" → "Job Search".
       Example 2: "Motivation", "Inspiration", "Drive" → "Motivation".
       Example 3: "Personal Growth", "Personal Development", "Self Improvement" → "Self Improvement".
       Example 4: "Scam Alert", "Job Scam" → "Scams".
    2. Each tag should follow the title case convention, e.g., "Motivation", "Job Search".
    3. Output should be a JSON object with no preamble.
    4. The output should map original tags to unified tags, e.g.,
       {{"Jobseeker": "Job Search", "Job Hunting": "Job Search", "Motivation": "Motivation"}}
    
    Here is the list of tags:
    {tags}
    '''
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke({'tags': unique_tags_list})  # Fixed input key

    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse tags.")
    
    return res  # Ensuring function returns the parsed response


def extract_metadata(post):
    template = '''
    You are given a LinkedIn post. You need to extract the number of lines, language of the post, and tags.
    1. Return a valid JSON with no preamble.
    2. JSON object should have exactly three keys: line_count, language, and tags.
    3. The "tags" key should be an array with a maximum of two text tags.
    4. Language should be either "English" or "Hinglish" (Hindi + English).
    
    Here is the actual post on which you need to perform this task:
    {post}
    '''
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke({'post': post})

    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse metadata.")
    
    return res


if __name__ == "__main__":
    process_posts("data/raw_posts.json", "data/processed.json")
