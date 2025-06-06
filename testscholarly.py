from scholarly import scholarly
 
title = 'deepseek-r1, papers'
search_query = scholarly.search_single_pub(title)
pub_info = f"Title: {search_query['bib'].get('title', 'N/A')} | Authors: {', '.join(search_query['bib'].get('author', ['N/A']))[:50]}... | Year: {search_query['bib'].get('pub_year', 'N/A')} | Citations: {search_query.get('num_citations', 0)}"
print(f"找到论文: {pub_info}")
