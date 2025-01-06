import arxiv
import pandas as pd
from transformers import pipeline
## Query to fetch AI-related papers

query = 'ai OR artificial intelligence OR machine learning'
search = arxiv.Search(query=query, max_results=10, sort_by=arxiv.SortCriterion.SubmittedDate)

# Fetch papers
papers = []
for result in search.results():
    papers.append({
      'published': result.published,
        'title': result.title,
        'abstract': result.summary,
        'categories': result.categories
    })

# Convert to DataFrame
df = pd.DataFrame(papers)

pd.set_option('display.max_colwidth', None)
print(df.head(10))

# Example abstract from API
abstract = df['abstract'][0]

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Summarization
summarization_result = summarizer(abstract)

print(summarization_result[0]['summary_text'])