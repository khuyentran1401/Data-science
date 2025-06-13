# How to Write Good Technical Articles

Articles put useful information inside other people's heads. Follow these tips to write better articles.

## Principles of Good Technical Articles

- **Easy to skim**: Few readers read linearly from top to bottom. They'll jump around, trying to assess which bit solves their problem, if any.
- **Broadly helpful**: The article should be helpful to a wide range of readers, including those who are new to the topic.
- **Clear and concise**: The article should be clear and concise, with a focus on the most important information.

## General Tips

### Don't Tell, Show

Graphics and code snippets are more effective than text. Whenever possible, use them to explain your points instead of lengthy paragraphs.

❌ Don't: Use only text to explain.

> DuckDB is a fast, in-process SQL OLAP database management system. It supports standard SQL queries on Parquet and CSV files, and provides seamless integration with Python DataFrames. DuckDB is useful for analytics workloads on local data without needing a server.

✅ Do: Use both text and code snippets to explain.

> You can query a Parquet file directly using DuckDB with a single line of SQL:
>
>```python
>import duckdb
>duckdb.query("SELECT COUNT(*) FROM 'data.parquet'").show()
>```
>
> This runs a SQL query on a local Parquet file without needing to load it into memory first.

### Use Action Verbs

Use action verbs instead of passive voice.

❌ Don't:
```
SQL operations on DataFrames are provided by DuckDB without server setup.
```

✅ Do:
```
DuckDB provides SQL operations on DataFrames without server setup.
```

### Know Your Audience

Understanding your audience is crucial for effective technical writing. Before writing, consider:

- Their technical background and experience level
- What problems they're trying to solve
- What information they need to succeed

For CodeCut articles, we write for data scientists who:

- Are proficient in Python
- Need to learn new tools quickly
- Want practical, working examples

Focus on delivering exactly what they need - no more, no less. Cut any content that doesn't directly help them solve their problem.

### Keep Paragraphs Short

Keep paragraphs short and focused. Opt for short paragraphs over long paragraphs that deliver the same information.

For step-by-step instructions, use bullet points instead of paragraphs to improve readability and make the sequence clear.

❌ Don't: Use long paragraphs.
```
Feature selection is an essential part of the machine learning pipeline, and depending on the data type, domain knowledge, and modeling goals, different methods such as mutual information, recursive feature elimination, or embedded methods can be used to improve model performance and interpretability.
```

✅ Do: Use short paragraphs.
```
Feature selection improves model performance and interpretability by removing irrelevant variables.
```

### Begin Sections with Self-Contained Preview

When readers skim, they focus on the first word, line, and sentence of each section. Start sections with sentences that make sense on their own, without relying on earlier content.

❌ Don't:
```
With the previous steps completed, let's now explore hyperparameter tuning.
```

✅ Do:
```
Hyperparameter tuning is a crucial step to improve model performance after initial training.
```


### Avoid Left-Branching Sentences

Avoid left-branching sentences as they force readers to hold information in memory until the end, which can be especially taxing.

❌ Don't: Use left-branching sentences.

```
You need historical sales data, holiday indicators, weather variables, and promotional events to build an accurate time series forecast.
```

✅ Do: Use right-branching sentences.

```
To build an accurate time series forecast, you need historical sales data, holiday indicators, weather variables, and promotional events.
```


### Be Consistent

Be consistent with formatting and naming: if you use Title Case, use it everywhere.

### Don't Tell Readers What They Think or What to Do

Avoid telling the reader what they think or what to do. This can annoy readers or undermine credibility.

❌ Don't: Presume the reader's thoughts or intentions.

```
Now you probably want to understand how to apply quantile forecasts.
```

✅ Do: Use neutral, direct phrasing.

```
To apply quantile forecasts, …
```

### Explain Things Simply

Explain things more simply than you think you need to. Many readers might not speak English as a first language. Many readers might be really confused about technical terminology and have little excess brainpower to spend on parsing English sentences.

❌ Don't: Use complex language.

```
DuckDB implements ACID-compliant transaction management mechanisms. The following elucidates the fundamental properties of ACID transactions:

- **Atomicity**: The transaction execution adheres to an all-or-nothing paradigm, wherein the entire sequence of operations must either culminate in successful completion or result in a complete rollback to the initial state, ensuring data integrity through transactional boundaries.

- **Consistency**: The database system enforces a comprehensive set of integrity constraints and business rules throughout the transaction lifecycle, maintaining a valid and coherent state that satisfies all predefined invariants and validation criteria.

- **Isolation**: Concurrent transactions operate within distinct execution contexts, preventing any form of interference or data corruption through sophisticated concurrency control mechanisms that maintain transaction independence.

- **Durability**: Once a transaction reaches the committed state, its effects become permanently persisted in the database, withstanding any subsequent system failures, crashes, or power outages through robust persistence mechanisms.
```

✅ Do: Use simple language.

```
DuckDB supports ACID transactions on your data. Here are the properties of ACID transactions:

- **Atomicity**: The transaction either completes entirely or has no effect at all. If any operation fails, all changes are rolled back.
- **Consistency**: The database maintains valid data by enforcing all rules and constraints throughout the transaction.
- **Isolation**: Transactions run independently without interfering with each other
- **Durability**: Committed changes are permanent and survive system failures
```

### Avoid Abbreviations

Write things out. The cost to experts is low and the benefit to beginners is high.

❌ Don't: Use abbreviations.

"RAG"

✅ Do: Write things out.

"Retrieval-augmented generation"

## About This Guideline

This guideline is written by [Khuyen Tran](https://www.linkedin.com/in/khuyen-tran-1401/), the founder of [CodeCut](https://codecut.ai). She has written hundreds of articles on data science and machine learning.

If you are interested in contributing to CodeCut, please refer to [Contribution Guidelines](contribution.md).