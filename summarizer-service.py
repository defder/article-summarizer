from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


class Article(BaseModel):
    text: str


@app.get("/")
def status_check():
    return "Article Summarizer active."


@app.post("/summarizer")
def summarize(article: Article):
    try:
        result = summarizer(article.text, max_length=150, min_length=40, do_sample=False)
        return {"summary": result[0]["summary_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
