import os
import re
import json
import base64
import tempfile
import httpx
import asyncio
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright
from openai import OpenAI

# ============================================
# CONFIGURATION
# ============================================
EXPECTED_SECRET = os.getenv("QUIZ_SECRET", "Key1234")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MY_EMAIL = os.getenv("MY_EMAIL", "24ds3000058@ds.study.iitm.ac.in")

# Configure OpenAI
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="LLM Quiz Solver", version="1.0.0")

# ============================================
# MODELS
# ============================================
class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# ============================================
# HEALTH CHECK
# ============================================
@app.get("/")
async def root():
    return {"status": "running", "message": "LLM Quiz Solver API is active"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# ============================================
# MAIN QUIZ ENDPOINT
# ============================================
@app.post("/")
async def quiz_handler_root(payload: QuizRequest):
    return await process_quiz(payload)

@app.post("/quiz")
async def quiz_handler(payload: QuizRequest):
    return await process_quiz(payload)

async def process_quiz(payload: QuizRequest):
    """Main quiz processing logic"""
    # 1. Validate secret
    if payload.secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # 2. Process the quiz URL
    try:
        result = await solve_quiz(payload.url, payload.email, payload.secret)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "status": "failed"},
            status_code=200  # Return 200 as per spec, include error in body
        )

# ============================================
# QUIZ SOLVER
# ============================================
async def solve_quiz(quiz_url: str, email: str, secret: str) -> dict:
    """
    Main quiz solving function that:
    1. Scrapes the quiz page
    2. Analyzes the question using LLM
    3. Solves it and submits the answer
    """
    # Scrape the quiz page
    page_content = await scrape_page(quiz_url)

    # Use LLM to analyze the question and determine what to do
    analysis = await analyze_question(page_content, quiz_url)

    # Execute the solution based on analysis
    answer = await execute_solution(analysis, page_content, quiz_url)

    # Find the submit URL from the page content
    submit_url = extract_submit_url(page_content)

    if not submit_url:
        return {"error": "Could not find submit URL", "page_content": page_content[:500]}

    # Submit the answer
    submit_payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer
    }

    async with httpx.AsyncClient(timeout=60.0) as client_http:
        response = await client_http.post(submit_url, json=submit_payload)
        result = response.json()

    # If there's a new URL, we could process it (but return for now)
    return {
        "submitted": True,
        "answer": answer,
        "response": result
    }

# ============================================
# PAGE SCRAPER
# ============================================
async def scrape_page(url: str) -> str:
    """Scrape a JavaScript-rendered page using Playwright"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            # Wait for any dynamic content
            await page.wait_for_timeout(2000)

            # Get the full page content
            content = await page.content()

            # Also get text content for easier parsing
            text_content = await page.evaluate("() => document.body.innerText")

            # Get all links
            links = await page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a')).map(a => ({
                    href: a.href,
                    text: a.innerText
                }));
            }""")

            # Get all pre/code blocks
            code_blocks = await page.evaluate("""() => {
                return Array.from(document.querySelectorAll('pre, code')).map(el => el.innerText);
            }""")

            result = {
                "html": content,
                "text": text_content,
                "links": links,
                "code_blocks": code_blocks,
                "url": url
            }

            return json.dumps(result)

        finally:
            await browser.close()

# ============================================
# LLM HELPER
# ============================================
def call_llm(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """Call OpenAI API"""
    if not client:
        return ""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM call failed: {e}")
        return ""

# ============================================
# QUESTION ANALYZER (LLM)
# ============================================
async def analyze_question(page_content: str, quiz_url: str) -> dict:
    """Use OpenAI to analyze the question and determine solution approach"""

    if not client:
        # Fallback to rule-based analysis
        return rule_based_analysis(page_content)

    prompt = f"""Analyze this quiz question and determine how to solve it.

Page content:
{page_content[:15000]}

Determine:
1. What type of question is this? (pdf_analysis, web_scraping, api_call, data_analysis, visualization, text_processing, calculation, other)
2. What data needs to be fetched? (URLs, API endpoints)
3. What operation needs to be performed?
4. What format should the answer be in? (number, string, boolean, json, base64_image)

Respond in JSON format:
{{
    "question_type": "...",
    "question_text": "the actual question being asked",
    "data_sources": ["url1", "url2"],
    "operation": "description of what to do",
    "answer_format": "...",
    "specific_instructions": "any specific details needed"
}}

Only respond with valid JSON, no other text."""

    try:
        response_text = call_llm(prompt)

        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return rule_based_analysis(page_content)
    except Exception as e:
        print(f"LLM analysis failed: {e}")
        return rule_based_analysis(page_content)

def rule_based_analysis(page_content: str) -> dict:
    """Fallback rule-based question analysis"""
    content = json.loads(page_content) if isinstance(page_content, str) else page_content
    text = content.get("text", "") if isinstance(content, dict) else str(content)

    analysis = {
        "question_type": "unknown",
        "question_text": text[:500],
        "data_sources": [],
        "operation": "",
        "answer_format": "string",
        "specific_instructions": ""
    }

    # Detect PDF questions
    if ".pdf" in text.lower() or "pdf" in text.lower():
        analysis["question_type"] = "pdf_analysis"
        # Extract PDF URLs
        links = content.get("links", []) if isinstance(content, dict) else []
        for link in links:
            if ".pdf" in link.get("href", "").lower():
                analysis["data_sources"].append(link["href"])

    # Detect API questions
    elif "api" in text.lower() or "endpoint" in text.lower():
        analysis["question_type"] = "api_call"

    # Detect calculation questions
    elif any(word in text.lower() for word in ["sum", "average", "count", "total", "calculate"]):
        analysis["question_type"] = "calculation"
        analysis["answer_format"] = "number"

    # Detect visualization questions
    elif any(word in text.lower() for word in ["chart", "plot", "graph", "visualize", "image"]):
        analysis["question_type"] = "visualization"
        analysis["answer_format"] = "base64_image"

    return analysis

# ============================================
# SOLUTION EXECUTOR
# ============================================
async def execute_solution(analysis: dict, page_content: str, quiz_url: str) -> Any:
    """Execute the solution based on the analysis"""

    question_type = analysis.get("question_type", "unknown")

    if question_type == "pdf_analysis":
        return await solve_pdf_question(analysis, page_content)
    elif question_type == "api_call":
        return await solve_api_question(analysis, page_content)
    elif question_type == "calculation":
        return await solve_calculation_question(analysis, page_content)
    elif question_type == "visualization":
        return await solve_visualization_question(analysis, page_content)
    elif question_type == "web_scraping":
        return await solve_scraping_question(analysis, page_content)
    else:
        # Use LLM to solve directly
        return await solve_with_llm(analysis, page_content)

# ============================================
# SPECIFIC SOLVERS
# ============================================
async def solve_pdf_question(analysis: dict, page_content: str) -> Any:
    """Solve PDF-related questions"""
    import pdfplumber

    # Get PDF URL
    pdf_urls = analysis.get("data_sources", [])
    if not pdf_urls:
        # Try to extract from page content
        content = json.loads(page_content) if isinstance(page_content, str) else page_content
        links = content.get("links", [])
        for link in links:
            if ".pdf" in link.get("href", "").lower():
                pdf_urls.append(link["href"])

    if not pdf_urls:
        return "Could not find PDF URL"

    pdf_url = pdf_urls[0]

    # Download PDF
    async with httpx.AsyncClient(timeout=60.0) as client_http:
        response = await client_http.get(pdf_url)
        pdf_content = response.content

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_content)
        temp_path = f.name

    try:
        # Extract text and tables from PDF
        with pdfplumber.open(temp_path) as pdf:
            all_text = ""
            all_tables = []

            for page in pdf.pages:
                all_text += page.extract_text() or ""
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)

        # Use LLM to answer the question based on PDF content
        if client:
            return await answer_pdf_question_with_llm(
                analysis.get("question_text", ""),
                all_text,
                all_tables
            )
        else:
            # Simple extraction - try to find numbers and sum them
            numbers = re.findall(r'\d+\.?\d*', all_text)
            if numbers:
                return sum(float(n) for n in numbers[:10])
            return all_text[:200]

    finally:
        os.unlink(temp_path)

async def answer_pdf_question_with_llm(question: str, text: str, tables: list) -> Any:
    """Use LLM to answer a question about PDF content"""
    tables_str = json.dumps(tables, indent=2) if tables else "No tables found"

    prompt = f"""Based on this PDF content, answer the question.

Question: {question}

PDF Text Content:
{text[:10000]}

Tables found:
{tables_str[:5000]}

Provide ONLY the answer, no explanation. If it's a number, provide just the number. If it's text, provide just the text.
If asked for a sum, calculate and provide the numeric result."""

    answer = call_llm(prompt)

    # Try to convert to number if it looks like one
    try:
        # Remove any commas from numbers
        clean_answer = answer.replace(',', '').strip()
        if '.' in clean_answer:
            return float(clean_answer)
        return int(clean_answer)
    except:
        return answer

async def solve_api_question(analysis: dict, page_content: str) -> Any:
    """Solve API-related questions"""
    # Extract API details from the question
    content = json.loads(page_content) if isinstance(page_content, str) else page_content

    # Use LLM to determine API call details
    if client:
        prompt = f"""Extract the API details from this question:
{content.get('text', '')}

Provide a JSON response with:
{{
    "url": "the API endpoint URL",
    "method": "GET or POST",
    "headers": {{}},
    "body": null or {{}}
}}

Only respond with valid JSON."""

        response_text = call_llm(prompt)

        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            api_details = json.loads(json_match.group())

            async with httpx.AsyncClient(timeout=60.0) as client_http:
                if api_details.get("method", "GET").upper() == "POST":
                    resp = await client_http.post(
                        api_details["url"],
                        headers=api_details.get("headers", {}),
                        json=api_details.get("body")
                    )
                else:
                    resp = await client_http.get(
                        api_details["url"],
                        headers=api_details.get("headers", {})
                    )

                return resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text

    return "Could not determine API details"

async def solve_calculation_question(analysis: dict, page_content: str) -> Any:
    """Solve calculation questions"""
    content = json.loads(page_content) if isinstance(page_content, str) else page_content
    text = content.get("text", "") if isinstance(content, dict) else str(content)

    if client:
        prompt = f"""Solve this calculation problem:

{text}

Provide ONLY the numeric answer, nothing else. No explanation, no units, just the number."""

        answer = call_llm(prompt)

        # Extract number from response
        numbers = re.findall(r'-?\d+\.?\d*', answer)
        if numbers:
            num = numbers[0]
            return float(num) if '.' in num else int(num)

    return 0

async def solve_visualization_question(analysis: dict, page_content: str) -> Any:
    """Solve visualization questions - generate charts"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io

    content = json.loads(page_content) if isinstance(page_content, str) else page_content
    text = content.get("text", "") if isinstance(content, dict) else str(content)

    if client:
        prompt = f"""Generate Python matplotlib code to create the visualization described:

{text}

Provide ONLY the Python code, no explanation. The code should:
1. Create the plot
2. Save to a BytesIO buffer
3. The figure variable should be named 'fig'

Example format:
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# ... plot code
"""

        code = call_llm(prompt)

        # Extract code block if present
        code_match = re.search(r'```python\n(.*?)```', code, re.DOTALL)
        if code_match:
            code = code_match.group(1)

        # Execute the code safely
        try:
            local_vars = {}
            exec(code, {"plt": plt, "np": __import__("numpy")}, local_vars)

            fig = local_vars.get("fig", plt.gcf())

            # Save to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)

            # Convert to base64
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            plt.close('all')
            return f"Visualization error: {e}"

    return "Visualization not supported without LLM"

async def solve_scraping_question(analysis: dict, page_content: str) -> Any:
    """Solve web scraping questions"""
    data_sources = analysis.get("data_sources", [])

    if not data_sources:
        content = json.loads(page_content) if isinstance(page_content, str) else page_content
        links = content.get("links", [])
        data_sources = [link["href"] for link in links if link.get("href")]

    results = []
    for url in data_sources[:3]:  # Limit to 3 sources
        try:
            scraped = await scrape_page(url)
            results.append(scraped)
        except:
            continue

    if client and results:
        prompt = f"""Based on the scraped content, answer the question:

Question: {analysis.get('question_text', '')}

Scraped content:
{json.dumps(results)[:10000]}

Provide ONLY the answer, no explanation."""

        return call_llm(prompt)

    return results[0] if results else "No data scraped"

async def solve_with_llm(analysis: dict, page_content: str) -> Any:
    """Generic LLM solver for unknown question types"""
    if not client:
        return "LLM not configured"

    content = json.loads(page_content) if isinstance(page_content, str) else page_content
    text = content.get("text", "") if isinstance(content, dict) else str(content)

    prompt = f"""Solve this quiz question:

{text}

Provide ONLY the answer in the format requested. No explanation."""

    answer = call_llm(prompt)

    # Try to parse as JSON if it looks like JSON
    if answer.startswith('{') or answer.startswith('['):
        try:
            return json.loads(answer)
        except:
            pass

    # Try to convert to number
    try:
        clean = answer.replace(',', '').strip()
        if '.' in clean:
            return float(clean)
        return int(clean)
    except:
        return answer

# ============================================
# UTILITY FUNCTIONS
# ============================================
def extract_submit_url(page_content: str) -> Optional[str]:
    """Extract the submit URL from page content"""
    content = json.loads(page_content) if isinstance(page_content, str) else page_content

    # Look in code blocks first
    code_blocks = content.get("code_blocks", [])
    for block in code_blocks:
        # Look for URLs in JSON-like structures
        urls = re.findall(r'https?://[^\s"\'<>]+', block)
        for url in urls:
            if "submit" in url.lower() or "answer" in url.lower():
                return url.rstrip('",}')

    # Look in text content
    text = content.get("text", "")
    urls = re.findall(r'https?://[^\s"\'<>]+', text)
    for url in urls:
        if "submit" in url.lower() or "answer" in url.lower():
            return url.rstrip('",}')

    # Look in HTML
    html = content.get("html", "")
    urls = re.findall(r'https?://[^\s"\'<>]+', html)
    for url in urls:
        if "submit" in url.lower():
            return url.rstrip('",}')

    # Return any URL that's not the quiz URL itself
    all_urls = re.findall(r'https?://[^\s"\'<>]+', str(content))
    for url in all_urls:
        if "submit" in url or "answer" in url:
            return url.rstrip('",}')

    return None

# ============================================
# ERROR HANDLERS
# ============================================
@app.exception_handler(json.JSONDecodeError)
async def json_exception_handler(request: Request, exc: json.JSONDecodeError):
    return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=200, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
